//! Server Process - Guard
//! ====================
//!
//! Launches and supervises a *single* instance of the inference-server
//! binary, guaranteeing robust clean-up across Linux/BSD, macOS and
//! Windows.
//!
//! Key responsibilities
//! --------------------
//! * **Spawn** the server with validated command-line arguments.
//! * **Persist** a unique *pid-file* to prevent concurrent launches.
//! * **Contain** the entire process tree using the best mechanism per
//!   platform (process-group, anonymous pipe, or Windows job object).
//! * **Clean up** gracefully (`SIGTERM`/`Kill` -> timeout -> force kill)
//!   and delete artefacts on [`stop`] or on `Drop`.
//!
//! The implementation favours **RAII** over global state: if a test or a
//! CI job crashes, the OS still reaps the child because the guard's
//! destructor runs when the process aborts.

use std::{
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use wait_timeout::ChildExt;

use super::{error::*, kill::*, pid::*, *};
use crate::server::types::start_args::ServerArgs;

/// Pattern to match in CLI output: `I llama_context: n_ctx         = 12032`
/// The number after `=` is the inferred context window.
const N_CTX_PATTERN: &str = "llama_context: n_ctx";
/// Timeout for waiting to read the n_ctx from server output.
const N_CTX_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// RAII handle owning a running server instance.
///
/// While this guard is alive:
/// * the child process is guaranteed to stay running, and
/// * external launches are prevented by the on-disk *pid-file*.
///
/// Dropping the guard (or calling [`stop`]) kills **the whole process
/// tree** and removes the pid-file, regardless of the platform-specific
/// containment method.
///
/// **Platform notes**
/// | OS | Containment method | Extra field |
/// |----|--------------------|-------------|
/// | Linux/BSD | Process-group + `prctl(PDEATHSIG)` | - |
/// | macOS | Anonymous pipe (*lifeline*) | `_life` |
/// | Windows | NT Job object (*kill-on-close*) | `_job` |
#[derive(Debug)]
pub struct ServerProcessGuard {
    /// Handle to the child process; `None` once reaped or transferred.
    child: std::sync::RwLock<Option<Child>>,
    /// Absolute path to the pid-file created on successful launch.
    pidfile: PathBuf,
    /// Inferred context window from server CLI output, if found.
    inferred_ctx: Arc<Mutex<Option<u32>>>,
    /// Windows-only: RAII wrapper around the Job object.
    _job: attach::JobGuard,
    /// macOS-only: write-end of the lifeline pipe.
    _life: attach::Lifeline,
}

impl ServerProcessGuard {
    #[must_use]
    pub fn new(
        bin_path: &Path,
        bin_dir: &Path,
        pidfile_path: &Path,
        server_args: &ServerArgs,
    ) -> Result<Self> {
        debug_assert!(bin_path.is_file(), "Server binary path must be a file");
        debug_assert!(
            bin_dir.is_dir(),
            "Server binary directory must be a directory"
        );
        // debug_assert!(server_args.validate().is_ok(), "ServerArgs must be valid");

        let mut cmd = Command::new(bin_path);
        cmd.current_dir(bin_dir);
        server_args.apply_env_to_command(&mut cmd);

        use cmdstruct::Command as _;
        cmd.args(server_args.command().get_args());

        let mut pidfile_handle = create_pidfile(&pidfile_path)?;

        let guard = match Self::new_inner(cmd, pidfile_path.to_path_buf()) {
            Ok(g) => g,
            Err(e) => {
                // start-up failed -> remove the empty pid-file we just created
                let _ = std::fs::remove_file(&pidfile_path);
                return Err(e);
            }
        };
        let pid = guard.pid();
        if let Err(e) = writeln!(&mut pidfile_handle, "{}", pid) {
            let _ = std::fs::remove_file(&pidfile_path); // rollback on failure
            return Err(ProcessError::CommandFailed {
                action: "write pidfile",
                source: e.into(),
            });
        }

        // The kernel never gives PID 0 to user processes; catching a hypothetical failure from spawn early costs nothing in release builds.fs
        debug_assert!(pid > 0, "OS returned an invalid PID (0)");

        debug_assert!(
            matches!(
                std::fs::read_to_string(&guard.pidfile),
                Ok(ref contents) if contents.trim() == pid.to_string()
            ),
            "pid-file was not created or contains the wrong PID"
        );

        Ok(guard)
    }

    fn new_inner(cmd: Command, pidfile: PathBuf) -> Result<Self> {
        crate::info!("Starting ServerProcessGuard with command: {:?}", cmd);
        let guard = attach::attach(cmd, pidfile)?;
        Ok(guard)
    }

    /// Attempt a best-effort, idempotent shutdown:
    /// 1. *Polite* -> send `SIGTERM` / `TerminateProcess`.
    /// 2. Wait up to [`POLITE_WAIT`].
    /// 3. *Force* -> `kill(-9)` / `Taskkill`.
    /// 4. Delete the pid-file (ignoring permissions errors).
    ///
    /// Logging is performed at **info** level for graceful exits and at
    /// **error** level for failures.
    pub fn stop(&self) -> Result<()> {
        match self.kill_child() {
            Ok(()) => {
                match std::fs::remove_file(&self.pidfile) {
                    Ok(()) => {
                        debug_assert!(
                            !self.pidfile.exists(),
                            "pid-file should be gone after successful remove_file"
                        );
                    }
                    Err(e) => {
                        crate::error!("Failed to remove pidfile {:?}: {}", self.pidfile, e);
                    }
                }

                Ok(())
            }
            Err(e) => {
                crate::error!("Failed to stop server process: {}", e);
                Err(e)
            }
        }
    }

    fn kill_child(&self) -> Result<()> {
        let Some(mut child) = self
            .child
            .write()
            .expect("Failed to acquire write lock")
            .take()
        else {
            return Ok(());
        };

        if child
            .try_wait()
            .map_err(|e| ProcessError::CommandFailed {
                action: "get exit status",
                source: e.into(),
            })?
            .is_some()
        {
            return Ok(());
        }

        if let Err(e) = kill_pid(child.id() as u32) {
            crate::error!("Failed to send TERM to PID {}: {}", child.id(), e);
        }

        if let Some(status) =
            child
                .wait_timeout(POLITE_WAIT)
                .map_err(|e| ProcessError::CommandFailed {
                    action: "polite wait for exit",
                    source: e.into(),
                })?
        {
            crate::info!("Server process exited gracefully with status: {}", status);
            return Ok(());
        }

        child.kill().map_err(|e| match e.kind() {
            std::io::ErrorKind::PermissionDenied => ProcessError::PermissionDenied {
                action: "force-kill",
                source: e.into(),
            },
            _ => ProcessError::CommandFailed {
                action: "force-kill",
                source: e.into(),
            },
        })?;

        match child
            .wait_timeout(Duration::from_secs(FORCE_KILL_TIMEOUT_SECS))
            .map_err(|e| ProcessError::CommandFailed {
                action: "wait after force-kill",
                source: e.into(),
            })? {
            Some(status) => {
                crate::info!("Server force-killed; exit status {status}");
                Ok(())
            }
            None => Err(ProcessError::TerminationTimeout {
                operation: "force-kill",
                elapsed: Duration::from_secs(FORCE_KILL_TIMEOUT_SECS),
                leftovers: vec![child.id()],
            }),
        }
    }

    pub fn pid(&self) -> u32 {
        self.child
            .read()
            .expect("Failed to acquire read lock")
            .as_ref()
            .map(|c| c.id())
            .expect("Child process not spawned")
    }

    /// Returns the inferred context window from the server CLI output.
    ///
    /// This value is extracted from lines like:
    /// `I llama_context: n_ctx         = 12032`
    ///
    /// Returns `None` if the pattern was not found in the CLI output
    /// (e.g., if the server output format changed or was not captured).
    pub fn inferred_ctx_window(&self) -> Option<u32> {
        *self
            .inferred_ctx
            .lock()
            .expect("Failed to lock inferred_ctx")
    }

    #[cfg(all(test, target_os = "linux"))]
    pub fn dummy() -> Self {
        Self {
            child: None.into(),
            pidfile: PathBuf::from("/dummy/path"),
            inferred_ctx: Arc::new(Mutex::new(None)),
            _job: (),
            _life: (),
        }
    }

    #[cfg(all(test, target_os = "macos"))]
    pub fn dummy() -> Self {
        Self {
            child: None.into(),
            pidfile: PathBuf::from("/dummy/path"),
            inferred_ctx: Arc::new(Mutex::new(None)),
            _job: (),
            _life: attach::Lifeline(None),
        }
    }

    #[cfg(all(test, windows))]
    pub fn dummy() -> Self {
        Self {
            child: None.into(),
            pidfile: PathBuf::from("/dummy/path"),
            inferred_ctx: Arc::new(Mutex::new(None)),
            _job: attach::JobGuard(None),
            _life: (),
        }
    }
}

impl Drop for ServerProcessGuard {
    fn drop(&mut self) {
        match self.stop() {
            Ok(()) => (),
            Err(e) => crate::error!("Failed to stop server process: {}", e),
        }
    }
}

/// Parse the n_ctx value from a line of server output.
///
/// Expected format: `I llama_context: n_ctx         = 12032`
/// Returns `Some(12032)` if found, `None` otherwise.
fn parse_n_ctx_from_line(line: &str) -> Option<u32> {
    if !line.contains(N_CTX_PATTERN) {
        return None;
    }

    // Split by '=' and take the right side
    line.split('=')
        .nth(1)
        .map(|value| value.trim().parse::<u32>().ok())
        .flatten()
}

/// Shared helper to read child output and extract n_ctx value.
/// This is DRY across all platforms.
///
/// Spawns a thread to stream from the reader and look for the n_ctx pattern.
/// The found value is stored in the shared `inferred_ctx` Arc<Mutex<Option<u32>>>.
///
/// Returns immediately after spawning the monitoring thread.
/// The thread will terminate when the reader reaches EOF or the process exits.
fn spawn_ctx_reader<R: std::io::Read + Send + 'static>(
    reader: R,
    inferred_ctx: Arc<Mutex<Option<u32>>>,
) {
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        let start_time = std::time::Instant::now();

        for line in reader.lines() {
            // Check for timeout
            if start_time.elapsed() > N_CTX_READ_TIMEOUT {
                crate::warn!(
                    "Timeout waiting for n_ctx in server output after {:?}",
                    N_CTX_READ_TIMEOUT
                );
                break;
            }

            match line {
                Ok(line) => {
                    if let Some(ctx) = parse_n_ctx_from_line(&line) {
                        crate::info!("Inferred context window from CLI: {}", ctx);
                        let mut guard = inferred_ctx.lock().expect("Failed to lock inferred_ctx");
                        if guard.is_none() {
                            *guard = Some(ctx);
                        }
                        // We found it, but continue reading to avoid breaking the pipe
                    }
                }
                Err(e) => {
                    crate::debug!("Error reading server output line: {}", e);
                    break;
                }
            }
        }
    });
}

// Linux / other Unix -----------------------
#[cfg(all(unix, not(target_os = "macos")))]
mod attach {
    //! Platform glue for `ServerProcessGuard`
    //!
    //! Each sub-module provides:
    //! * a platform-specific `attach` function that spawns the child and
    //!   returns a fully-initialised [`ServerProcessGuard`], and
    //! * minimal helper types (`JobGuard`, `Lifeline`) whose sole purpose is
    //!   to automate resource release via `Drop`.

    use std::os::unix::process::CommandExt;

    use nix::{
        sys::{prctl::set_pdeathsig, signal::Signal},
        unistd::{Pid, setpgid},
    };

    use super::*;

    pub type JobGuard = ();

    pub type Lifeline = ();

    pub fn attach(mut cmd: Command, pidfile: PathBuf) -> Result<ServerProcessGuard> {
        // Capture stdout and stderr to extract n_ctx
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Put the child in its own PGID **and** arm the parent-death signal.
        unsafe {
            cmd.pre_exec(|| {
                // Child becomes leader of a new process-group (PGID = its PID)
                setpgid(Pid::from_raw(0), Pid::from_raw(0))
                    .map_err(|e| std::io::Error::from_raw_os_error(e as i32))?;

                // Ensure the kernel delivers SIGTERM to the child if the parent dies
                set_pdeathsig(Some(Signal::SIGTERM))?;

                Ok(())
            })
        };

        let mut child = cmd.spawn().map_err(|e| ProcessError::CommandFailed {
            action: "spawn child process",
            source: e.into(),
        })?;

        // Shared state for inferred context window
        let inferred_ctx: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));

        // Spawn threads to read stdout and stderr, looking for n_ctx
        if let Some(stdout) = child.stdout.take() {
            spawn_ctx_reader(stdout, inferred_ctx.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_ctx_reader(stderr, inferred_ctx.clone());
        }

        Ok(ServerProcessGuard {
            child: Some(child).into(),
            pidfile,
            inferred_ctx,
            _job: (),
            _life: (),
        })
    }
}

// macOS ---------------------------------------------------------
#[cfg(target_os = "macos")]
mod attach {
    //! Platform glue for `ServerProcessGuard`
    //!
    //! Each sub-module provides:
    //! * a platform-specific `attach` function that spawns the child and
    //!   returns a fully-initialised [`ServerProcessGuard`], and
    //! * minimal helper types (`JobGuard`, `Lifeline`) whose sole purpose is
    //!   to automate resource release via `Drop`.

    use std::os::{
        fd::{AsFd, AsRawFd},
        unix::io::OwnedFd,
    };

    use nix::{
        fcntl::{FcntlArg, FdFlag, fcntl},
        unistd::pipe,
    };

    use super::*;

    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct Lifeline(pub Option<OwnedFd>);

    // No manual Drop needed - OwnedFd handles it automatically!

    pub type JobGuard = (); // no job object on mac

    pub fn attach(mut cmd: Command, pidfile: PathBuf) -> Result<ServerProcessGuard> {
        // Capture stdout and stderr to extract n_ctx
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // ✓ 1. Open the pipe (still fallible -> same error handling)
        let (r, w) = pipe().map_err(|e| ProcessError::CommandFailed {
            action: "create pipe for pre_exec",
            source: e.into(),
        })?;

        // ✓ 2. Assert they're different; ">= 0" is redundant for OwnedFd
        debug_assert!(
            r.as_raw_fd() != w.as_raw_fd(),
            "pipe() returned duplicate FDs"
        );

        // ✓ 3. fcntl works with raw FDs

        fcntl(r.as_fd(), FcntlArg::F_SETFD(FdFlag::empty())).map_err(|e| {
            ProcessError::CommandFailed {
                action: "set pipe read end flags",
                source: e.into(),
            }
        })?;

        fcntl(w.as_fd(), FcntlArg::F_SETFD(FdFlag::FD_CLOEXEC)).map_err(|e| {
            ProcessError::CommandFailed {
                action: "set pipe write end flags",
                source: e.into(),
            }
        })?;

        // ✓ 4. Pass the read end to the child
        let lifeline_fd = r.as_raw_fd();
        cmd.env("LIFELINE_FD", lifeline_fd.to_string());

        // ✓ 5. Spawn, then drop the read end in the parent
        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => {
                drop(r); // close read end in case of error
                drop(w);
                return Err(ProcessError::CommandFailed {
                    action: "spawn",
                    source: Box::new(e),
                });
            }
        };
        drop(r); // parent no longer needs it

        // ✓ 6. Keep the write end alive as Lifeline
        let owned_w = w; // already an OwnedFd; no unsafe

        // Shared state for inferred context window
        let inferred_ctx: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));

        // Spawn threads to read stdout and stderr, looking for n_ctx
        if let Some(stdout) = child.stdout.take() {
            spawn_ctx_reader(stdout, inferred_ctx.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_ctx_reader(stderr, inferred_ctx.clone());
        }

        Ok(ServerProcessGuard {
            child: Some(child).into(),
            pidfile,
            inferred_ctx,
            _job: (),
            _life: Lifeline(Some(owned_w)),
        })
    }
}

// Windows -------------------------------------------------------
#[cfg(windows)]
mod attach {
    //! Platform glue for `ServerProcessGuard`
    //!
    //! Each sub-module provides:
    //! * a platform-specific `attach` function that spawns the child and
    //!   returns a fully-initialised [`ServerProcessGuard`], and
    //! * minimal helper types (`JobGuard`, `Lifeline`) whose sole purpose is
    //!   to automate resource release via `Drop`.

    use std::os::windows::{io::AsRawHandle, process::CommandExt};

    use windows::Win32::{
        Foundation::{CloseHandle, HANDLE},
        System::{
            JobObjects::*,
            Threading::{CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW},
        },
    };

    use super::*;

    #[derive(Debug)]
    pub struct JobGuard(pub Option<HANDLE>);

    impl Drop for JobGuard {
        fn drop(&mut self) {
            if let Some(h) = self.0.take() {
                unsafe {
                    if CloseHandle(h).is_err() {
                        let err = std::io::Error::last_os_error();
                        crate::error!("Failed to close Job handle {h:?}: {err}");
                    }
                }
            }
        }
    }

    pub type Lifeline = (); // no lifeline on Windows

    pub fn attach(mut cmd: Command, pidfile: PathBuf) -> Result<ServerProcessGuard> {
        // Capture stdout and stderr to extract n_ctx
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // ---- 1. Create a job object ------------------------------------
        let hjob = unsafe {
            CreateJobObjectW(None, None).map_err(|e| ProcessError::CommandFailed {
                action: "CreateJobObjectW",
                source: Box::new(e),
            })?
        };

        // Job handle validity
        debug_assert!(!hjob.is_invalid(), "CreateJobObjectW returned NULL handle");

        let mut job_guard = JobGuard(Some(hjob));

        // ---- 2. Set "kill on job close" so dropping the handle nukes the tree ----
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        unsafe {
            SetInformationJobObject(
                hjob,
                JobObjectExtendedLimitInformation,
                &info as *const _ as _,
                std::mem::size_of_val(&info) as _,
            )
            .map_err(|e| ProcessError::CommandFailed {
                action: "SetInformationJobObject",
                source: Box::new(e),
            })?;
        }

        // ---- 3. Spawn the child process normally ------------------------
        cmd.creation_flags((CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW).0);
        let mut child = cmd.spawn().map_err(|e| ProcessError::CommandFailed {
            action: "spawn",
            source: Box::new(e),
        })?;

        // ---- 4. Put the child into the job (ignore "already in a job") ----
        let assign_res = unsafe { AssignProcessToJobObject(hjob, HANDLE(child.as_raw_handle())) };

        if let Err(e) = assign_res {
            drop(job_guard);
            return Err(ProcessError::CommandFailed {
                action: "AssignProcessToJobObject",
                source: Box::new(e),
            });
        }

        // ---- 5. Success! Transfer ownership of the job handle ----------
        // Extract the handle without dropping it
        let hjob = job_guard
            .0
            .take()
            .expect("job handle should still be present");

        // Shared state for inferred context window
        let inferred_ctx: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));

        // Spawn threads to read stdout and stderr, looking for n_ctx
        if let Some(stdout) = child.stdout.take() {
            spawn_ctx_reader(stdout, inferred_ctx.clone());
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_ctx_reader(stderr, inferred_ctx.clone());
        }

        Ok(ServerProcessGuard {
            child: Some(child).into(),
            pidfile,
            inferred_ctx,
            _job: JobGuard(Some(hjob)), // RAII: handle closed (and tree killed) on Drop
            _life: (),                  // no lifeline semantics needed on Windows
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::{
        process::{error::ProcessError, tests_helpers::*},
        types::start_args::ServerArgs,
    };
    /// Cross-platform helper: make a directory read-only so a subsequent
    /// `std::fs::remove_file` call is guaranteed to fail.
    fn make_dir_read_only(path: &std::path::Path) -> std::io::Result<()> {
        let meta = std::fs::metadata(path)?;
        let mut perms = meta.permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            perms.set_mode(0o555);
        }
        #[cfg(windows)]
        {
            perms.set_readonly(true);
        }
        std::fs::set_permissions(path, perms)
    }

    // ---------------------- Consolidated happy- / sad-paths -----------------

    /// One test that exercises every internal state branch of `ServerProcessGuard::stop()`.
    #[test]
    fn stop_variants() {
        // ---- 1. Child has already exited --------------------------------
        {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            let cmd = std::process::Command::new("true");
            #[cfg(target_os = "windows")]
            let cmd = {
                let mut c = std::process::Command::new("cmd");
                c.args(["/C", "exit", "0"]);
                c
            };

            let pf = tempfile::NamedTempFile::new().expect("tmp pidfile");
            let g = attach::attach(cmd, pf.path().into()).unwrap();
            g.child.write().unwrap().as_mut().unwrap().wait().ok(); // child finished
            assert!(g.stop().is_ok(), "stop() should succeed after exit");
        }

        // ---- 2. `self.child` is already `None` (ownership moved) --------
        {
            let td = tempfile::tempdir().unwrap();
            let pf = td.path().join("pid");
            let g = attach::attach(long_cmd(), pf).unwrap();
            let _ = g.child.write().unwrap().take(); // simulate transfer
            assert!(g.stop().is_ok(), "stop() should succeed when child == None");
        }

        // ---- 3. Child still running; must be killed gracefully/forcibly --
        {
            let td = tempfile::tempdir().unwrap();
            let pf = td.path().join("pid");
            let g = attach::attach(long_cmd(), pf).unwrap();
            assert!(g.stop().is_ok(), "stop() should succeed for live child");
        }
    }

    // ---------------------- Stand-alone negative paths ---------------------

    #[test]
    fn attach_invalid_bin_path_errors() {
        let cmd = std::process::Command::new("definitely-does-not-exist-xyz");
        let td = tempfile::tempdir().unwrap();
        let pf = td.path().join("pid");
        assert!(crate::server::process::guard::attach::attach(cmd, pf).is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn new_guard_writes_pidfile() {
        let td = tempfile::tempdir().unwrap();
        let bin_dir = td.path();
        #[cfg(target_os = "macos")]
        let bin_path = Path::new("/usr/bin/true");
        #[cfg(all(unix, not(target_os = "macos")))]
        let bin_path = Path::new("/bin/true");
        let pf = td.path().join(format!("true_write_test.pid"));

        let guard = ServerProcessGuard::new(
            bin_path,
            bin_dir,
            &pf,
            &ServerArgs::builder().hf_repo("dummy/repo").unwrap().build(),
        )
        .unwrap();

        assert!(pf.exists(), "new() must create pid-file");
        let _ = guard.stop();
    }

    #[test]
    fn guard_drop_cleans_up() {
        let td = tempfile::tempdir().unwrap();
        let pf = td.path().join(TEST_EXE).join("drop-test");
        let pid = {
            let guard = attach::attach(long_cmd(), pf.clone()).unwrap();
            guard.child.write().unwrap().as_mut().unwrap().id()
        };
        std::thread::sleep(std::time::Duration::from_secs(2));
        assert!(!pf.exists(), "pid-file should be gone after Drop");
        assert!(!pid_alive(pid).unwrap());
    }

    #[test]
    fn stop_pidfile_removal_failure_is_ok() {
        let td = tempfile::tempdir().unwrap();
        let ro_dir = td.path().join("ro");
        std::fs::create_dir(&ro_dir).unwrap();
        let pf = ro_dir.join("pid");
        std::fs::write(&pf, "").unwrap(); // ensure remove_file will be attempted
        make_dir_read_only(&ro_dir).unwrap();

        let guard = attach::attach(long_cmd(), pf).unwrap();
        assert!(
            guard.stop().is_ok(),
            "stop() should ignore remove_file errors and still return Ok"
        );
    }

    #[test]
    fn new_guard_refuses_stale_pidfile() {
        let td = tempfile::tempdir().unwrap();
        let bin_dir = td.path();
        #[cfg(target_os = "macos")]
        let bin_path = Path::new("/usr/bin/true");
        #[cfg(all(unix, not(target_os = "macos")))]
        let bin_path = Path::new("/bin/true");
        #[cfg(windows)]
        let bin_path = std::env::current_exe().unwrap();
        let pf = bin_dir.join("stale.pid");
        std::fs::write(&pf, "9999").unwrap(); // simulate stale file

        let err = ServerProcessGuard::new(
            &bin_path,
            bin_dir,
            &pf,
            &ServerArgs::builder().hf_repo("dummy/repo").unwrap().build(),
        )
        .expect_err("should fail when pid-file already exists");

        match err {
            ProcessError::CommandFailed { action, .. } => {
                assert_eq!(action, "check pidfile existence");
            }
            _ => panic!("unexpected error variant: {err:?}"),
        }
    }

    // ---------------------- n_ctx parsing tests ---------------------------

    #[test]
    fn parse_n_ctx_from_line_valid() {
        let line = "0.00.368.622 I llama_context: n_ctx         = 12032";
        assert_eq!(parse_n_ctx_from_line(line), Some(12032));
    }

    #[test]
    fn parse_n_ctx_from_line_with_different_spacing() {
        let line = "I llama_context: n_ctx = 4096";
        assert_eq!(parse_n_ctx_from_line(line), Some(4096));
    }

    #[test]
    fn parse_n_ctx_from_line_no_match() {
        let line = "Some other log line without n_ctx";
        assert_eq!(parse_n_ctx_from_line(line), None);
    }

    #[test]
    fn parse_n_ctx_from_line_empty_value() {
        let line = "I llama_context: n_ctx         = ";
        assert_eq!(parse_n_ctx_from_line(line), None);
    }

    #[test]
    fn parse_n_ctx_from_line_invalid_value() {
        let line = "I llama_context: n_ctx         = not-a-number";
        assert_eq!(parse_n_ctx_from_line(line), None);
    }

    #[test]
    fn inferred_ctx_window_initially_none() {
        let guard = ServerProcessGuard::dummy();
        assert_eq!(guard.inferred_ctx_window(), None);
    }
}
