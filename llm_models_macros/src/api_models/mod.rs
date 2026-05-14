mod data;
mod model;
mod provider;

use data::*;
use model::{DeCloudLlms, MacroCloudLlm, MacroCloudLlms};
use proc_macro2::{Ident, TokenStream};
use provider::MacroCloudLlmProvider;
use quote::{format_ident, quote};

use super::*;

pub fn generate_api_providers_and_models(output_path: std::path::PathBuf) {
    model::generate(&output_path);
    provider::generate(&output_path);
}
