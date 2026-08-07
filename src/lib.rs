#![forbid(unsafe_code)]

pub const VERSION: &str = match option_env!("MCODE_BUILD_VERSION") {
    Some(version) => version,
    None => env!("CARGO_PKG_VERSION"),
};

pub const USER_AGENT: &str = "mcode/0.147.0";

pub mod agent;
pub mod approval;
pub mod cli;
pub mod compaction;
pub mod config;
pub mod event;
mod highlight;
mod latex;
pub mod openai;
pub mod protocol;
pub mod sandbox;
pub mod session;
pub mod tools;
pub mod ui;
pub mod update;
pub mod web_access;
