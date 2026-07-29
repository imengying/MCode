use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-4.1";
const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionSettings {
    pub enabled: bool,
    pub reserve_tokens: u64,
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        }
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, ValueEnum,
)]
#[serde(rename_all = "lowercase")]
#[value(rename_all = "lower")]
pub enum ReasoningEffort {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    pub const ALL: [Self; 7] = [
        Self::Off,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Xhigh,
        Self::Max,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiProtocol {
    #[serde(rename = "openai-completions")]
    ChatCompletions,
    #[serde(rename = "openai-responses")]
    Responses,
}

impl ApiProtocol {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChatCompletions => "openai-completions",
            Self::Responses => "openai-responses",
        }
    }
}

impl fmt::Display for ApiProtocol {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
#[value(rename_all = "lower")]
pub enum WebSearchMode {
    #[default]
    Disabled,
    Cached,
    Live,
}

impl WebSearchMode {
    pub const ALL: [Self; 3] = [Self::Disabled, Self::Cached, Self::Live];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Cached => "cached",
            Self::Live => "live",
        }
    }

    #[must_use]
    pub const fn label_zh(self) -> &'static str {
        match self {
            Self::Disabled => "禁用",
            Self::Cached => "缓存",
            Self::Live => "实时",
        }
    }

    #[must_use]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

impl fmt::Display for WebSearchMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchProvider {
    #[default]
    Auto,
    Exa,
    Brave,
    Searxng,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WebSearchSettings {
    pub mode: WebSearchMode,
    pub provider: WebSearchProvider,
    pub allowed_domains: Vec<String>,
    pub exa_api_key: Option<String>,
    pub brave_api_key: Option<String>,
    pub searxng_base_url: Option<String>,
    pub trust_env_proxy: bool,
}

#[derive(Clone)]
pub struct ModelProfile {
    pub provider: String,
    pub id: String,
    pub name: Option<String>,
    pub api: ApiProtocol,
    pub base_url: String,
    pub api_key: Option<String>,
    pub context_window: u64,
    pub max_input_tokens: u64,
    pub reasoning: bool,
    pub compat: ModelCompat,
    default_reasoning_effort: ReasoningEffort,
    thinking_level_map: BTreeMap<ReasoningEffort, Option<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCompat {
    pub reasoning_effort: bool,
    pub usage_in_streaming: bool,
    pub strict_tools: bool,
}

impl ModelProfile {
    #[must_use]
    pub fn qualified_id(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }

    pub fn reasoning_value(&self, effort: ReasoningEffort) -> Result<Option<String>> {
        if !self.supports_reasoning(effort) {
            bail!("model {} does not support {effort} reasoning", self.id);
        }
        if effort == ReasoningEffort::Off {
            return Ok(None);
        }
        match self.thinking_level_map.get(&effort) {
            Some(Some(value)) if value.trim().is_empty() => {
                bail!(
                    "model {} maps {effort} reasoning to an empty value",
                    self.id
                )
            }
            Some(Some(value)) => Ok(Some(value.clone())),
            Some(None) => bail!("model {} does not support {effort} reasoning", self.id),
            None => bail!("model {} does not configure {effort} reasoning", self.id),
        }
    }

    #[must_use]
    pub fn supports_reasoning(&self, effort: ReasoningEffort) -> bool {
        if !self.reasoning {
            return effort == ReasoningEffort::Off;
        }
        matches!(self.thinking_level_map.get(&effort), Some(Some(_)))
    }

    #[must_use]
    pub fn supported_reasoning_efforts(&self) -> Vec<ReasoningEffort> {
        if !self.reasoning {
            return vec![ReasoningEffort::Off];
        }
        ReasoningEffort::ALL
            .into_iter()
            .filter(|effort| self.supports_reasoning(*effort))
            .collect()
    }

    #[must_use]
    pub const fn default_reasoning_effort(&self) -> ReasoningEffort {
        self.default_reasoning_effort
    }

    #[must_use]
    pub fn clamp_reasoning_effort(&self, requested: ReasoningEffort) -> ReasoningEffort {
        if self.supports_reasoning(requested) {
            return requested;
        }
        let requested_index = ReasoningEffort::ALL
            .iter()
            .position(|effort| *effort == requested)
            .unwrap_or_default();
        ReasoningEffort::ALL[requested_index..]
            .iter()
            .chain(ReasoningEffort::ALL[..requested_index].iter().rev())
            .copied()
            .find(|effort| self.supports_reasoning(*effort))
            .unwrap_or(ReasoningEffort::Off)
    }
}

#[derive(Clone)]
pub struct AppConfig {
    pub model: String,
    pub provider: Option<String>,
    pub api: ApiProtocol,
    pub reasoning_effort: ReasoningEffort,
    pub reasoning_value: Option<String>,
    pub base_url: String,
    pub api_key: Option<String>,
    pub context_window: u64,
    pub max_input_tokens: u64,
    pub supports_reasoning_effort: bool,
    pub supports_usage_in_streaming: bool,
    pub supports_strict_tools: bool,
    pub cwd: PathBuf,
    pub max_tool_turns: usize,
    pub request_timeout_secs: u64,
    pub compaction: CompactionSettings,
    pub web_search: WebSearchSettings,
    pub model_profiles: Vec<ModelProfile>,
    pub mcp_servers: Vec<McpServerConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct ConfigOverrides {
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub context_window: Option<u64>,
    pub max_input_tokens: Option<u64>,
    pub cwd: Option<PathBuf>,
    pub max_tool_turns: Option<usize>,
    pub request_timeout_secs: Option<u64>,
    pub web_search: Option<WebSearchMode>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct SettingsFile {
    #[serde(rename = "defaultProvider")]
    provider: Option<String>,
    #[serde(rename = "defaultModel")]
    model: Option<String>,
    #[serde(default, rename = "mcpServers")]
    mcp_servers: BTreeMap<String, McpServerFile>,
    #[serde(default)]
    compaction: CompactionSettingsFile,
    #[serde(rename = "webSearch")]
    web_search: Option<WebSearchMode>,
    #[serde(default, rename = "webSearchConfig")]
    web_search_config: WebSearchSettingsFile,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CompactionSettingsFile {
    enabled: Option<bool>,
    reserve_tokens: Option<u64>,
    keep_recent_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WebSearchSettingsFile {
    provider: Option<WebSearchProvider>,
    allowed_domains: Option<Vec<String>>,
    exa_api_key: Option<String>,
    brave_api_key: Option<String>,
    searxng_base_url: Option<String>,
    trust_env_proxy: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct McpServerFile {
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    enabled: Option<bool>,
}

impl SettingsFile {
    fn overlay(&mut self, other: Self) {
        if other.provider.is_some() {
            self.provider = other.provider;
        }
        if other.model.is_some() {
            self.model = other.model;
        }
        if other.compaction.enabled.is_some() {
            self.compaction.enabled = other.compaction.enabled;
        }
        if other.compaction.reserve_tokens.is_some() {
            self.compaction.reserve_tokens = other.compaction.reserve_tokens;
        }
        if other.compaction.keep_recent_tokens.is_some() {
            self.compaction.keep_recent_tokens = other.compaction.keep_recent_tokens;
        }
        if other.web_search.is_some() {
            self.web_search = other.web_search;
        }
        if other.web_search_config.provider.is_some() {
            self.web_search_config.provider = other.web_search_config.provider;
        }
        if other.web_search_config.allowed_domains.is_some() {
            self.web_search_config.allowed_domains = other.web_search_config.allowed_domains;
        }
        if other.web_search_config.exa_api_key.is_some() {
            self.web_search_config.exa_api_key = other.web_search_config.exa_api_key;
        }
        if other.web_search_config.brave_api_key.is_some() {
            self.web_search_config.brave_api_key = other.web_search_config.brave_api_key;
        }
        if other.web_search_config.searxng_base_url.is_some() {
            self.web_search_config.searxng_base_url = other.web_search_config.searxng_base_url;
        }
        if other.web_search_config.trust_env_proxy.is_some() {
            self.web_search_config.trust_env_proxy = other.web_search_config.trust_env_proxy;
        }
        self.mcp_servers.extend(other.mcp_servers);
    }
}

#[derive(Debug, Default, Deserialize)]
struct ModelsFile {
    #[serde(default)]
    providers: BTreeMap<String, ProviderFile>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProviderFile {
    base_url: Option<String>,
    api: Option<String>,
    api_key: Option<String>,
    #[serde(default)]
    compat: CompatFile,
    #[serde(default)]
    models: Vec<ModelFile>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelFile {
    id: String,
    name: Option<String>,
    api: Option<String>,
    reasoning: Option<bool>,
    default: Option<ReasoningEffort>,
    context_window: Option<u64>,
    max_input_tokens: Option<u64>,
    #[serde(default)]
    compat: CompatFile,
    #[serde(default)]
    thinking_level_map: BTreeMap<String, Option<String>>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
struct CompatFile {
    #[serde(rename = "supportsReasoningEffort")]
    reasoning_effort: Option<bool>,
    #[serde(rename = "supportsUsageInStreaming")]
    usage_in_streaming: Option<bool>,
    #[serde(rename = "supportsStrictTools")]
    strict_tools: Option<bool>,
}

impl CompatFile {
    fn merge(self, model: Self) -> Self {
        Self {
            reasoning_effort: model.reasoning_effort.or(self.reasoning_effort),
            usage_in_streaming: model.usage_in_streaming.or(self.usage_in_streaming),
            strict_tools: model.strict_tools.or(self.strict_tools),
        }
    }
}

impl AppConfig {
    pub fn load(overrides: &ConfigOverrides) -> Result<Self> {
        let cwd = overrides
            .cwd
            .clone()
            .unwrap_or(env::current_dir().context("failed to determine current directory")?);
        let cwd = cwd
            .canonicalize()
            .with_context(|| format!("working directory does not exist: {}", cwd.display()))?;

        let home = mcode_home_dir();
        let mut settings = SettingsFile::default();
        if let Some(home) = &home {
            let global_path = home.join("agent/settings.json");
            if global_path.is_file() {
                settings.overlay(read_json(&global_path)?);
            }
        }
        let project_path = cwd.join(".mcode/settings.json");
        if project_path.is_file() {
            settings.overlay(read_json(&project_path)?);
        }

        let base_url_override = overrides
            .base_url
            .clone()
            .or_else(|| env_non_empty("OPENAI_BASE_URL"));
        let forced_api_key = overrides.api_key_env.as_deref().and_then(env_non_empty);
        let fallback_api_key = env_non_empty("OPENAI_API_KEY");
        let environment_context = env_u64("OPENAI_CONTEXT_WINDOW").transpose()?;
        let context_window_override = overrides.context_window.or(environment_context);
        let environment_max_input = env_u64("OPENAI_MAX_INPUT_TOKENS").transpose()?;
        let max_input_tokens_override = overrides.max_input_tokens.or(environment_max_input);
        let mut model_profiles = if let Some(home) = &home {
            let models_path = home.join("agent/models.json");
            if models_path.is_file() {
                build_model_profiles(
                    read_json::<ModelsFile>(&models_path)?,
                    base_url_override.as_deref(),
                    overrides.api_key_env.as_ref().map(|_| &forced_api_key),
                    fallback_api_key.as_ref(),
                    context_window_override,
                    max_input_tokens_override,
                )?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        model_profiles
            .sort_by(|left, right| (&left.provider, &left.id).cmp(&(&right.provider, &right.id)));

        let preferred_provider = settings.provider.as_deref();
        let configured_model = overrides
            .model
            .clone()
            .or_else(|| env_non_empty("OPENAI_MODEL"))
            .or_else(|| settings.model.clone());
        let model = configured_model.unwrap_or_else(|| {
            if let [only_profile] = model_profiles.as_slice() {
                only_profile.qualified_id()
            } else {
                DEFAULT_MODEL.to_string()
            }
        });
        let selected_profile = find_model_profile(&model_profiles, preferred_provider, &model)?;
        let selected_model =
            selected_profile.map_or_else(|| model.clone(), |profile| profile.id.clone());
        let provider = selected_profile
            .map(|profile| profile.provider.clone())
            .or_else(|| settings.provider.clone());
        let configured_web_search = overrides.web_search.or(settings.web_search);
        let api = selected_profile.map_or(ApiProtocol::ChatCompletions, |profile| profile.api);

        let environment_reasoning = env_reasoning("OPENAI_REASONING_EFFORT").transpose()?;
        let requested_reasoning_effort = overrides.reasoning_effort.or(environment_reasoning);
        let reasoning_effort =
            resolve_initial_reasoning_effort(selected_profile, requested_reasoning_effort);
        let reasoning_value = selected_profile.map_or_else(
            || {
                Ok((reasoning_effort != ReasoningEffort::Off)
                    .then(|| reasoning_effort.as_str().to_string()))
            },
            |profile| profile.reasoning_value(reasoning_effort),
        )?;

        let base_url = base_url_override
            .or_else(|| selected_profile.map(|profile| profile.base_url.clone()))
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let api_key = if overrides.api_key_env.is_some() {
            forced_api_key
        } else {
            selected_profile.map_or(fallback_api_key, |profile| profile.api_key.clone())
        };
        let context_window = context_window_override
            .or_else(|| selected_profile.map(|profile| profile.context_window))
            .unwrap_or(DEFAULT_CONTEXT_WINDOW);
        let max_input_tokens = max_input_tokens_override
            .or_else(|| selected_profile.map(|profile| profile.max_input_tokens))
            .unwrap_or(context_window);
        let supports_reasoning_effort =
            selected_profile.is_none_or(|profile| profile.compat.reasoning_effort);
        let supports_usage_in_streaming =
            selected_profile.is_none_or(|profile| profile.compat.usage_in_streaming);
        let supports_strict_tools = selected_profile.map_or_else(
            || api == ApiProtocol::Responses && is_official_openai_url(&base_url),
            |profile| profile.compat.strict_tools,
        );
        let max_tool_turns = overrides.max_tool_turns.unwrap_or(32);
        let request_timeout_secs = overrides.request_timeout_secs.unwrap_or(300);
        let defaults = CompactionSettings::default();
        let compaction = CompactionSettings {
            enabled: settings.compaction.enabled.unwrap_or(defaults.enabled),
            reserve_tokens: settings
                .compaction
                .reserve_tokens
                .unwrap_or(defaults.reserve_tokens),
            keep_recent_tokens: settings
                .compaction
                .keep_recent_tokens
                .unwrap_or(defaults.keep_recent_tokens),
        };
        let web_search = WebSearchSettings {
            mode: configured_web_search.unwrap_or_default(),
            provider: settings.web_search_config.provider.unwrap_or_default(),
            allowed_domains: normalize_allowed_web_search_domains(
                settings
                    .web_search_config
                    .allowed_domains
                    .unwrap_or_default(),
            )?,
            exa_api_key: resolve_web_search_secret(
                settings.web_search_config.exa_api_key.as_deref(),
                "EXA_API_KEY",
            ),
            brave_api_key: resolve_web_search_secret(
                settings.web_search_config.brave_api_key.as_deref(),
                "BRAVE_API_KEY",
            ),
            searxng_base_url: settings
                .web_search_config
                .searxng_base_url
                .or_else(|| env_non_empty("SEARXNG_BASE_URL")),
            trust_env_proxy: settings.web_search_config.trust_env_proxy.unwrap_or(false),
        };
        let mcp_servers = build_mcp_servers(&settings.mcp_servers)?;

        if selected_model.trim().is_empty() {
            bail!("model cannot be empty");
        }
        if context_window == 0 {
            bail!("context window must be at least 1");
        }
        if max_input_tokens == 0 || max_input_tokens > context_window {
            bail!(
                "max input tokens must be between 1 and the {context_window}-token context window"
            );
        }
        if max_tool_turns == 0 {
            bail!("max tool turns must be at least 1");
        }
        if request_timeout_secs == 0 {
            bail!("request timeout must be at least 1 second");
        }
        if compaction.reserve_tokens == 0 {
            bail!("compaction.reserveTokens must be at least 1");
        }
        if compaction.keep_recent_tokens == 0 {
            bail!("compaction.keepRecentTokens must be at least 1");
        }
        validate_web_search(&web_search)?;

        Ok(Self {
            model: selected_model,
            provider,
            api,
            reasoning_effort,
            reasoning_value,
            base_url,
            api_key,
            context_window,
            max_input_tokens,
            supports_reasoning_effort,
            supports_usage_in_streaming,
            supports_strict_tools,
            cwd,
            max_tool_turns,
            request_timeout_secs,
            compaction,
            web_search,
            model_profiles,
            mcp_servers,
        })
    }

    pub fn select_model(&mut self, query: &str) -> Result<()> {
        let profile =
            find_model_profile(&self.model_profiles, self.provider.as_deref(), query)?.cloned();
        if let Some(profile) = profile {
            self.reasoning_effort = profile.clamp_reasoning_effort(self.reasoning_effort);
            self.reasoning_value = profile.reasoning_value(self.reasoning_effort)?;
            self.model = profile.id;
            self.provider = Some(profile.provider);
            self.api = profile.api;
            self.base_url = profile.base_url;
            self.api_key = profile.api_key;
            self.context_window = profile.context_window;
            self.max_input_tokens = profile.max_input_tokens;
            self.supports_reasoning_effort = profile.compat.reasoning_effort;
            self.supports_usage_in_streaming = profile.compat.usage_in_streaming;
            self.supports_strict_tools = profile.compat.strict_tools;
        } else {
            let query = query.trim();
            if query.is_empty() {
                bail!("model cannot be empty");
            }
            self.model = query.to_string();
            self.reasoning_value = (self.reasoning_effort != ReasoningEffort::Off)
                .then(|| self.reasoning_effort.as_str().to_string());
        }
        Ok(())
    }

    pub fn select_reasoning_effort(&mut self, effort: ReasoningEffort) -> Result<()> {
        let profile =
            find_model_profile(&self.model_profiles, self.provider.as_deref(), &self.model)?;
        let effective_effort =
            profile.map_or(effort, |profile| profile.clamp_reasoning_effort(effort));
        self.reasoning_value = profile.map_or_else(
            || {
                Ok((effective_effort != ReasoningEffort::Off)
                    .then(|| effective_effort.as_str().to_string()))
            },
            |profile| profile.reasoning_value(effective_effort),
        )?;
        self.reasoning_effort = effective_effort;
        Ok(())
    }

    pub fn select_model_and_reasoning(
        &mut self,
        model: &str,
        reasoning_effort: ReasoningEffort,
    ) -> Result<()> {
        let previous_effort = self.reasoning_effort;
        let previous_value = self.reasoning_value.clone();
        self.reasoning_effort = reasoning_effort;
        if let Err(error) = self.select_model(model) {
            self.reasoning_effort = previous_effort;
            self.reasoning_value = previous_value;
            return Err(error);
        }
        Ok(())
    }
}

fn resolve_initial_reasoning_effort(
    profile: Option<&ModelProfile>,
    requested: Option<ReasoningEffort>,
) -> ReasoningEffort {
    match (profile, requested) {
        (Some(profile), Some(requested)) => profile.clamp_reasoning_effort(requested),
        (Some(profile), None) => profile.default_reasoning_effort(),
        (None, requested) => requested.unwrap_or_default(),
    }
}

#[must_use]
pub fn mcode_home_dir() -> Option<PathBuf> {
    env::var_os("MCODE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".mcode")))
}

fn build_mcp_servers(servers: &BTreeMap<String, McpServerFile>) -> Result<Vec<McpServerConfig>> {
    let mut configured = Vec::new();
    for (name, server) in servers {
        if server.enabled == Some(false) {
            continue;
        }
        if name.trim().is_empty() {
            bail!("MCP server name cannot be empty");
        }
        let command = server
            .command
            .as_deref()
            .filter(|command| !command.trim().is_empty())
            .with_context(|| format!("MCP server {name:?} is missing command"))?;
        let args = server.args.clone();
        let env = server
            .env
            .iter()
            .map(|(key, value)| {
                if key.trim().is_empty() {
                    bail!("MCP server {name:?} contains an empty environment variable name");
                }
                let value = interpolate_environment(value, |variable| env::var(variable).ok())
                    .with_context(|| {
                        format!(
                            "MCP server {name:?} environment value {key:?} references a missing variable"
                        )
                    })?;
                Ok((key.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        configured.push(McpServerConfig {
            name: name.clone(),
            command: command.to_string(),
            args,
            env,
        });
    }
    Ok(configured)
}

fn validate_web_search(settings: &WebSearchSettings) -> Result<()> {
    normalize_allowed_web_search_domains(settings.allowed_domains.clone())?;
    if settings.provider == WebSearchProvider::Brave && settings.brave_api_key.is_none() {
        bail!("webSearchConfig.provider is brave but BRAVE_API_KEY is not configured");
    }
    if settings.provider == WebSearchProvider::Searxng {
        let base_url = settings.searxng_base_url.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "webSearchConfig.provider is searxng but searxngBaseUrl is not configured"
            )
        })?;
        validate_http_base_url(base_url, "webSearchConfig.searxngBaseUrl")?;
    } else if let Some(base_url) = settings.searxng_base_url.as_deref() {
        validate_http_base_url(base_url, "webSearchConfig.searxngBaseUrl")?;
    }
    Ok(())
}

fn normalize_allowed_web_search_domains(domains: Vec<String>) -> Result<Vec<String>> {
    normalize_web_search_domains(domains, false)
}

pub(crate) fn normalize_web_search_domain_filters(domains: Vec<String>) -> Result<Vec<String>> {
    normalize_web_search_domains(domains, true)
}

fn normalize_web_search_domains(
    domains: Vec<String>,
    allow_exclusions: bool,
) -> Result<Vec<String>> {
    if domains.len() > 20 {
        bail!("web search domain filters cannot contain more than 20 domains");
    }
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(domains.len());
    for domain in domains {
        let domain = domain.trim();
        let (excluded, hostname) = domain
            .strip_prefix('-')
            .map_or((false, domain), |hostname| (true, hostname));
        if excluded && !allow_exclusions {
            bail!("invalid allowed domain {domain:?}; exclusions are only valid in tool filters");
        }
        let url::Host::Domain(hostname) = url::Host::parse(hostname).map_err(|_| {
            anyhow::anyhow!(
                "invalid web search domain {domain:?}; use a hostname without a URL scheme"
            )
        })?
        else {
            bail!("invalid web search domain {domain:?}; use a DNS hostname");
        };
        let normalized_domain = if excluded {
            format!("-{hostname}")
        } else {
            hostname
        };
        if seen.insert(normalized_domain.clone()) {
            normalized.push(normalized_domain);
        }
    }
    Ok(normalized)
}

fn resolve_web_search_secret(configured: Option<&str>, environment: &str) -> Option<String> {
    configured
        .map_or_else(|| env_non_empty(environment), resolve_static_config_value)
        .filter(|value| !value.trim().is_empty())
}

fn validate_http_base_url(value: &str, field: &str) -> Result<()> {
    let url = url::Url::parse(value).with_context(|| format!("invalid {field}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        bail!("{field} must be an HTTP or HTTPS URL with a hostname");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("{field} must not contain credentials");
    }
    Ok(())
}

pub fn find_model_profile<'a>(
    profiles: &'a [ModelProfile],
    preferred_provider: Option<&str>,
    query: &str,
) -> Result<Option<&'a ModelProfile>> {
    let query = query.trim();
    if query.is_empty() {
        bail!("model cannot be empty");
    }

    if let Some(profile) = profiles
        .iter()
        .find(|profile| profile.qualified_id() == query)
    {
        return Ok(Some(profile));
    }
    if let Some(provider) = preferred_provider
        && let Some(profile) = profiles.iter().find(|profile| {
            profile.provider == provider
                && (profile.id == query
                    || profile
                        .name
                        .as_deref()
                        .is_some_and(|name| name.eq_ignore_ascii_case(query)))
        })
    {
        return Ok(Some(profile));
    }

    let matches = profiles
        .iter()
        .filter(|profile| {
            profile.id == query
                || profile
                    .name
                    .as_deref()
                    .is_some_and(|name| name.eq_ignore_ascii_case(query))
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [profile] => Ok(Some(*profile)),
        _ => bail!(
            "model {query:?} is ambiguous; use provider/model (matches: {})",
            matches
                .iter()
                .map(|profile| profile.qualified_id())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn build_model_profiles(
    file: ModelsFile,
    base_url_override: Option<&str>,
    forced_api_key: Option<&Option<String>>,
    fallback_api_key: Option<&String>,
    context_window_override: Option<u64>,
    max_input_tokens_override: Option<u64>,
) -> Result<Vec<ModelProfile>> {
    let mut profiles = Vec::new();
    for (provider_name, provider) in file.providers {
        for model in provider.models {
            let Some(api) = model
                .api
                .as_deref()
                .or(provider.api.as_deref())
                .and_then(parse_api_protocol)
            else {
                continue;
            };
            if model.id.trim().is_empty() {
                bail!("provider {provider_name} contains a model with an empty id");
            }
            let base_url = base_url_override
                .map(ToString::to_string)
                .or_else(|| provider.base_url.clone())
                .or_else(|| (provider_name == "openai").then(|| DEFAULT_BASE_URL.to_string()))
                .with_context(|| {
                    format!(
                        "OpenAI-compatible provider {provider_name:?} is missing baseUrl in models.json"
                    )
                })?;
            let api_key = forced_api_key.map_or_else(
                || match provider.api_key.as_deref() {
                    Some(value) => resolve_static_config_value(value),
                    None => fallback_api_key.cloned(),
                },
                |value| (*value).clone(),
            );
            let mut thinking_level_map = BTreeMap::new();
            for (level, value) in model.thinking_level_map {
                let level = parse_reasoning_effort(&level).with_context(|| {
                    format!(
                        "invalid thinkingLevelMap key for {}/{}",
                        provider_name, model.id
                    )
                })?;
                thinking_level_map.insert(level, value);
            }
            let context_window = context_window_override
                .or(model.context_window)
                .unwrap_or(DEFAULT_CONTEXT_WINDOW);
            let max_input_tokens = max_input_tokens_override
                .or(context_window_override)
                .or(model.max_input_tokens)
                .unwrap_or(context_window);
            let compat = provider.compat.merge(model.compat);
            let reasoning = model.reasoning.unwrap_or(false);
            let default_reasoning_effort = model.default.with_context(|| {
                format!(
                    "模型 {}/{} 缺少 default；请指定默认 effort",
                    provider_name, model.id
                )
            })?;
            if context_window == 0 {
                bail!(
                    "model {}/{} has a zero contextWindow",
                    provider_name,
                    model.id
                );
            }
            if max_input_tokens == 0 || max_input_tokens > context_window {
                bail!(
                    "model {}/{} has maxInputTokens outside its 1..={context_window} context window",
                    provider_name,
                    model.id
                );
            }
            let supports_strict_tools = compat.strict_tools.unwrap_or_else(|| {
                api == ApiProtocol::Responses && is_official_openai_url(&base_url)
            });
            let profile = ModelProfile {
                provider: provider_name.clone(),
                id: model.id,
                name: model.name,
                api,
                base_url,
                api_key,
                context_window,
                max_input_tokens,
                reasoning,
                compat: ModelCompat {
                    reasoning_effort: compat.reasoning_effort.unwrap_or(true),
                    usage_in_streaming: compat.usage_in_streaming.unwrap_or(true),
                    strict_tools: supports_strict_tools,
                },
                default_reasoning_effort,
                thinking_level_map,
            };
            let configured_efforts = profile.supported_reasoning_efforts();
            if profile.reasoning && configured_efforts.is_empty() {
                bail!(
                    "模型 {} 必须在 thinkingLevelMap 中配置至少一个非 null 等级",
                    profile.qualified_id()
                );
            }
            if !configured_efforts.contains(&profile.default_reasoning_effort) {
                let configured = configured_efforts
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                bail!(
                    "模型 {} 的默认 effort {} 不可用；可选值：{configured}",
                    profile.qualified_id(),
                    profile.default_reasoning_effort
                );
            }
            profiles.push(profile);
        }
    }
    Ok(profiles)
}

fn is_official_openai_url(value: &str) -> bool {
    url::Url::parse(value)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| host.eq_ignore_ascii_case("api.openai.com"))
}

fn parse_api_protocol(value: &str) -> Option<ApiProtocol> {
    match value {
        "openai-completions" => Some(ApiProtocol::ChatCompletions),
        "openai-responses" => Some(ApiProtocol::Responses),
        _ => None,
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read config: {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("invalid JSON config: {}", path.display()))
}

fn parse_reasoning_effort(value: &str) -> Result<ReasoningEffort> {
    ReasoningEffort::from_str(value, true)
        .map_err(|_| anyhow::anyhow!("unknown reasoning level {value:?}"))
}

fn env_reasoning(name: &str) -> Option<Result<ReasoningEffort>> {
    env_non_empty(name).map(|value| parse_reasoning_effort(&value))
}

fn env_u64(name: &str) -> Option<Result<u64>> {
    env_non_empty(name).map(|value| {
        value
            .parse::<u64>()
            .with_context(|| format!("{name} must be a positive integer"))
    })
}

fn env_non_empty(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn resolve_static_config_value(value: &str) -> Option<String> {
    if value.starts_with('!') {
        return None;
    }
    interpolate_environment(value, |name| env::var(name).ok())
}

fn interpolate_environment(
    value: &str,
    mut lookup: impl FnMut(&str) -> Option<String>,
) -> Option<String> {
    let chars = value.chars().collect::<Vec<_>>();
    let mut output = String::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '$' {
            output.push(chars[index]);
            index += 1;
            continue;
        }
        if chars.get(index + 1) == Some(&'$') {
            output.push('$');
            index += 2;
            continue;
        }
        if chars.get(index + 1) == Some(&'!') {
            output.push('!');
            index += 2;
            continue;
        }
        if chars.get(index + 1) == Some(&'{') {
            let end = chars[index + 2..]
                .iter()
                .position(|character| character == &'}')?
                + index
                + 2;
            let name = chars[index + 2..end].iter().collect::<String>();
            output.push_str(&lookup(&name)?);
            index = end + 1;
            continue;
        }
        let end = chars[index + 1..]
            .iter()
            .position(|character| !character.is_ascii_alphanumeric() && *character != '_')
            .map_or(chars.len(), |offset| index + 1 + offset);
        if end == index + 1 {
            output.push('$');
            index += 1;
            continue;
        }
        let name = chars[index + 1..end].iter().collect::<String>();
        output.push_str(&lookup(&name)?);
        index = end;
    }
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_settings_overlay_only_present_values() {
        let mut base = SettingsFile {
            provider: Some("one".into()),
            model: Some("model-one".into()),
            mcp_servers: BTreeMap::new(),
            compaction: CompactionSettingsFile::default(),
            web_search: None,
            web_search_config: WebSearchSettingsFile::default(),
        };
        base.overlay(SettingsFile {
            model: Some("model-two".into()),
            ..SettingsFile::default()
        });
        assert_eq!(base.provider.as_deref(), Some("one"));
        assert_eq!(base.model.as_deref(), Some("model-two"));
    }

    #[test]
    fn compaction_settings_use_pi_fields_and_overlay_individually() {
        let mut global: SettingsFile = serde_json::from_str(
            r#"{
                "compaction": {
                    "enabled": false,
                    "reserveTokens": 8192,
                    "keepRecentTokens": 12000
                }
            }"#,
        )
        .unwrap();
        let project: SettingsFile =
            serde_json::from_str(r#"{"compaction":{"enabled":true,"keepRecentTokens":24000}}"#)
                .unwrap();
        global.overlay(project);

        assert_eq!(global.compaction.enabled, Some(true));
        assert_eq!(global.compaction.reserve_tokens, Some(8_192));
        assert_eq!(global.compaction.keep_recent_tokens, Some(24_000));
        assert_eq!(CompactionSettings::default().reserve_tokens, 16_384);
        assert_eq!(CompactionSettings::default().keep_recent_tokens, 20_000);
    }

    #[test]
    fn parses_pi_models_and_reasoning_map() {
        let file: ModelsFile = serde_json::from_str(
            r#"{
                "providers": {
                    "proxy": {
                        "baseUrl": "https://proxy.test/v1",
                        "api": "openai-completions",
                        "apiKey": "$TEST_KEY",
                        "compat": {
                            "supportsReasoningEffort": false,
                            "supportsUsageInStreaming": false
                        },
                        "models": [{
                            "id": "coder",
                            "name": "Coder",
                            "reasoning": true,
                            "default": "high",
                            "contextWindow": 200000,
                            "compat": {"supportsReasoningEffort": true},
                            "thinkingLevelMap": {"high": "high", "max": null}
                        }]
                    }
                }
            }"#,
        )
        .unwrap();
        let profiles = build_model_profiles(file, None, None, None, None, None).unwrap();
        let profile = &profiles[0];
        assert_eq!(profile.qualified_id(), "proxy/coder");
        assert_eq!(profile.api, ApiProtocol::ChatCompletions);
        assert_eq!(profile.context_window, 200_000);
        assert!(profile.reasoning);
        assert!(profile.compat.reasoning_effort);
        assert_eq!(profile.default_reasoning_effort(), ReasoningEffort::High);
        assert_eq!(
            resolve_initial_reasoning_effort(Some(profile), None),
            ReasoningEffort::High
        );
        assert_eq!(
            profile.supported_reasoning_efforts(),
            vec![ReasoningEffort::High]
        );
        assert!(!profile.compat.usage_in_streaming);
        assert_eq!(
            profile.reasoning_value(ReasoningEffort::High).unwrap(),
            Some("high".to_string())
        );
        assert_eq!(
            profile.clamp_reasoning_effort(ReasoningEffort::Max),
            ReasoningEffort::High
        );
        assert!(profile.reasoning_value(ReasoningEffort::Max).is_err());
    }

    #[test]
    fn rejects_a_model_default_outside_its_configured_efforts() {
        let file: ModelsFile = serde_json::from_str(
            r#"{
                "providers": {
                    "proxy": {
                        "baseUrl": "https://proxy.test/v1",
                        "api": "openai-completions",
                        "models": [{
                            "id": "coder",
                            "reasoning": true,
                            "default": "medium",
                            "thinkingLevelMap": {"high": "high"}
                        }]
                    }
                }
            }"#,
        )
        .unwrap();
        let error = build_model_profiles(file, None, None, None, None, None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("可选值：high"));
    }

    #[test]
    fn rejects_a_model_without_an_explicit_default_effort() {
        let file: ModelsFile = serde_json::from_str(
            r#"{
                "providers": {
                    "proxy": {
                        "baseUrl": "https://proxy.test/v1",
                        "api": "openai-completions",
                        "models": [{"id": "coder"}]
                    }
                }
            }"#,
        )
        .unwrap();
        let error = build_model_profiles(file, None, None, None, None, None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("缺少 default"));
    }

    #[test]
    fn example_configs_cover_supported_compatible_providers() {
        let models: ModelsFile = serde_json::from_str(include_str!("../models.example.json"))
            .expect("models.example.json should be valid");
        let profiles = build_model_profiles(models, None, None, None, None, None).unwrap();
        let qualified = profiles
            .iter()
            .map(ModelProfile::qualified_id)
            .collect::<BTreeSet<_>>();
        for expected in [
            "xai/grok-4.3",
            "deepseek/deepseek-v4-pro",
            "kimi/kimi-k3",
            "glm/glm-5.2",
        ] {
            assert!(qualified.contains(expected), "missing {expected}");
        }

        let settings: SettingsFile = serde_json::from_str(include_str!("../settings.example.json"))
            .expect("settings.example.json should be valid");
        assert_eq!(settings.provider.as_deref(), Some("openai-compatible"));
        assert_eq!(
            settings.web_search_config.provider,
            Some(WebSearchProvider::Auto)
        );
    }

    #[test]
    fn provider_keys_do_not_fall_back_when_an_explicit_secret_is_missing() {
        let models: ModelsFile = serde_json::from_str(
            r#"{
                "providers": {
                    "explicit": {
                        "baseUrl": "https://explicit.test/v1",
                        "api": "openai-completions",
                        "apiKey": "$MCODE_TEST_EXPLICIT_KEY_THAT_DOES_NOT_EXIST",
                        "models": [{"id": "one", "default": "off"}]
                    },
                    "implicit": {
                        "baseUrl": "https://implicit.test/v1",
                        "api": "openai-completions",
                        "models": [{"id": "two", "default": "off"}]
                    }
                }
            }"#,
        )
        .unwrap();
        let fallback = "openai-fallback".to_string();
        let profiles =
            build_model_profiles(models, None, None, Some(&fallback), None, None).unwrap();
        let explicit = profiles
            .iter()
            .find(|profile| profile.provider == "explicit")
            .unwrap();
        let implicit = profiles
            .iter()
            .find(|profile| profile.provider == "implicit")
            .unwrap();

        assert!(explicit.api_key.is_none());
        assert_eq!(implicit.api_key.as_deref(), Some("openai-fallback"));
    }

    #[test]
    fn parses_responses_model_and_hybrid_web_search_settings() {
        let settings: SettingsFile = serde_json::from_str(
            r#"{
                "webSearch": "live",
                "webSearchConfig": {
                    "provider": "searxng",
                    "allowedDomains": ["openai.com", "rust-lang.org"],
                    "searxngBaseUrl": "https://search.example.com/",
                    "trustEnvProxy": true
                }
            }"#,
        )
        .unwrap();
        assert_eq!(settings.web_search, Some(WebSearchMode::Live));
        assert_eq!(
            settings.web_search_config.provider,
            Some(WebSearchProvider::Searxng)
        );
        assert_eq!(
            settings.web_search_config.allowed_domains.as_deref(),
            Some(["openai.com".to_string(), "rust-lang.org".to_string()].as_slice())
        );
        assert_eq!(
            settings.web_search_config.searxng_base_url.as_deref(),
            Some("https://search.example.com/")
        );
        assert_eq!(settings.web_search_config.trust_env_proxy, Some(true));

        let models: ModelsFile = serde_json::from_str(
            r#"{
                "providers": {
                    "openai": {
                        "api": "openai-responses",
                        "models": [{
                            "id": "gpt-test",
                            "default": "off",
                            "contextWindow": 300000,
                            "maxInputTokens": 272000
                        }]
                    }
                }
            }"#,
        )
        .unwrap();
        let profiles = build_model_profiles(models, None, None, None, None, None).unwrap();
        assert_eq!(profiles[0].api, ApiProtocol::Responses);
        assert_eq!(profiles[0].context_window, 300_000);
        assert_eq!(profiles[0].max_input_tokens, 272_000);

        let search = WebSearchSettings {
            mode: WebSearchMode::Live,
            provider: settings.web_search_config.provider.unwrap(),
            allowed_domains: settings
                .web_search_config
                .allowed_domains
                .unwrap_or_default(),
            searxng_base_url: settings.web_search_config.searxng_base_url,
            trust_env_proxy: settings
                .web_search_config
                .trust_env_proxy
                .unwrap_or_default(),
            ..WebSearchSettings::default()
        };
        validate_web_search(&search).unwrap();
    }

    #[test]
    fn normalizes_and_validates_web_search_domains() {
        assert_eq!(
            normalize_allowed_web_search_domains(vec![
                " Example.COM ".to_string(),
                "example.com".to_string(),
            ])
            .unwrap(),
            vec!["example.com".to_string()]
        );
        assert_eq!(
            normalize_web_search_domain_filters(vec!["-Docs.Example.com".to_string()]).unwrap(),
            vec!["-docs.example.com".to_string()]
        );
        assert!(normalize_allowed_web_search_domains(vec!["-example.com".to_string()]).is_err());
        assert!(
            normalize_allowed_web_search_domains(vec!["https://example.com".to_string()]).is_err()
        );
        assert!(normalize_allowed_web_search_domains(vec!["127.0.0.1".to_string()]).is_err());
        assert!(normalize_allowed_web_search_domains(vec!["example.com".to_string(); 21]).is_err());
    }

    #[test]
    fn environment_interpolation_matches_pi_static_values() {
        let value = interpolate_environment("Bearer ${TOKEN}_$SUFFIX $$ $!", |name| match name {
            "TOKEN" => Some("abc".to_string()),
            "SUFFIX" => Some("prod".to_string()),
            _ => None,
        });
        assert_eq!(value.as_deref(), Some("Bearer abc_prod $ !"));
    }

    #[test]
    fn strict_tools_default_only_for_official_responses_profiles() {
        let models: ModelsFile = serde_json::from_str(
            r#"{
                "providers": {
                    "openai": {
                        "api": "openai-responses",
                        "models": [
                            {"id": "official-default", "default": "off"},
                            {
                                "id": "official-disabled",
                                "default": "off",
                                "compat": {"supportsStrictTools": false}
                            }
                        ]
                    },
                    "proxy": {
                        "baseUrl": "https://proxy.test/v1",
                        "api": "openai-responses",
                        "models": [{"id": "proxy-default", "default": "off"}]
                    }
                }
            }"#,
        )
        .unwrap();
        let profiles = build_model_profiles(models, None, None, None, None, None).unwrap();
        let strict = |provider: &str, model: &str| {
            profiles
                .iter()
                .find(|profile| profile.provider == provider && profile.id == model)
                .unwrap()
                .compat
                .strict_tools
        };

        assert!(strict("openai", "official-default"));
        assert!(!strict("openai", "official-disabled"));
        assert!(!strict("proxy", "proxy-default"));
    }

    #[test]
    fn model_resolution_requires_provider_for_ambiguous_ids() {
        let profile = |provider: &str| ModelProfile {
            provider: provider.to_string(),
            id: "same".to_string(),
            name: None,
            api: ApiProtocol::ChatCompletions,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            context_window: DEFAULT_CONTEXT_WINDOW,
            max_input_tokens: DEFAULT_CONTEXT_WINDOW,
            reasoning: false,
            compat: ModelCompat {
                reasoning_effort: true,
                usage_in_streaming: true,
                strict_tools: false,
            },
            default_reasoning_effort: ReasoningEffort::Off,
            thinking_level_map: BTreeMap::new(),
        };
        let profiles = vec![profile("one"), profile("two")];
        assert!(find_model_profile(&profiles, None, "same").is_err());
        assert_eq!(
            find_model_profile(&profiles, Some("two"), "same")
                .unwrap()
                .unwrap()
                .provider,
            "two"
        );
    }

    #[test]
    fn selecting_profile_updates_endpoint_context_and_reasoning_mapping() {
        let mut map = BTreeMap::new();
        map.insert(ReasoningEffort::High, Some("deep".to_string()));
        let profile = ModelProfile {
            provider: "proxy".to_string(),
            id: "coder".to_string(),
            name: None,
            api: ApiProtocol::ChatCompletions,
            base_url: "https://proxy.test/v1".to_string(),
            api_key: Some("secret".to_string()),
            context_window: 200_000,
            max_input_tokens: 180_000,
            reasoning: true,
            compat: ModelCompat {
                reasoning_effort: false,
                usage_in_streaming: false,
                strict_tools: false,
            },
            default_reasoning_effort: ReasoningEffort::High,
            thinking_level_map: map,
        };
        let mut config = AppConfig {
            model: "old-model".to_string(),
            provider: None,
            api: ApiProtocol::ChatCompletions,
            reasoning_effort: ReasoningEffort::High,
            reasoning_value: Some("high".to_string()),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            context_window: DEFAULT_CONTEXT_WINDOW,
            max_input_tokens: DEFAULT_CONTEXT_WINDOW,
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
            supports_strict_tools: true,
            cwd: PathBuf::from("."),
            max_tool_turns: 32,
            request_timeout_secs: 300,
            compaction: CompactionSettings::default(),
            web_search: WebSearchSettings::default(),
            model_profiles: vec![profile],
            mcp_servers: Vec::new(),
        };

        config.select_model("proxy/coder").unwrap();
        assert_eq!(config.model, "coder");
        assert_eq!(config.provider.as_deref(), Some("proxy"));
        assert_eq!(config.base_url, "https://proxy.test/v1");
        assert_eq!(config.context_window, 200_000);
        assert_eq!(config.max_input_tokens, 180_000);
        assert_eq!(config.reasoning_value.as_deref(), Some("deep"));
        assert!(!config.supports_reasoning_effort);
        assert!(!config.supports_usage_in_streaming);
        assert!(!config.supports_strict_tools);
    }

    #[test]
    fn non_reasoning_profile_rejects_reasoning_effort() {
        let profile = ModelProfile {
            provider: "proxy".to_string(),
            id: "plain".to_string(),
            name: None,
            api: ApiProtocol::ChatCompletions,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            context_window: DEFAULT_CONTEXT_WINDOW,
            max_input_tokens: DEFAULT_CONTEXT_WINDOW,
            reasoning: false,
            compat: ModelCompat {
                reasoning_effort: true,
                usage_in_streaming: true,
                strict_tools: false,
            },
            default_reasoning_effort: ReasoningEffort::Off,
            thinking_level_map: BTreeMap::new(),
        };
        assert!(profile.reasoning_value(ReasoningEffort::Medium).is_err());
        assert_eq!(profile.reasoning_value(ReasoningEffort::Off).unwrap(), None);
        assert_eq!(
            profile.clamp_reasoning_effort(ReasoningEffort::Medium),
            ReasoningEffort::Off
        );
    }

    #[test]
    fn a_single_profile_can_be_used_as_the_unambiguous_default() {
        let profile = ModelProfile {
            provider: "proxy".to_string(),
            id: "only-model".to_string(),
            name: None,
            api: ApiProtocol::ChatCompletions,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            context_window: DEFAULT_CONTEXT_WINDOW,
            max_input_tokens: DEFAULT_CONTEXT_WINDOW,
            reasoning: false,
            compat: ModelCompat {
                reasoning_effort: true,
                usage_in_streaming: true,
                strict_tools: false,
            },
            default_reasoning_effort: ReasoningEffort::Off,
            thinking_level_map: BTreeMap::new(),
        };
        let profiles = [profile];
        let selected = find_model_profile(&profiles, None, &profiles[0].qualified_id())
            .unwrap()
            .unwrap();
        assert_eq!(selected.id, "only-model");
    }

    #[test]
    fn project_mcp_servers_override_global_servers_by_name() {
        let mut settings: SettingsFile = serde_json::from_str(
            r#"{
                "mcpServers": {
                    "files": {"command": "global-server", "args": ["global"]},
                    "disabled": {"enabled": false}
                }
            }"#,
        )
        .unwrap();
        let project: SettingsFile = serde_json::from_str(
            r#"{
                "mcpServers": {
                    "files": {"command": "project-server", "args": ["project"]}
                }
            }"#,
        )
        .unwrap();
        settings.overlay(project);

        let servers = build_mcp_servers(&settings.mcp_servers).unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "files");
        assert_eq!(servers[0].command, "project-server");
        assert_eq!(servers[0].args, ["project"]);
    }
}
