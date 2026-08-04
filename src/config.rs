use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
const SUPPORTED_PROVIDERS: [&str; 4] = ["xai", "deepseek", "glm", "kimi"];

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
    pub max_output_tokens: Option<u64>,
    pub reasoning: bool,
    pub supports_images: bool,
    pub compat: ModelCompat,
    default_reasoning_effort: ReasoningEffort,
    thinking_level_map: BTreeMap<ReasoningEffort, Option<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct ModelCompat {
    pub reasoning_effort: bool,
    pub usage_in_streaming: bool,
    pub finish_reason: bool,
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
    pub provider: String,
    pub api: ApiProtocol,
    pub reasoning_effort: ReasoningEffort,
    pub reasoning_value: Option<String>,
    pub base_url: String,
    pub api_key: Option<String>,
    pub context_window: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: Option<u64>,
    pub compat: ModelCompat,
    pub cwd: PathBuf,
    pub request_timeout_secs: u64,
    pub compaction: CompactionSettings,
    pub web_search: WebSearchSettings,
    pub model_profiles: Vec<ModelProfile>,
    pub mcp_servers: Vec<McpServerConfig>,
    pub reload_overrides: ConfigOverrides,
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
    pub max_output_tokens: Option<u64>,
    pub cwd: Option<PathBuf>,
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
    args: Option<Vec<String>>,
    env: Option<BTreeMap<String, String>>,
    enabled: Option<bool>,
}

impl McpServerFile {
    fn overlay(&mut self, other: Self) {
        overlay_option(&mut self.command, other.command);
        overlay_option(&mut self.args, other.args);
        overlay_option(&mut self.enabled, other.enabled);
        if let Some(env) = other.env {
            self.env.get_or_insert_default().extend(env);
        }
    }
}

impl SettingsFile {
    fn overlay(&mut self, other: Self) {
        overlay_option(&mut self.provider, other.provider);
        overlay_option(&mut self.model, other.model);
        overlay_option(&mut self.compaction.enabled, other.compaction.enabled);
        overlay_option(
            &mut self.compaction.reserve_tokens,
            other.compaction.reserve_tokens,
        );
        overlay_option(
            &mut self.compaction.keep_recent_tokens,
            other.compaction.keep_recent_tokens,
        );
        overlay_option(&mut self.web_search, other.web_search);
        overlay_option(
            &mut self.web_search_config.provider,
            other.web_search_config.provider,
        );
        overlay_option(
            &mut self.web_search_config.allowed_domains,
            other.web_search_config.allowed_domains,
        );
        overlay_option(
            &mut self.web_search_config.exa_api_key,
            other.web_search_config.exa_api_key,
        );
        overlay_option(
            &mut self.web_search_config.brave_api_key,
            other.web_search_config.brave_api_key,
        );
        overlay_option(
            &mut self.web_search_config.searxng_base_url,
            other.web_search_config.searxng_base_url,
        );
        overlay_option(
            &mut self.web_search_config.trust_env_proxy,
            other.web_search_config.trust_env_proxy,
        );
        for (name, server) in other.mcp_servers {
            if let Some(current) = self.mcp_servers.get_mut(&name) {
                current.overlay(server);
            } else {
                self.mcp_servers.insert(name, server);
            }
        }
    }
}

fn overlay_option<T>(current: &mut Option<T>, overlay: Option<T>) {
    if let Some(value) = overlay {
        *current = Some(value);
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
    input: Option<Vec<InputModality>>,
    default: Option<ReasoningEffort>,
    context_window: Option<u64>,
    max_input_tokens: Option<u64>,
    max_output_tokens: Option<u64>,
    #[serde(default)]
    compat: CompatFile,
    #[serde(default)]
    thinking_level_map: BTreeMap<String, Option<String>>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum InputModality {
    Text,
    Image,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
struct CompatFile {
    #[serde(rename = "supportsReasoningEffort")]
    reasoning_effort: Option<bool>,
    #[serde(rename = "supportsUsageInStreaming")]
    usage_in_streaming: Option<bool>,
    #[serde(rename = "supportsFinishReason")]
    finish_reason: Option<bool>,
    #[serde(rename = "supportsStrictTools")]
    strict_tools: Option<bool>,
}

impl CompatFile {
    fn merge(self, model: Self) -> Self {
        Self {
            reasoning_effort: model.reasoning_effort.or(self.reasoning_effort),
            usage_in_streaming: model.usage_in_streaming.or(self.usage_in_streaming),
            finish_reason: model.finish_reason.or(self.finish_reason),
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
        let mut reload_overrides = overrides.clone();
        reload_overrides.cwd = Some(cwd.clone());

        let home = mcode_home_dir();
        let mut settings = SettingsFile::default();
        if let Some(home) = &home {
            let global_path = home.join("settings.json");
            if global_path.is_file() {
                settings.overlay(read_json(&global_path)?);
            }
        }
        let project_path = cwd.join(".mcode/settings.json");
        if project_path.is_file() {
            settings.overlay(read_json(&project_path)?);
        }

        let model_profiles = load_model_profiles(overrides)?;

        if model_profiles.is_empty() {
            bail!("~/.mcode/models.json 中没有可用模型；请配置 xai（Grok）、deepseek、glm 或 kimi");
        }

        let preferred_provider = settings.provider.as_deref();
        if let Some(provider) = preferred_provider {
            validate_supported_provider(provider)?;
        }
        let configured_model = overrides
            .model
            .clone()
            .or_else(|| env_non_empty("MCODE_MODEL"))
            .or_else(|| settings.model.clone());
        let model = configured_model
            .or_else(|| {
                let [only_profile] = model_profiles.as_slice() else {
                    return None;
                };
                Some(only_profile.qualified_id())
            })
            .context("未配置默认模型；请在 ~/.mcode/settings.json 中设置 defaultModel")?;
        let selected_profile = find_model_profile(&model_profiles, preferred_provider, &model)?
            .with_context(|| format!("模型 {model:?} 不在 ~/.mcode/models.json 中"))?;
        let selected_model = selected_profile.id.clone();
        let provider = selected_profile.provider.clone();
        let configured_web_search = overrides.web_search.or(settings.web_search);
        let api = selected_profile.api;

        let environment_reasoning = env_reasoning("MCODE_REASONING_EFFORT").transpose()?;
        let requested_reasoning_effort = overrides.reasoning_effort.or(environment_reasoning);
        let reasoning_effort = requested_reasoning_effort.map_or_else(
            || selected_profile.default_reasoning_effort(),
            |requested| selected_profile.clamp_reasoning_effort(requested),
        );
        let reasoning_value = selected_profile.reasoning_value(reasoning_effort)?;

        let base_url = selected_profile.base_url.clone();
        let api_key = selected_profile.api_key.clone();
        let context_window = selected_profile.context_window;
        let max_input_tokens = selected_profile.max_input_tokens;
        let max_output_tokens = selected_profile.max_output_tokens;
        let compat = selected_profile.compat;
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
        let mcp_servers = build_mcp_servers(settings.mcp_servers)?;

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
        if max_output_tokens.is_some_and(|tokens| tokens == 0 || tokens > context_window) {
            bail!(
                "max output tokens must be between 1 and the {context_window}-token context window"
            );
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
            max_output_tokens,
            compat,
            cwd,
            request_timeout_secs,
            compaction,
            web_search,
            model_profiles,
            mcp_servers,
            reload_overrides,
        })
    }

    pub fn select_model(&mut self, query: &str) -> Result<()> {
        let profile = find_model_profile(&self.model_profiles, Some(&self.provider), query)?
            .with_context(|| format!("模型 {query:?} 不在 ~/.mcode/models.json 中"))?
            .clone();
        self.reasoning_effort = profile.clamp_reasoning_effort(self.reasoning_effort);
        self.reasoning_value = profile.reasoning_value(self.reasoning_effort)?;
        self.model = profile.id;
        self.provider = profile.provider;
        self.api = profile.api;
        self.base_url = profile.base_url;
        self.api_key = profile.api_key;
        self.context_window = profile.context_window;
        self.max_input_tokens = profile.max_input_tokens;
        self.max_output_tokens = profile.max_output_tokens;
        self.compat = profile.compat;
        Ok(())
    }

    pub fn select_reasoning_effort(&mut self, effort: ReasoningEffort) -> Result<()> {
        let profile = find_model_profile(&self.model_profiles, Some(&self.provider), &self.model)?
            .with_context(|| format!("模型 {:?} 不在 ~/.mcode/models.json 中", self.model))?;
        let effective_effort = profile.clamp_reasoning_effort(effort);
        self.reasoning_value = profile.reasoning_value(effective_effort)?;
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

pub(crate) fn load_model_profiles(overrides: &ConfigOverrides) -> Result<Vec<ModelProfile>> {
    let base_url_override = overrides
        .base_url
        .clone()
        .or_else(|| env_non_empty("MCODE_BASE_URL"));
    let api_key_override = overrides
        .api_key_env
        .as_deref()
        .map(|name| {
            env_non_empty(name)
                .with_context(|| format!("环境变量 {name} 未配置或为空，无法用作 API 密钥"))
        })
        .transpose()?;
    let environment_context = env_u64("MCODE_CONTEXT_WINDOW").transpose()?;
    let context_window_override = overrides.context_window.or(environment_context);
    let environment_max_input = env_u64("MCODE_MAX_INPUT_TOKENS").transpose()?;
    let max_input_tokens_override = overrides.max_input_tokens.or(environment_max_input);
    let environment_max_output = env_u64("MCODE_MAX_OUTPUT_TOKENS").transpose()?;
    let max_output_tokens_override = overrides.max_output_tokens.or(environment_max_output);
    let mut profiles = if let Some(home) = mcode_home_dir() {
        let models_path = home.join("models.json");
        if models_path.is_file() {
            build_model_profiles(
                read_json::<ModelsFile>(&models_path)?,
                base_url_override.as_deref(),
                api_key_override.as_deref(),
                context_window_override,
                max_input_tokens_override,
                max_output_tokens_override,
            )?
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    profiles.sort_by(|left, right| (&left.provider, &left.id).cmp(&(&right.provider, &right.id)));
    Ok(profiles)
}

#[must_use]
pub fn mcode_home_dir() -> Option<PathBuf> {
    env::var_os("MCODE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".mcode")))
}

fn build_mcp_servers(servers: BTreeMap<String, McpServerFile>) -> Result<Vec<McpServerConfig>> {
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
            .filter(|command| !command.trim().is_empty())
            .with_context(|| format!("MCP server {name:?} is missing command"))?;
        let env = server
            .env
            .unwrap_or_default()
            .into_iter()
            .map(|(key, value)| {
                if key.trim().is_empty() {
                    bail!("MCP server {name:?} contains an empty environment variable name");
                }
                let value = interpolate_environment(&value, |variable| env::var(variable).ok())
                    .with_context(|| {
                        format!(
                            "MCP server {name:?} environment value {key:?} references a missing variable"
                        )
                    })?;
                Ok((key, value))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        configured.push(McpServerConfig {
            name,
            command,
            args: server.args.unwrap_or_default(),
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
    api_key_override: Option<&str>,
    context_window_override: Option<u64>,
    max_input_tokens_override: Option<u64>,
    max_output_tokens_override: Option<u64>,
) -> Result<Vec<ModelProfile>> {
    let mut profiles = Vec::new();
    for (provider_name, provider) in file.providers {
        validate_supported_provider(&provider_name)?;
        for model in provider.models {
            let api_name = model
                .api
                .as_deref()
                .or(provider.api.as_deref())
                .with_context(|| format!("provider {provider_name} is missing api"))?;
            let api = parse_api_protocol(api_name).with_context(|| {
                format!("provider {provider_name} uses unsupported api {api_name:?}")
            })?;
            if model.id.trim().is_empty() {
                bail!("provider {provider_name} contains a model with an empty id");
            }
            let base_url = base_url_override
                .map(ToString::to_string)
                .or_else(|| provider.base_url.clone())
                .with_context(|| {
                    format!("provider {provider_name:?} is missing baseUrl in models.json")
                })?;
            validate_http_base_url(&base_url, &format!("provider {provider_name} baseUrl"))?;
            let api_key = api_key_override.map_or_else(
                || {
                    provider
                        .api_key
                        .as_deref()
                        .and_then(resolve_static_config_value)
                },
                |value| Some(value.to_string()),
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
            let max_output_tokens = max_output_tokens_override.or(model.max_output_tokens);
            let compat = provider.compat.merge(model.compat);
            let reasoning = model.reasoning.unwrap_or(false);
            let input = model.input.unwrap_or_else(|| vec![InputModality::Text]);
            if !input.contains(&InputModality::Text) {
                bail!(
                    "model {}/{} must support text input",
                    provider_name,
                    model.id
                );
            }
            let supports_images = input.contains(&InputModality::Image);
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
            if max_output_tokens.is_some_and(|tokens| tokens == 0 || tokens > context_window) {
                bail!(
                    "model {}/{} has maxOutputTokens outside its 1..={context_window} context window",
                    provider_name,
                    model.id
                );
            }
            let supports_strict_tools = compat.strict_tools.unwrap_or(false);
            let profile = ModelProfile {
                provider: provider_name.clone(),
                id: model.id,
                name: model.name,
                api,
                base_url,
                api_key,
                context_window,
                max_input_tokens,
                max_output_tokens,
                reasoning,
                supports_images,
                compat: ModelCompat {
                    reasoning_effort: compat.reasoning_effort.unwrap_or(true),
                    usage_in_streaming: compat.usage_in_streaming.unwrap_or(true),
                    finish_reason: compat.finish_reason.unwrap_or(true),
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

fn validate_supported_provider(provider: &str) -> Result<()> {
    if SUPPORTED_PROVIDERS.contains(&provider) {
        return Ok(());
    }
    bail!("不支持 provider {provider:?}；MCode 仅支持 xai（Grok）、deepseek、glm 和 kimi")
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
    fn model_catalog_allows_only_the_supported_four() {
        let file = serde_json::from_str(include_str!("../models.example.json")).unwrap();
        let profiles = build_model_profiles(file, None, None, None, None, None).unwrap();
        let providers = profiles
            .iter()
            .map(|profile| profile.provider.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            providers,
            BTreeSet::from(["deepseek", "glm", "kimi", "xai"])
        );

        let file = serde_json::from_value(serde_json::json!({
            "providers": {
                "other": {
                    "baseUrl": "https://example.com/v1",
                    "api": "openai-completions",
                    "models": []
                }
            }
        }))
        .unwrap();
        let error = build_model_profiles(file, None, None, None, None, None)
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("仅支持 xai（Grok）、deepseek、glm 和 kimi")
        );
    }

    #[test]
    fn project_mcp_settings_merge_each_server_field() {
        let mut settings: SettingsFile = serde_json::from_value(serde_json::json!({
            "mcpServers": {
                "docs": {
                    "command": "npx",
                    "args": ["server"],
                    "env": {"TOKEN": "global", "KEEP": "yes"}
                }
            }
        }))
        .unwrap();
        let project: SettingsFile = serde_json::from_value(serde_json::json!({
            "mcpServers": {
                "docs": {
                    "env": {"TOKEN": "project"},
                    "enabled": true
                }
            }
        }))
        .unwrap();

        settings.overlay(project);
        let server = settings.mcp_servers.get("docs").unwrap();
        assert_eq!(server.command.as_deref(), Some("npx"));
        assert_eq!(server.args.as_deref().unwrap(), ["server"]);
        assert_eq!(
            server.env.as_ref().unwrap().get("TOKEN").unwrap(),
            "project"
        );
        assert_eq!(server.env.as_ref().unwrap().get("KEEP").unwrap(), "yes");
        assert_eq!(server.enabled, Some(true));
    }
}
