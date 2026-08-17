use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

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

#[derive(Clone)]
pub struct ModelProfile {
    pub provider: String,
    pub id: String,
    pub name: Option<String>,
    pub base_url: String,
    pub api_key: Option<String>,
    pub context_window: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: Option<u64>,
    pub supports_images: bool,
    default_reasoning_effort: ReasoningEffort,
    thinking_level_map: BTreeMap<ReasoningEffort, String>,
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
        Ok(self.thinking_level_map.get(&effort).cloned())
    }

    #[must_use]
    pub fn supports_reasoning(&self, effort: ReasoningEffort) -> bool {
        effort == ReasoningEffort::Off && self.thinking_level_map.is_empty()
            || self.thinking_level_map.contains_key(&effort)
    }

    #[must_use]
    pub fn supported_reasoning_efforts(&self) -> Vec<ReasoningEffort> {
        if self.thinking_level_map.is_empty() {
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
    pub reasoning_effort: ReasoningEffort,
    pub reasoning_value: Option<String>,
    pub base_url: String,
    pub api_key: Option<String>,
    pub context_window: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: Option<u64>,
    pub cwd: PathBuf,
    pub request_timeout_secs: u64,
    pub compaction: CompactionSettings,
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
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub cwd: Option<PathBuf>,
    pub request_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct SettingsFile {
    #[serde(rename = "defaultProvider")]
    provider: Option<String>,
    #[serde(rename = "defaultModel")]
    model: Option<String>,
    #[serde(default, rename = "mcpServers")]
    mcp_servers: BTreeMap<String, McpServerFile>,
    #[serde(default)]
    compaction: CompactionSettingsFile,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct CompactionSettingsFile {
    enabled: Option<bool>,
    reserve_tokens: Option<u64>,
    keep_recent_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
struct ModelsFile {
    #[serde(default)]
    providers: BTreeMap<String, ProviderFile>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ProviderFile {
    base_url: Option<String>,
    api_key: Option<String>,
    #[serde(default)]
    models: Vec<ModelFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ModelFile {
    id: String,
    name: Option<String>,
    input: Option<Vec<InputModality>>,
    default: ReasoningEffort,
    context_window: Option<u64>,
    max_input_tokens: Option<u64>,
    max_output_tokens: Option<u64>,
    #[serde(default)]
    thinking_level_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum InputModality {
    Text,
    Image,
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

        let reasoning_effort = selected_profile.default_reasoning_effort();
        let reasoning_value = selected_profile.reasoning_value(reasoning_effort)?;

        let base_url = selected_profile.base_url.clone();
        let api_key = selected_profile.api_key.clone();
        let context_window = selected_profile.context_window;
        let max_input_tokens = selected_profile.max_input_tokens;
        let max_output_tokens = selected_profile.max_output_tokens;
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
        Ok(Self {
            model: selected_model,
            provider,
            reasoning_effort,
            reasoning_value,
            base_url,
            api_key,
            context_window,
            max_input_tokens,
            max_output_tokens,
            cwd,
            request_timeout_secs,
            compaction,
            model_profiles,
            mcp_servers,
            reload_overrides,
        })
    }

    pub fn select_model(&mut self, query: &str, reasoning_effort: ReasoningEffort) -> Result<()> {
        let profile = find_model_profile(&self.model_profiles, Some(&self.provider), query)?
            .with_context(|| format!("模型 {query:?} 不在 ~/.mcode/models.json 中"))?
            .clone();
        let reasoning_effort = profile.clamp_reasoning_effort(reasoning_effort);
        let reasoning_value = profile.reasoning_value(reasoning_effort)?;
        self.model = profile.id;
        self.provider = profile.provider;
        self.reasoning_effort = reasoning_effort;
        self.reasoning_value = reasoning_value;
        self.base_url = profile.base_url;
        self.api_key = profile.api_key;
        self.context_window = profile.context_window;
        self.max_input_tokens = profile.max_input_tokens;
        self.max_output_tokens = profile.max_output_tokens;
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
    let mut profiles = if let Some(home) = mcode_home_dir() {
        let models_path = home.join("models.json");
        if models_path.is_file() {
            build_model_profiles(
                read_json::<ModelsFile>(&models_path)?,
                base_url_override.as_deref(),
                api_key_override.as_deref(),
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
) -> Result<Vec<ModelProfile>> {
    let mut profiles = Vec::new();
    for (provider_name, provider) in file.providers {
        validate_supported_provider(&provider_name)?;
        for model in provider.models {
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
                if level == ReasoningEffort::Off {
                    bail!(
                        "thinkingLevelMap for {}/{} must not contain off",
                        provider_name,
                        model.id
                    );
                }
                if value.trim().is_empty() {
                    bail!(
                        "thinkingLevelMap for {}/{} maps {level} to an empty value",
                        provider_name,
                        model.id
                    );
                }
                thinking_level_map.insert(level, value);
            }
            let context_window = model.context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW);
            let max_input_tokens = model.max_input_tokens.unwrap_or(context_window);
            let max_output_tokens = model.max_output_tokens;
            let input = model.input.unwrap_or_else(|| vec![InputModality::Text]);
            if !input.contains(&InputModality::Text) {
                bail!(
                    "model {}/{} must support text input",
                    provider_name,
                    model.id
                );
            }
            let supports_images = input.contains(&InputModality::Image);
            let default_reasoning_effort = model.default;
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
            let profile = ModelProfile {
                provider: provider_name.clone(),
                id: model.id,
                name: model.name,
                base_url,
                api_key,
                context_window,
                max_input_tokens,
                max_output_tokens,
                supports_images,
                default_reasoning_effort,
                thinking_level_map,
            };
            let configured_efforts = profile.supported_reasoning_efforts();
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
    match provider {
        "xai" | "deepseek" | "glm" | "kimi" => Ok(()),
        _ => bail!("不支持 provider {provider:?}；MCode 仅支持 xai（Grok）、deepseek、glm 和 kimi"),
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
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn model_catalog_allows_only_the_supported_four() {
        let file = serde_json::from_str(include_str!("../models.example.json")).unwrap();
        let profiles = build_model_profiles(file, None, None).unwrap();
        let providers = profiles
            .iter()
            .map(|profile| profile.provider.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            providers,
            BTreeSet::from(["deepseek", "glm", "kimi", "xai"])
        );
        let grok = profiles
            .iter()
            .find(|profile| profile.provider == "xai")
            .unwrap();
        assert_eq!(grok.id, "grok-4.6");
        assert_eq!(grok.context_window, 500_000);
        assert_eq!(grok.max_input_tokens, 500_000);
        assert_eq!(grok.max_output_tokens, None);
        assert!(grok.supports_images);
        assert_eq!(
            grok.supported_reasoning_efforts(),
            vec![
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
                ReasoningEffort::Xhigh
            ]
        );
        for profile in profiles
            .iter()
            .filter(|profile| profile.provider == "deepseek")
        {
            assert_eq!(profile.context_window, 1_048_576);
            assert_eq!(profile.max_input_tokens, 1_048_576);
            assert_eq!(profile.max_output_tokens, Some(384_000));
            assert_eq!(
                profile.supported_reasoning_efforts(),
                vec![
                    ReasoningEffort::Low,
                    ReasoningEffort::High,
                    ReasoningEffort::Max
                ]
            );
        }

        let file = serde_json::from_value(serde_json::json!({
            "providers": {
                "other": {
                    "baseUrl": "https://example.com/v1",
                    "models": []
                }
            }
        }))
        .unwrap();
        let error = build_model_profiles(file, None, None).err().unwrap();
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
