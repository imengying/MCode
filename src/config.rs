use std::collections::BTreeMap;
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
    pub reasoning: bool,
    pub supports_reasoning_effort: bool,
    pub supports_usage_in_streaming: bool,
    thinking_level_map: BTreeMap<ReasoningEffort, Option<String>>,
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
        match self.thinking_level_map.get(&effort) {
            Some(Some(value)) if value.trim().is_empty() => {
                bail!(
                    "model {} maps {effort} reasoning to an empty value",
                    self.id
                )
            }
            Some(Some(value)) => Ok(Some(value.clone())),
            Some(None) => bail!("model {} does not support {effort} reasoning", self.id),
            None if effort == ReasoningEffort::Off => Ok(None),
            None => Ok(Some(effort.as_str().to_string())),
        }
    }

    #[must_use]
    pub fn supports_reasoning(&self, effort: ReasoningEffort) -> bool {
        if !self.reasoning {
            return effort == ReasoningEffort::Off;
        }
        match self.thinking_level_map.get(&effort) {
            Some(None) => false,
            Some(Some(_)) => true,
            None => !matches!(effort, ReasoningEffort::Xhigh | ReasoningEffort::Max),
        }
    }

    #[must_use]
    pub fn supported_reasoning_efforts(&self) -> Vec<ReasoningEffort> {
        ReasoningEffort::ALL
            .into_iter()
            .filter(|effort| self.supports_reasoning(*effort))
            .collect()
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
    pub reasoning_effort: ReasoningEffort,
    pub reasoning_value: Option<String>,
    pub base_url: String,
    pub api_key: Option<String>,
    pub context_window: u64,
    pub supports_reasoning_effort: bool,
    pub supports_usage_in_streaming: bool,
    pub cwd: PathBuf,
    pub max_tool_turns: usize,
    pub request_timeout_secs: u64,
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
    pub cwd: Option<PathBuf>,
    pub max_tool_turns: Option<usize>,
    pub request_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct SettingsFile {
    #[serde(rename = "defaultProvider")]
    provider: Option<String>,
    #[serde(rename = "defaultModel")]
    model: Option<String>,
    #[serde(rename = "defaultThinkingLevel")]
    thinking_level: Option<ReasoningEffort>,
    #[serde(default, rename = "mcpServers")]
    mcp_servers: BTreeMap<String, McpServerFile>,
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
        if other.thinking_level.is_some() {
            self.thinking_level = other.thinking_level;
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
    context_window: Option<u64>,
    #[serde(default)]
    compat: CompatFile,
    #[serde(default)]
    thinking_level_map: BTreeMap<String, Option<String>>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CompatFile {
    supports_reasoning_effort: Option<bool>,
    supports_usage_in_streaming: Option<bool>,
}

impl CompatFile {
    fn merge(self, model: Self) -> Self {
        Self {
            supports_reasoning_effort: model
                .supports_reasoning_effort
                .or(self.supports_reasoning_effort),
            supports_usage_in_streaming: model
                .supports_usage_in_streaming
                .or(self.supports_usage_in_streaming),
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

        let home = dirs::home_dir();
        let mut settings = SettingsFile::default();
        if let Some(home) = &home {
            let global_path = home.join(".mcode/agent/settings.json");
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
        let mut model_profiles = if let Some(home) = &home {
            let models_path = home.join(".mcode/agent/models.json");
            if models_path.is_file() {
                build_model_profiles(
                    read_json::<ModelsFile>(&models_path)?,
                    base_url_override.as_deref(),
                    overrides.api_key_env.as_ref().map(|_| &forced_api_key),
                    fallback_api_key.as_ref(),
                    context_window_override,
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

        let environment_reasoning = env_reasoning("OPENAI_REASONING_EFFORT").transpose()?;
        let requested_reasoning_effort = overrides
            .reasoning_effort
            .or(environment_reasoning)
            .or(settings.thinking_level)
            .unwrap_or_default();
        let reasoning_effort = selected_profile.map_or(requested_reasoning_effort, |profile| {
            profile.clamp_reasoning_effort(requested_reasoning_effort)
        });
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
            selected_profile
                .and_then(|profile| profile.api_key.clone())
                .or(fallback_api_key)
        };
        let context_window = context_window_override
            .or_else(|| selected_profile.map(|profile| profile.context_window))
            .unwrap_or(DEFAULT_CONTEXT_WINDOW);
        let supports_reasoning_effort =
            selected_profile.is_none_or(|profile| profile.supports_reasoning_effort);
        let supports_usage_in_streaming =
            selected_profile.is_none_or(|profile| profile.supports_usage_in_streaming);
        let max_tool_turns = overrides.max_tool_turns.unwrap_or(32);
        let request_timeout_secs = overrides.request_timeout_secs.unwrap_or(300);
        let mcp_servers = build_mcp_servers(&settings.mcp_servers)?;

        if selected_model.trim().is_empty() {
            bail!("model cannot be empty");
        }
        if context_window == 0 {
            bail!("context window must be at least 1");
        }
        if max_tool_turns == 0 {
            bail!("max tool turns must be at least 1");
        }
        if request_timeout_secs == 0 {
            bail!("request timeout must be at least 1 second");
        }

        Ok(Self {
            model: selected_model,
            provider,
            reasoning_effort,
            reasoning_value,
            base_url,
            api_key,
            context_window,
            supports_reasoning_effort,
            supports_usage_in_streaming,
            cwd,
            max_tool_turns,
            request_timeout_secs,
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
            self.base_url = profile.base_url;
            self.api_key = profile.api_key;
            self.context_window = profile.context_window;
            self.supports_reasoning_effort = profile.supports_reasoning_effort;
            self.supports_usage_in_streaming = profile.supports_usage_in_streaming;
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
    if let Some(provider) = preferred_provider {
        if let Some(profile) = profiles.iter().find(|profile| {
            profile.provider == provider
                && (profile.id == query
                    || profile
                        .name
                        .as_deref()
                        .is_some_and(|name| name.eq_ignore_ascii_case(query)))
        }) {
            return Ok(Some(profile));
        }
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
) -> Result<Vec<ModelProfile>> {
    let mut profiles = Vec::new();
    for (provider_name, provider) in file.providers {
        for model in provider.models {
            let api = model.api.as_deref().or(provider.api.as_deref());
            if api != Some("openai-completions") {
                continue;
            }
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
                || {
                    provider
                        .api_key
                        .as_deref()
                        .and_then(resolve_static_config_value)
                        .or_else(|| fallback_api_key.cloned())
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
            let compat = provider.compat.merge(model.compat);
            if context_window == 0 {
                bail!(
                    "model {}/{} has a zero contextWindow",
                    provider_name,
                    model.id
                );
            }
            profiles.push(ModelProfile {
                provider: provider_name.clone(),
                id: model.id,
                name: model.name,
                base_url,
                api_key,
                context_window,
                reasoning: model.reasoning.unwrap_or(false),
                supports_reasoning_effort: compat.supports_reasoning_effort.unwrap_or(true),
                supports_usage_in_streaming: compat.supports_usage_in_streaming.unwrap_or(true),
                thinking_level_map,
            });
        }
    }
    Ok(profiles)
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
            thinking_level: Some(ReasoningEffort::Low),
            mcp_servers: BTreeMap::new(),
        };
        base.overlay(SettingsFile {
            model: Some("model-two".into()),
            thinking_level: Some(ReasoningEffort::High),
            ..SettingsFile::default()
        });
        assert_eq!(base.provider.as_deref(), Some("one"));
        assert_eq!(base.model.as_deref(), Some("model-two"));
        assert_eq!(base.thinking_level, Some(ReasoningEffort::High));
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
                            "contextWindow": 200000,
                            "compat": {"supportsReasoningEffort": true},
                            "thinkingLevelMap": {"high": "high", "max": null}
                        }]
                    }
                }
            }"#,
        )
        .unwrap();
        let profiles = build_model_profiles(file, None, None, None, None).unwrap();
        let profile = &profiles[0];
        assert_eq!(profile.qualified_id(), "proxy/coder");
        assert_eq!(profile.context_window, 200_000);
        assert!(profile.reasoning);
        assert!(profile.supports_reasoning_effort);
        assert!(!profile.supports_usage_in_streaming);
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
    fn environment_interpolation_matches_pi_static_values() {
        let value = interpolate_environment("Bearer ${TOKEN}_$SUFFIX $$ $!", |name| match name {
            "TOKEN" => Some("abc".to_string()),
            "SUFFIX" => Some("prod".to_string()),
            _ => None,
        });
        assert_eq!(value.as_deref(), Some("Bearer abc_prod $ !"));
    }

    #[test]
    fn model_resolution_requires_provider_for_ambiguous_ids() {
        let profile = |provider: &str| ModelProfile {
            provider: provider.to_string(),
            id: "same".to_string(),
            name: None,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            context_window: DEFAULT_CONTEXT_WINDOW,
            reasoning: false,
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
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
            base_url: "https://proxy.test/v1".to_string(),
            api_key: Some("secret".to_string()),
            context_window: 200_000,
            reasoning: true,
            supports_reasoning_effort: false,
            supports_usage_in_streaming: false,
            thinking_level_map: map,
        };
        let mut config = AppConfig {
            model: "old-model".to_string(),
            provider: None,
            reasoning_effort: ReasoningEffort::High,
            reasoning_value: Some("high".to_string()),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            context_window: DEFAULT_CONTEXT_WINDOW,
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
            cwd: PathBuf::from("."),
            max_tool_turns: 32,
            request_timeout_secs: 300,
            model_profiles: vec![profile],
            mcp_servers: Vec::new(),
        };

        config.select_model("proxy/coder").unwrap();
        assert_eq!(config.model, "coder");
        assert_eq!(config.provider.as_deref(), Some("proxy"));
        assert_eq!(config.base_url, "https://proxy.test/v1");
        assert_eq!(config.context_window, 200_000);
        assert_eq!(config.reasoning_value.as_deref(), Some("deep"));
        assert!(!config.supports_reasoning_effort);
        assert!(!config.supports_usage_in_streaming);
    }

    #[test]
    fn non_reasoning_profile_rejects_reasoning_effort() {
        let profile = ModelProfile {
            provider: "proxy".to_string(),
            id: "plain".to_string(),
            name: None,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            context_window: DEFAULT_CONTEXT_WINDOW,
            reasoning: false,
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
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
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            context_window: DEFAULT_CONTEXT_WINDOW,
            reasoning: true,
            supports_reasoning_effort: true,
            supports_usage_in_streaming: true,
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
