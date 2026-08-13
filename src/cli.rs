use std::path::PathBuf;

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};

use crate::config::ReasoningEffort;

const HELP_TEMPLATE: &str = "{before-help}{about-with-newline}\
用法：{usage}\n\n\
{all-args}{after-help}";

#[derive(Debug, Parser)]
#[command(
    name = "mcode",
    version = crate::VERSION,
    about = "MCode，支持 Grok、DeepSeek、GLM 和 Kimi 的终端编码 Agent",
    args_conflicts_with_subcommands = true,
    after_help = "示例：\n  mcode\n  mcode -i screenshot.png \"检查这个界面\"\n  mcode exec \"修复失败的测试\"\n  mcode resume\n  mcode resume <SESSION_ID>\n  mcode sessions\n  mcode doctor\n  mcode update\n  mcode delete <SESSION_ID>"
)]
pub struct Cli {
    /// 附加到初始提示词的图片。
    #[arg(
        long = "image",
        short = 'i',
        value_name = "文件",
        value_delimiter = ',',
        global = true
    )]
    pub images: Vec<PathBuf>,

    /// models.json 中配置的模型名称。
    #[arg(short = 'm', long, global = true, value_name = "模型")]
    pub model: Option<String>,

    /// 独立于模型选择的思考强度：off、minimal、low、medium、high、xhigh 或 max。
    #[arg(
        short = 'r',
        long = "reasoning",
        global = true,
        value_enum,
        value_name = "级别",
        hide_possible_values = true
    )]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// 覆盖所选供应商的 API 根地址。
    #[arg(long, global = true, value_name = "URL")]
    pub base_url: Option<String>,

    /// 保存 API 密钥的环境变量。
    #[arg(long, global = true, value_name = "环境变量")]
    pub api_key_env: Option<String>,

    /// 用于状态和上下文用量计算的上下文窗口。
    #[arg(long, global = true, value_name = "TOKEN数")]
    pub context_window: Option<u64>,

    /// 模型可接受的最大输入 token 数。
    #[arg(long, global = true, value_name = "TOKEN数")]
    pub max_input_tokens: Option<u64>,

    /// 单次模型响应允许的最大输出 token 数。
    #[arg(long, global = true, value_name = "TOKEN数")]
    pub max_output_tokens: Option<u64>,

    /// 以指定目录作为工作目录运行。
    #[arg(short = 'C', long = "cd", global = true, value_name = "目录")]
    pub cwd: Option<PathBuf>,

    /// 不保存新建的交互式会话。
    #[arg(long, global = true)]
    pub no_session: bool,

    /// 供应商请求超时时间（秒）。
    #[arg(long, global = true, value_name = "秒")]
    pub request_timeout: Option<u64>,

    /// 无需确认即可运行 shell 和 MCP 工具，并关闭系统沙箱。
    #[arg(long = "dangerously-bypass-approvals", global = true)]
    pub dangerously_bypass_approvals: bool,

    #[command(subcommand)]
    pub command: Option<Command>,

    /// 交互模式的可选初始提示词。
    #[arg(value_name = "提示词", trailing_var_arg = true)]
    pub prompt: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// 非交互式运行提示词。
    Exec(ExecArgs),

    /// 恢复最新或指定的会话。
    Resume(ResumeArgs),

    /// 永久删除已保存的会话。
    Delete(DeleteArgs),

    /// 列出当前工作目录保存的会话。
    Sessions(OutputArgs),

    /// 诊断本地配置，不发起 API 请求。
    Doctor(OutputArgs),

    /// 安装最新的 GitHub Release。
    Update,
}

#[derive(Debug, Args)]
pub struct OutputArgs {
    /// 输出机器可读的 JSON。
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct ExecArgs {
    /// 输出逐行 JSON 事件。
    #[arg(long)]
    pub json: bool,

    /// 提示词文本；使用 '-' 或省略以读取标准输入。
    #[arg(value_name = "提示词", trailing_var_arg = true)]
    pub prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub struct ResumeArgs {
    /// 会话 ID 片段、JSONL 路径或 'last'；多会话时省略将打开选择器。
    #[arg(value_name = "会话")]
    pub session: Option<String>,

    /// 恢复后立即提交的可选提示词。
    #[arg(value_name = "提示词", trailing_var_arg = true)]
    pub prompt: Vec<String>,
}

#[derive(Debug, Args)]
pub struct DeleteArgs {
    /// 会话 UUID 或无歧义的 UUID 片段。
    #[arg(value_name = "会话")]
    pub session: String,

    /// 不经确认直接删除；SESSION 必须是完整 UUID。
    #[arg(long)]
    pub force: bool,
}

impl Cli {
    #[must_use]
    pub fn parse_localized() -> Self {
        let matches = localized_command(Self::command()).get_matches();
        Self::from_arg_matches(&matches).unwrap_or_else(|error| error.exit())
    }

    #[must_use]
    pub fn root_prompt(&self) -> Option<String> {
        join_prompt(&self.prompt)
    }
}

fn localized_command(mut command: clap::Command) -> clap::Command {
    command.build();
    let has_help = command
        .get_arguments()
        .any(|argument| argument.get_id() == "help");
    let has_version = command
        .get_arguments()
        .any(|argument| argument.get_id() == "version");
    let mut command = command
        .help_template(HELP_TEMPLATE)
        .subcommand_help_heading("命令")
        .subcommand_value_name("命令")
        .mut_args(|argument| {
            let heading = if argument.is_positional() {
                "参数"
            } else {
                "选项"
            };
            argument.help_heading(heading)
        });
    if has_help {
        command = command.mut_arg("help", |argument| {
            argument.help("显示帮助").help_heading("选项")
        });
    }
    if has_version {
        command = command.mut_arg("version", |argument| {
            argument.help("显示版本").help_heading("选项")
        });
    }
    command.mut_subcommands(|subcommand| {
        if subcommand.get_name() == "help" {
            subcommand
                .about("显示指定命令的帮助")
                .help_template(HELP_TEMPLATE)
        } else {
            localized_command(subcommand)
        }
    })
}

#[must_use]
pub fn join_prompt(parts: &[String]) -> Option<String> {
    let prompt = parts.join(" ");
    (!prompt.trim().is_empty()).then_some(prompt)
}
