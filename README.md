# MCode

[![CI](https://github.com/imengying/MCode/actions/workflows/ci.yml/badge.svg)](https://github.com/imengying/MCode/actions/workflows/ci.yml)

面向 Linux 的 Rust 终端 coding agent，提供 Codex 风格交互，支持 OpenAI-compatible
Chat Completions 和 Responses API。

## 功能

- 交互式 TUI 与非交互 `exec`
- `read_file`、`write_file`、`edit_file`、`shell` 内置工具
- Responses 托管搜索，兼容端点本地 Web Search 与网页正文提取
- 图片输入和本地 stdio MCP
- shell/MCP 执行审批
- Pi 风格自动上下文压缩和溢出恢复
- JSONL v3 会话、崩溃恢复和累计 token 统计
- 模型、reasoning 和搜索模式独立切换
- 当前正式支持 Linux x86_64 与 ARM64

## 安装

安装最新 Release 到 `~/.local/bin`：

```bash
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/imengying/MCode/main/install.sh | sh
```

安装脚本会自动识别架构。自定义版本或目录：

```bash
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/imengying/MCode/main/install.sh | \
  MCODE_VERSION=0.1.0 MCODE_INSTALL_DIR="$HOME/bin" sh
```

显式升级到最新正式版：

```bash
mcode update
```

若 zsh 尚未包含默认安装目录：

```zsh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

从源码构建需要当前正式版 Rust，本版本以 Rust 1.97 验证：

```bash
cargo build --release --locked
cargo install --path . --locked
```

## 快速开始

```bash
export OPENAI_API_KEY="..."
mcode doctor
mcode
```

连接兼容端点：

```bash
export OPENAI_MODEL="example-model"
export OPENAI_BASE_URL="https://api.example.com/v1"
mcode
```

API key 可以省略，以连接不需要认证的端点。MCode 不会保存凭据。

## 配置

| 文件 | 作用域 |
|---|---|
| `~/.mcode/agent/settings.json` | 全局设置 |
| `.mcode/settings.json` | 项目设置，覆盖全局设置 |
| `~/.mcode/agent/models.json` | 模型与端点 |

完整配置见 [settings.example.json](settings.example.json) 和
[models.example.json](models.example.json)。优先级为：命令行、环境变量、项目设置、全局设置、
内置默认值。`MCODE_HOME` 可覆盖默认的 `~/.mcode`。

常用配置：

- `api`：`openai-completions` 或 `openai-responses`
- `contextWindow` / `maxInputTokens`：上下文窗口与最大输入
- `compat`：控制 reasoning effort、流式 usage 和 strict tools
- `webSearch`：`disabled`、`cached` 或 `live`
- `compaction`：自动压缩开关、预留 token 和最近历史预算
- `mcpServers`：本地 stdio MCP；项目配置按名称覆盖全局配置

`models.example.json` 已包含 OpenAI、Grok、DeepSeek、Kimi、GLM 和通用
OpenAI-compatible profile。对应密钥环境变量为 `OPENAI_API_KEY`、`XAI_API_KEY`、
`DEEPSEEK_API_KEY`、`MOONSHOT_API_KEY` 和 `ZHIPUAI_API_KEY`。

Web Search 依 API 协议分流：

- `openai-responses` 使用模型供应商的托管搜索，并提供 `fetch_content`。
- `openai-completions` 提供本地 `web_search` 和 `fetch_content`。
- 本地搜索支持 Exa 零配置 MCP、`EXA_API_KEY`、`BRAVE_API_KEY` 或
  `SEARXNG_BASE_URL`；`provider: "auto"` 会按可用性回退。

当前工作目录中的普通 UTF-8 `AGENTS.md` 会加入 system instructions，大小上限 64 KiB；不会
读取父目录、目录项或符号链接。

## 使用

```bash
# 交互模式
mcode
mcode "检查当前项目"
mcode -i screenshot.png "检查这个界面"
mcode --search "检查依赖的最新版本"

# 非交互模式
mcode exec "修复失败的测试"
git diff | mcode exec "审查这个 diff"
mcode exec --json "说明 src 目录"

# 会话与诊断
mcode resume
mcode resume <SESSION_ID>
mcode sessions --json
mcode doctor --json
mcode update
mcode delete <SESSION_ID> --force
```

仅在完全可信、已隔离的自动化环境中使用：

```bash
mcode exec --dangerously-bypass-approvals "运行并修复测试"
```

TUI 命令：

| 命令 | 作用 |
|---|---|
| `/model [provider/model]` | 查看或切换模型 |
| `/reasoning [LEVEL]` | 查看或切换思考强度 |
| `/thinking [show\|hide]` | 展开或折叠已完成的思考过程 |
| `/search [disabled\|cached\|live]` | 查看或切换网页搜索 |
| `/compact [INSTRUCTIONS]` | 手动压缩上下文 |
| `/image [PATH\|clear]` | 管理下一条消息的图片 |
| `/status` | 显示模型、端点和 token |
| `/new` | 新建会话 |
| `/delete` | 二次确认后删除当前会话 |
| `/clear` | 清屏 |
| `/help` | 显示帮助 |
| `/exit` | 退出 |

思考过程在生成时展开，完成后自动折叠；使用 `/thinking show` 可查看全文。

Enter 提交，Shift+Enter 或 Alt+Enter 换行，Escape 取消当前任务。审批提示中 `y` 允许一次，
`a` 在当前会话允许同名工具，`n` 拒绝。

## 会话恢复

会话保存在 `$MCODE_HOME/sessions/<project-hash>/`，文件名采用
`rollout-YYYY-MM-DDTHH-MM-SS-<UUIDv7>.jsonl`。同一会话只允许一个写进程。

自动压缩保留最近历史并摘要更早内容；失败时退回硬裁剪。中断恢复时，`read_file` 可以重放；
写文件、编辑、shell 和 MCP 不会重放未知结果，模型会收到 synthetic error 后继续处理。

## 安全

- 文件工具拒绝工作目录外路径，写入使用原子替换。
- 网页抓取限制 2 MiB，逐跳校验重定向和 DNS/IP，拒绝内网地址。
- shell 和 MCP 以当前用户权限运行；审批不是系统沙箱。
- MCP 服务器、`AGENTS.md` 和网页搜索结果都应视为不可信输入。
- `resume` 只接受当前项目的会话，并校验会话工作目录。
- 管道输入或 JSON 模式无法询问时，危险工具默认拒绝执行。

## 开发

```bash
cargo fmt --all -- --check
sh -n install.sh
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
cargo build --release --locked
```

默认测试不联网。真实 API 测试会产生费用，必须显式运行：

```bash
MCODE_REAL_API_MODEL=gpt-5.6-terra \
  cargo test --test real_api -- --ignored --nocapture
```

GitHub Actions 在 Linux 上执行格式、Clippy、RustSec、测试和 release 构建。`v*` tag 只触发
Release workflow，发布 Linux musl x86_64/ARM64 二进制归档，并用 tag 之间的 commit 生成更新
日志；发布二进制的版本取自 tag，源码归档由 GitHub 自动提供。
