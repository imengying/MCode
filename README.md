# MCode

面向 Linux 的 Rust 终端 coding agent，提供 Codex 风格交互，支持 Grok、DeepSeek、GLM 和
Kimi。

## 功能

- 交互式 TUI 与非交互 `exec`
- `read_file`、`write_file`、`edit_file`、`shell` 内置工具
- 四家模型原生 Web Search 与网页正文提取
- 图片输入、Wayland/X11 剪贴板和本地 stdio MCP
- Markdown 与终端 Unicode 数学公式渲染
- Linux Bubblewrap 沙箱与 shell/MCP 执行审批
- Pi 风格自动上下文压缩和溢出恢复
- JSONL 会话、崩溃恢复、累计 token 与缓存命中统计
- 模型与 effort 切换
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

默认 shell 权限需要系统安装 `bubblewrap`（命令名 `bwrap`）：

```bash
# Arch Linux
sudo pacman -S bubblewrap

# Debian / Ubuntu
sudo apt install bubblewrap
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
mkdir -p ~/.mcode
curl -fsSL https://raw.githubusercontent.com/imengying/MCode/main/models.example.json \
  -o ~/.mcode/models.json
curl -fsSL https://raw.githubusercontent.com/imengying/MCode/main/settings.example.json \
  -o ~/.mcode/settings.json
export XAI_API_KEY="..."
mcode doctor
mcode
```

切换供应商时配置对应密钥和模型：

```bash
export DEEPSEEK_API_KEY="..."  # DeepSeek
export MOONSHOT_API_KEY="..."  # Kimi
export ZHIPUAI_API_KEY="..."   # GLM
MCODE_MODEL=deepseek/deepseek-v4-flash mcode
```

MCode 不会把 API 密钥写入配置或会话。

## 配置

| 文件 | 作用域 |
|---|---|
| `~/.mcode/settings.json` | 全局设置 |
| `.mcode/settings.json` | 项目设置，覆盖全局设置 |
| `~/.mcode/models.json` | 模型与端点 |

完整配置见 [settings.example.json](settings.example.json) 和
[models.example.json](models.example.json)。优先级为：命令行、环境变量、项目设置、全局设置、
单模型自动选择。`MCODE_HOME` 可覆盖默认的 `~/.mcode`。

常用配置：

- `contextWindow` / `maxInputTokens` / `maxOutputTokens`：上下文窗口、最大输入和可选输出上限；
  配置输出上限后可精确判断是否需要压缩重试
- `input`：模型输入模态；文本模型使用 `["text"]`
- `default`：每个模型必填，指定默认 effort；非推理模型使用 `off`
- `thinkingLevelMap`：只列出当前推理模型支持的 effort，`default` 必须是其中一个等级
- `baseUrl`：提供 OpenAI Responses API 的中转站根地址；MCode 会请求其 `/responses`
- `compaction`：自动压缩开关、预留 token 和最近历史预算
- `mcpServers`：本地 stdio MCP；项目配置按服务名称逐字段合并全局配置

MCode 只使用 OpenAI Responses API，不包含 Chat Completions 回退。`models.example.json`
只包含 Grok、DeepSeek、Kimi 和 GLM；其中 GLM/Kimi 的示例地址是中转站占位符，使用前需替换。
对应密钥环境变量为
`XAI_API_KEY`、`DEEPSEEK_API_KEY`、`MOONSHOT_API_KEY` 和 `ZHIPUAI_API_KEY`。Grok 4.6
使用 Responses API，支持 500K 上下文、图片输入以及 `low` / `medium` / `high` / `xhigh`
effort。DeepSeek V4 Flash 与 V4 Pro 均使用 Responses API，支持 1M 上下文、384K 最大输出
以及 `low` / `high` / `max` effort。未知 provider 会被拒绝。

`MCODE_MODEL` 和 `MCODE_BASE_URL` 可覆盖当前选择；`--api-key-env` 可临时指定密钥环境变量。

Web Search 默认开启，不需要配置。四家模型均通过 Responses API 使用原生 `web_search`，
并可使用 `fetch_content` 读取公开网页正文；中转站需要实现对应工具协议。

MCode 会从 Git 仓库根目录到当前工作目录逐层读取普通 UTF-8 `AGENTS.md`；同层的
`AGENTS.override.md` 优先，合计大小上限 64 KiB。不会读取仓库外文件、目录项或符号链接。
除此之外，MCode 不注入身份、语气或工作流提示；工具能力仅通过标准工具 schema 提供。

## 使用

```bash
# 交互模式
mcode
mcode "检查当前项目"
mcode -i screenshot.png "检查这个界面"

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

`mcode resume` 在当前项目只有一个会话时直接恢复；有多个会话时打开选择器。也可传入
会话 ID，或使用 `mcode resume last` 明确恢复最新会话。

仅在完全可信、已隔离的自动化环境中使用：

```bash
mcode exec --dangerously-bypass-approvals "运行并修复测试"
```

TUI 命令：

| 命令 | 作用 |
|---|---|
| `/model` | 依次选择模型与思考等级 |
| `/permissions` | 选择只读、工作区可写或完全访问 |
| `/diff` | 显示当前 Git 改动 |
| `/review [FOCUS]` | 审查当前工作区改动 |
| `/compact [INSTRUCTIONS]` | 手动压缩上下文 |
| `/status` | 显示模型、端点和 token |
| `/new` | 新建会话 |
| `/resume` | 选择并恢复其他会话 |
| `/delete` | 选择 Yes 后删除当前会话并退出 |
| `/clear` | 清屏 |
| `/help` | 显示帮助 |
| `/exit` | 退出 |

输入 `/` 后可用方向键选择，Tab 补全，Enter 补全并立即执行命令。输入 `@` 可模糊查找并
引用工作区文件。Enter 或 Tab 提交；
任务运行中则排队为后续消息。Shift+Enter 或 Alt+Enter 换行，Escape 先关闭命令候选，再取消当前任务。
`Ctrl+C` 先清空草稿，空草稿时一秒内再按一次退出。使用 `Ctrl+V` 粘贴系统剪贴板中的图片或文本；
Wayland 不可用或报错时自动回退 X11。也可将单个图片文件拖入终端；启动时可用
`-i/--image` 附加图片。使用上下方向键恢复历史输入；对话记录使用终端原生回滚，可直接用
鼠标选择、复制和滚动。
删除确认默认选择 No，
使用方向键切换并按 Enter 确认。审批提示中 `y` 允许一次，`a` 在当前会话允许当前 shell
命令或同名 MCP 工具，`n` 拒绝。

## 会话恢复

会话保存在 `$MCODE_HOME/sessions/<project-hash>/`，文件名采用
`rollout-YYYY-MM-DDTHH-MM-SS-<UUIDv7>.jsonl`。同一会话只允许一个写进程。

自动压缩保留最近历史并摘要更早内容；失败时退回硬裁剪。中断恢复时，`read_file` 可以重放；
写文件、编辑、shell 和 MCP 不会重放未知结果，模型会收到 synthetic error 后继续处理。
超过上下文保留上限的工具输出会以首尾摘要写入 JSONL，完整内容保存在对应会话目录的
`tool-results/` 中；删除会话时一并删除。

## 安全

- 文件工具拒绝工作目录外路径，写入使用原子替换。
- 网页抓取限制 2 MiB，逐跳校验重定向和 DNS/IP，拒绝内网地址。
- 默认 shell 通过 Bubblewrap 禁用网络，并只允许写当前项目与 `/tmp`；`/permissions`
  可切换为只读或完全访问。Bubblewrap 不可用时安全失败，不会静默降级。
- shell 和 MCP 仍需审批；“本次会话内始终允许”会在权限档位改变后清空。
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

默认测试不联网。GitHub Actions 仅保留 Release workflow：`v*` tag 发布 Linux glibc
x86_64/ARM64 二进制归档，并用 tag 之间的 commit 生成更新日志；发布二进制的版本取自 tag，
源码归档由 GitHub 自动提供。
