# MCode

一个专注、可审计的 Rust 终端 coding agent。交互和命令采用 Codex 风格，只连接
OpenAI-compatible Chat Completions API。

## 当前范围

- `mcode`：启动交互式 TUI
- `mcode exec`：执行一次非交互任务
- `mcode resume`：恢复当前项目最近或指定的会话
- `mcode delete`：永久删除当前项目的指定会话
- OpenAI `/v1/chat/completions` 流式 SSE 与 function calling
- `read_file`、`write_file`、`edit_file`、`shell` 四个内置工具
- JSONL 会话、流式响应、工具状态、滚动历史和多行编辑
- PNG、JPEG、GIF、WebP 图片输入
- 本地 stdio MCP 服务器与工具发现
- 模型与 reasoning 独立选择，思考内容单独渲染
- 底栏显示模型、reasoning、上下文占用和输入/输出 token

项目刻意不包含 OAuth、非 OpenAI Provider API、本地 LLM、Responses API、远程 MCP transport、
扩展系统、技能市场和主题系统。

## 构建

~~~bash
cargo build --release
./target/release/mcode --help
~~~

最低 Rust 版本为 1.87。

也可以安装到 Cargo bin 目录：

~~~bash
cargo install --path .
~~~

## 配置

最低配置：

~~~bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_REASONING_EFFORT="high"
export OPENAI_CONTEXT_WINDOW="128000"
~~~

API key 可以省略，以连接不需要认证的兼容服务。项目不会写入 key 或凭据。

配置字段沿用 Pi 的 JSON 结构，但路径全部属于 MCode，不会读取 `~/.pi`：

| 文件 | 作用域 |
|---|---|
| `~/.mcode/agent/settings.json` | 全局设置 |
| `.mcode/settings.json` | 项目设置，覆盖全局设置 |
| `~/.mcode/agent/models.json` | OpenAI-compatible 模型和端点 |

`settings.json` 使用 Pi 原字段，模型与思考强度互相独立：

~~~json
{
  "defaultProvider": "openai-compatible",
  "defaultModel": "example-coder",
  "defaultThinkingLevel": "high"
}
~~~

`models.json` 只加载 `api: "openai-completions"` 的条目：

~~~json
{
  "providers": {
    "openai-compatible": {
      "baseUrl": "https://api.example.com/v1",
      "api": "openai-completions",
      "apiKey": "$OPENAI_API_KEY",
      "models": [
        {
          "id": "example-coder",
          "reasoning": true,
          "contextWindow": 128000,
          "thinkingLevelMap": {
            "high": "high",
            "max": null
          }
        }
      ]
    }
  }
}
~~~

完整样例见 `settings.example.json` 和 `models.example.json`。`apiKey` 支持字面值、
`$ENV_VAR` 和 `${ENV_VAR}`；为了保持配置无执行能力，不运行 Pi 的 `!command`。
项目也不读取 `auth.json`，不包含 OAuth。非 `openai-completions` 模型会被忽略。

支持 Pi 的 `compat.supportsReasoningEffort` 和 `compat.supportsUsageInStreaming`。两项默认均为
`true`；设为 `false` 时分别省略 `reasoning_effort` 和
`stream_options: { "include_usage": true }`。`compat` 可写在 provider 或 model 上，model 级配置
覆盖 provider 级配置。

优先级为：命令行参数、环境变量、项目 `settings.json`、全局 `settings.json`、内置默认值。

### MCP

MCP 配置写在全局或项目 `settings.json` 的 `mcpServers` 中：

~~~json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": {
        "EXAMPLE_TOKEN": "$EXAMPLE_TOKEN"
      },
      "enabled": true
    }
  }
}
~~~

只支持本地 stdio 服务器。`env` 值支持 `$ENV_VAR` 和 `${ENV_VAR}`；引用不存在的变量会在
启动时明确报错。项目配置按服务器名替换全局同名配置，`{"enabled": false}` 可以禁用继承的
服务器。MCP 工具以 `mcp__<server>__<tool>` 暴露给模型，`/status` 显示已连接服务器和工具数。

## 使用

~~~bash
# 交互模式
mcode
mcode "检查这个项目的测试失败"

# 非交互模式
mcode exec "修复 cargo test 的失败"
git diff | mcode exec "审查这个 diff"
mcode exec --json "说明 src/ 的结构"

# 图片输入；可重复 -i，也可用逗号分隔
mcode -i screenshot.png "检查这个界面"
mcode exec -i before.png,after.png "比较这两张图"

# 恢复会话
mcode resume
mcode resume <SESSION_ID>
mcode resume <SESSION_ID> "继续完成剩余工作"

# 删除会话；省略 --force 时会要求终端确认
mcode delete <SESSION_ID>
mcode delete <SESSION_ID> --force

# 自定义模型、思考和兼容端点
mcode --base-url https://example.com/v1 --model example-model --reasoning high
mcode --context-window 200000
~~~

交互模式支持：

- Enter 提交
- Shift+Enter 或 Alt+Enter 插入换行
- PageUp / PageDown 浏览历史
- Escape 取消当前任务
- Ctrl+C 取消当前任务；空闲时退出
- `/model [provider/model]`：查看或切换模型
- `/reasoning [off|minimal|low|medium|high|xhigh|max]`：独立查看或切换思考强度
- `/image <PATH>`：把图片加入下一条消息；`/image` 查看，`/image clear` 清空
- `/status`：查看模型、端点、上下文和 token 明细
- `/new`：开启新会话和新上下文
- `/delete`：二次确认后永久删除当前会话并退出
- `/clear`：只清理当前屏幕
- `/help`、`/quit`

底栏第二行采用紧凑状态格式，例如：

~~~text
in 24k out 2.1k | context 26k/128k (20.3%)              example-coder | reasoning high
~~~

服务返回 usage 时显示精确统计；未返回时使用本地估算，并以 `~` 标记。

图片会校验格式并编码成 Chat Completions `image_url` data URL。单张上限为 20 MiB，附件数据
会写进会话 JSONL，确保恢复会话时仍可发送原图；这也意味着含图片的会话文件会明显变大。

## OpenAI 兼容性

`base_url` 可以是 API 根路径，也可以直接是 `chat/completions` 完整地址。所有请求固定发送
当前采用的正式版 Codex `User-Agent: codex_cli_rs/0.145.0`，无需额外配置。客户端支持：

- SSE 增量文本
- `reasoning_content` / `reasoning` 增量的兼容读取
- 独立的 `reasoning_effort` 请求参数（可由 Pi `compat` 关闭）
- 分片 function call 参数
- 多轮工具调用
- 文本与图片 content parts
- 可配置的 `stream_options.include_usage` 与缺失 usage 时的本地估算

兼容服务必须实现流式 Chat Completions 和 OpenAI function calling 格式；需要处理图片时，
模型和端点也必须支持 Chat Completions 图像输入。

界面只展示兼容端点明确返回的 `reasoning_content` 或 `reasoning`，不会生成或推断模型的
隐藏思维链。普通 `mcode exec` 的 stdout 仍只输出最终回答；JSON 模式会保留 reasoning 事件。

## 安全边界

`read_file`、`write_file` 和 `edit_file` 拒绝访问工作目录外的路径，并检查符号链接解析结果。

`shell` 工具和启用的 MCP 服务器都以当前用户权限在工作目录运行。`shell` 支持超时和取消，
但没有容器或系统级沙箱；只应启用可信的 MCP 命令。不要在不可信仓库或持有高权限凭据的
环境中直接运行。

## 验证

~~~bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
~~~

测试包括一个本地模拟 OpenAI SSE 服务，覆盖“模型请求工具、执行文件写入、回传工具结果、
模型继续回答”的完整两轮流程，以及图片请求序列化、MCP 工具发现/调用和会话删除。
