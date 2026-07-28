# MCode

[![CI](https://github.com/imengying/MCode/actions/workflows/ci.yml/badge.svg)](https://github.com/imengying/MCode/actions/workflows/ci.yml)

一个专注、可审计的 Rust 终端 coding agent。交互和命令采用 Codex 风格，支持
OpenAI-compatible Chat Completions API 与 OpenAI Responses API。

## 当前范围

- `mcode`：启动交互式 TUI
- `mcode exec`：执行一次非交互任务
- `mcode resume`：恢复当前项目最近或指定的会话
- `mcode sessions`：列出当前项目保存的会话
- `mcode doctor`：离线检查配置、模型、端点、会话目录和 MCP 命令
- `mcode delete`：永久删除当前项目的指定会话
- OpenAI `/v1/chat/completions` 和 `/v1/responses` 流式 SSE 与 function calling
- Codex 风格托管 Web Search：disabled、cached、live 模式、搜索状态和 URL 引用
- `read_file`、`write_file`、`edit_file`、`shell` 四个内置工具
- JSONL v3 durable run journal、崩溃恢复、流式响应、滚动历史和多行编辑
- PNG、JPEG、GIF、WebP 图片输入
- 本地 stdio MCP 服务器与工具发现
- shell/MCP 执行前审批，支持单次或本会话授权
- Pi 风格自动上下文压缩：迭代摘要、最近历史保留和溢出恢复
- 模型与 reasoning 独立选择，思考内容单独渲染
- 底栏显示模型、reasoning、上下文占用和输入/输出 token

项目刻意不包含 OAuth、非 OpenAI Provider API、本地 LLM、远程 MCP transport、扩展系统、
技能市场和主题系统。

## 构建

~~~bash
cargo build --release
./target/release/mcode --help
~~~

使用当前正式版 Rust 构建；本版本以 Rust 1.97 验证。

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
export OPENAI_MAX_INPUT_TOKENS="128000"
~~~

API key 可以省略，以连接不需要认证的兼容服务。项目不会写入 key 或凭据。
`MCODE_HOME` 可覆盖默认的 `~/.mcode`，适合 CI 和隔离测试。

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
  "defaultThinkingLevel": "high",
  "webSearch": "disabled",
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000
  }
}
~~~

`models.json` 加载 `api: "openai-completions"` 或 `api: "openai-responses"` 的条目：

~~~json
{
  "providers": {
    "openai-compatible": {
      "baseUrl": "https://api.example.com/v1",
      "api": "openai-completions",
      "apiKey": "$OPENAI_API_KEY",
      "compat": {
        "supportsStrictTools": false
      },
      "models": [
        {
          "id": "example-coder",
          "reasoning": true,
          "contextWindow": 128000,
          "maxInputTokens": 128000,
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
项目也不读取 `auth.json`，不包含 OAuth。其他 API 类型会被忽略。

支持 Pi 的 `compat.supportsReasoningEffort`、`compat.supportsUsageInStreaming` 和
`compat.supportsStrictTools`。前两项默认均为 `true`；设为 `false` 时分别省略
`reasoning_effort` 和 `stream_options: { "include_usage": true }`。strict tools 只在官方
`api.openai.com` 的 Responses profile 默认启用，其他兼容端点默认关闭；关闭时不发送 `strict`。
四个内置工具的 schema 满足 strict 要求，动态 MCP schema 保持非 strict。`compat` 可写在
provider 或 model 上，model 级配置覆盖 provider 级配置。

`contextWindow` 是模型的输入与输出总上下文，`maxInputTokens` 是单次请求允许的最大输入；后者
省略时等于 `contextWindow`，且不能更大。MCode 使用 `maxInputTokens` 做裁剪和自动压缩判断，
不会把最大输出 token 误当成输入上限。GPT-5.6 样例使用费用更可控的 272,000 token 默认窗口；
更大的长上下文窗口应由用户明确覆盖，避免无意进入长上下文计费档。

优先级为：命令行参数、环境变量、项目 `settings.json`、全局 `settings.json`、内置默认值。

### Web Search

Web Search 使用 OpenAI Responses API 的托管 `web_search` 工具，不会通过 shell 抓取网页，也不
依赖第三方搜索站点。模型配置必须使用 `api: "openai-responses"`；该协议的模型默认采用
`cached`，Chat Completions 模型默认采用 `disabled`。显式为 Chat Completions 启用搜索会直接
报错，避免静默忽略配置。

~~~json
{
  "webSearch": "cached",
  "webSearchConfig": {
    "contextSize": "medium",
    "allowedDomains": ["openai.com", "rust-lang.org"],
    "location": {
      "country": "CN",
      "timezone": "Asia/Shanghai"
    }
  }
}
~~~

- `disabled`：不向模型提供搜索工具
- `cached`：`external_web_access: false`，只使用 OpenAI 维护的搜索缓存
- `live`：允许获取实时网页结果，等同命令行 `--search`

`contextSize` 可为 `low`、`medium` 或 `high`；`allowedDomains` 最多 100 个，只写域名，不带
`http://` 或 `https://`。搜索完成后，MCode 会把响应中的 URL annotations 去重并追加成
Markdown `Sources` 列表，确保终端中能看到并打开引用。协议与字段参考
[OpenAI Web Search 指南](https://developers.openai.com/api/docs/guides/tools-web-search)。

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
服务器。单个服务器启动或工具发现失败只禁用该服务器，并保留捕获的启动 stderr；其他服务器
和内置工具仍可用。MCP 调用有 120 秒超时。工具以 `mcp__<server>__<tool>` 暴露给模型，
`/status` 显示已连接服务器和工具数。

### 项目指令

启动 agent 时会读取当前工作目录中的 `AGENTS.md`，并把内容加入 system instructions。MCode
只读取这一层，不向父目录递归查找；文件必须是普通 UTF-8 文件，目录和符号链接都会忽略，大小
上限为 64 KiB。这样项目约束能够随仓库一起审查，同时不会通过同名链接越过当前工作目录。

## 使用

~~~bash
# 交互模式
mcode
mcode "检查这个项目的测试失败"

# 非交互模式
mcode exec "修复 cargo test 的失败"
git diff | mcode exec "审查这个 diff"
mcode exec --json "说明 src/ 的结构"

# 使用 Responses API 实时搜索；cached 模式可在 settings.json 中配置
mcode --search "检查这个依赖的最新正式版本"
mcode exec --search "汇总今天相关的安全公告并给出来源"

# 仅用于已隔离且完全可信的自动化环境
mcode exec --dangerously-bypass-approvals "运行并修复测试"

# 图片输入；可重复 -i，也可用逗号分隔
mcode -i screenshot.png "检查这个界面"
mcode exec -i before.png,after.png "比较这两张图"

# 恢复会话
mcode resume
mcode resume <SESSION_ID>
mcode resume <SESSION_ID> "继续完成剩余工作"

# 查看会话或做离线诊断；两者均支持 --json
mcode sessions
mcode doctor

# 删除会话；省略 --force 时会要求终端确认
mcode delete <SESSION_ID>
mcode delete <SESSION_ID> --force

# 自定义模型、思考和兼容端点
mcode --base-url https://example.com/v1 --model example-model --reasoning high
mcode --context-window 200000
mcode --max-input-tokens 180000
~~~

交互模式支持：

- Enter 提交
- Shift+Enter 或 Alt+Enter 插入换行
- PageUp / PageDown 浏览历史
- Escape 取消当前任务
- Ctrl+C 取消当前任务；空闲时退出
- shell/MCP 请求执行时，`y` 允许一次、`a` 在当前会话始终允许该工具、`n` 拒绝
- `/model [provider/model]`：查看或切换模型
- `/reasoning [off|minimal|low|medium|high|xhigh|max]`：独立查看或切换思考强度
- `/search [disabled|cached|live]`：查看或切换当前会话的网页搜索模式
- `/compact [INSTRUCTIONS]`：立即压缩上下文；可附加摘要关注点
- `/image <PATH>`：把图片加入下一条消息；`/image` 查看，`/image clear` 清空
- `/status`：查看模型、端点、上下文和 token 明细
- `/new`：开启新会话和新上下文
- `/delete`：二次确认后永久删除当前会话并退出
- `/clear`：只清理当前屏幕
- `/help`、`/quit`

底栏第二行采用紧凑状态格式，例如：

~~~text
in 24k out 2.1k | input 26k/128k (20.3%)                example-coder | reasoning high
~~~

服务返回 usage 时显示精确统计；未返回时使用本地估算，并以 `~` 标记。

自动压缩与 Pi 对齐：当 `contextTokens > maxInputTokens - reserveTokens` 时触发，默认预留
16,384 token，并反向选择约 20,000 token 的最近历史继续保留。更早的消息通过当前
OpenAI-compatible Chat Completions 模型生成结构化摘要；后续压缩会把上一份摘要与新增历史
迭代合并。正常情况只在完整用户轮次处切分；单个超大轮次超过保留预算时，会额外摘要该轮次
被切开的前缀。工具结果在摘要请求中最多保留 2,000 字符，原始会话记录不会因此截断。

压缩检查点（摘要、保留边界、压缩前 token 和摘要 usage）会写入 JSONL，`resume` 后重建为
“摘要 + 保留后缀”。`compaction.enabled` 只控制自动触发，关闭后仍可使用 `/compact`。
若自动摘要失败，MCode 才退回约 80% 输入预算的硬裁剪；若兼容端点明确返回上下文溢出，
MCode 会压缩并自动重试一次。

会话保存在 `$MCODE_HOME/sessions/<project-hash>/`（默认 `~/.mcode`）。JSONL schema v3 的头记录
保存 provider、model、API protocol、reasoning 和 Web Search 模式；run journal 依次记录 run、
generation、assistant completion、工具执行意图/结果和 run outcome。成功 generation 的 usage 与回答
在同一个 durable 边界后才更新界面，恢复会话时会重建累计 token。Responses API 的 typed output
items（包括加密 reasoning state）原样写入并在恢复后回放。

新文件沿用 Codex rollout 命名：
`rollout-YYYY-MM-DDTHH-MM-SS-<UUIDv7>.jsonl`，其中时间为 UTC。单个会话同一时间只允许一个写入
进程；`mcode sessions` 会显示累计 token 和 `[interrupted]`，JSON 输出包含 `total_usage` 与
`has_pending_run`。Unix 上会话目录为 `0700`、文件为 `0600`。

`mcode resume` 遇到未完成 run 时会先继续该 run。TUI 会自动恢复，并把同时提供的新 prompt 留在
编辑器中；非交互模式要求先不带新 prompt 完成恢复。崩溃发生在工具结果落盘前时，只有只读的
`read_file` 会安全重放；`write_file`、`edit_file`、`shell` 和 MCP 工具不会重复执行，而是写入
synthetic error 交给模型继续处理，避免未知副作用被执行两次。

图片会校验格式并编码成 data URL，再按协议发送为 Chat Completions `image_url` 或 Responses
`input_image`。单张上限为 20 MiB，附件数据会写进会话 JSONL，确保恢复会话时仍可发送原图；
这也意味着含图片的会话文件会明显变大。

## OpenAI 兼容性

`base_url` 可以是 API 根路径，也可以直接是 `chat/completions` 或 `responses` 完整地址；实际
路径由模型 profile 的 `api` 决定。请求发送 `User-Agent: mcode/<version>`。客户端支持：

- SSE 增量文本
- `reasoning_content` / `reasoning` 增量的兼容读取
- 独立的 `reasoning_effort` 请求参数（可由 Pi `compat` 关闭）
- 分片 function call 参数
- 多轮工具调用
- 文本与图片 content parts
- 可配置的 `stream_options.include_usage` 与缺失 usage 时的本地估算
- Responses `output_text`、reasoning summary、function call、usage 与失败事件
- Responses typed output item 原样回放，以及 `store: false` 时的加密 reasoning state
- Responses 托管 Web Search 的 search、open page、find in page 状态和 URL annotations
- 官方 Responses profile 的 strict function tools，以及兼容端点的显式能力开关
- 429、5xx 和连接失败的指数退避重试，遵循 `Retry-After` 并在错误中显示 request ID
- Chat Completions 与 Responses 流完整性检查；提前断流会完整重试一次，第二次仍截断才失败

`openai-completions` 服务必须实现流式 Chat Completions 和 function calling；
`openai-responses` 服务必须实现流式 Responses 和扁平 function tool schema。托管 Web Search
依赖服务端实际支持 OpenAI `web_search` 工具。

界面只展示兼容端点明确返回的 `reasoning_content` 或 `reasoning`，不会生成或推断模型的
隐藏思维链。普通 `mcode exec` 的 stdout 仍只输出最终回答；JSON 模式会保留 reasoning 事件。

## 安全边界

`read_file`、`write_file` 和 `edit_file` 拒绝访问工作目录外的路径，并检查符号链接解析结果。
写入和编辑使用同目录临时文件原子替换，并保留已有文件权限。模型、工具、MCP 和服务端错误中
的终端控制字符会在文本/TUI 输出边界过滤；`exec --json` 则由 JSON 转义安全保留原值。

配置的 MCP 服务器进程在启动前需要确认，后续 `shell` 和 MCP 工具调用也默认需要确认，因为
它们以当前用户权限运行且不受文件工具的项目目录限制。交互 TUI 和带终端的普通 `exec` 可以
允许一次、在本会话始终允许同名工具，或拒绝；管道输入、重定向环境和 `exec --json` 无法安全
询问时默认禁用 MCP 服务器并拒绝危险工具。自动化环境只有显式传入
`--dangerously-bypass-approvals` 才会跳过这些确认。

MCode 没有容器或系统级沙箱；审批只是在执行前取得同意，不会降低命令权限。只应启用可信的
MCP 服务器，并且不要在未隔离的不可信仓库或持有高权限凭据的环境中使用危险绕过参数。

网页搜索结果同样属于不可信外部输入，可能包含提示注入或误导内容。`cached` 模式减少直接访问
任意实时网页的暴露面，但不会消除风险；只有任务确实依赖最新信息时才使用 `live`。

`resume` 只接受当前 `-C` 项目对应会话目录中的 ID 或 JSONL 路径，并再次校验会话头记录的
工作目录，避免把 A 项目的历史恢复到 B 项目后执行工具。进程中断留下的未完成 JSONL 尾行会
在恢复时截断修复；文件中段或已完整换行的损坏记录仍会报错，避免静默丢失有效历史。项目内的
`AGENTS.md` 也属于仓库输入，启用前应像代码本身一样审查。

## 验证

~~~bash
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --all-targets --locked
cargo build --release --locked
~~~

默认门禁不会联网。只有确认会产生真实 API 用量时，才设置 `OPENAI_API_KEY` 并显式运行：

~~~bash
MCODE_REAL_API_MODEL=gpt-5.6-terra cargo test --test real_api -- --ignored --nocapture
~~~

GitHub Actions 使用最新正式版 Rust 在 Linux、macOS 和 Windows 上测试及构建，并执行
RustSec 审计。`v*` tag 会产出 Linux musl x86_64/arm64、macOS x86_64/arm64 和 Windows
x86_64 压缩包，同时用 `git archive | gzip -n` 生成确定性的 `mcode-<version>-source.tar.gz`，
并为全部产物发布 `SHA256SUMS`。源码包解压后可用对应正式版 Rust 执行
`cargo build --release --locked` 重建。真实 OpenAI API 冒烟测试只能手动触发，并要求显式配置
仓库 secret，普通 CI 不会产生 API 费用。

测试包括本地模拟 OpenAI SSE 服务，覆盖“模型请求工具、执行文件写入、回传工具结果、模型
继续回答”的完整两轮流程，以及危险工具拒绝无副作用、自动/迭代压缩、摘要失败硬裁剪、上下文
溢出恢复、截断流重试、durable 工具恢复策略、usage 累计、单 writer 锁、跨项目恢复拒绝、损坏
尾行修复、项目指令边界、图片请求序列化、MCP 工具发现/调用和会话删除。Responses 测试额外
覆盖 strict 开关、本地工具与托管搜索串联、搜索模式参数、事件、usage 和引用持久化。
