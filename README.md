# MyGO!!!!! QQ 机器人（三个角色 Bot）

一组以《BanG Dream! It's MyGO!!!!!》角色为原型的 QQ 群聊 AI 机器人。基于 **NoneBot2 + OneBot v11（NapCat）**，对话模型走 **OpenAI 兼容网关**（默认 DeepSeek）。

> 本仓库是从个人运行版本导出的**脱敏模板**：所有真实群号 / QQ 号已替换为占位符，运行数据、日志、表情包素材与 NapCat/QQ 客户端均未包含。

## 三个 Bot

| 目录 | 角色 | 人设 | 主要能力 |
|---|---|---|---|
| `mutsumi/` | 若叶睦 / 墨提斯（双人格） | 话少、直白、短句，被叫必出声 | 群聊触发、梗库、上下文/群摘要、黄瓜农场小游戏、每日新闻、语音、识图、联网搜索、好感度、私聊 |
| `anon/` | 千早爱音 | 小太阳、嘴硬心软、社区二设 | 闲聊/任务分流（planner → 最终出声 voice）、联网快搜、滚动群摘要、持久化群友档案、识图、自主动作（表情）、经验库、熔断保护 |
| `viola/` | 薇欧拉 | 独立人设（成人向角色扮演） | 群聊、主动搭话概率、群摘要、任务代理、识图、昵称/带话等子人格指令 |

## 架构

```
QQ ←→ NapCat（注入 QQ NT，OneBot v11 WS）←→ NoneBot2（bot.py → src/plugins/*.py）←→ LLM（OpenAI 兼容）
                                                        ├─ chat.py / anonbot.py：触发、人设、发送
                                                        ├─ chat_logic.py / anon_context.py：纯逻辑（可离线单测）
                                                        ├─ agent_core.py：planner / 任务代理 / voice pass
                                                        └─ *_search.py / experience.py：联网检索、经验库
```

设计要点：

- **人设为 system 消息**，走模型前缀缓存，不重复塞进每条 user 提示词。
- **纯逻辑层与 I/O 分离**：上下文组装、截断、判定都能离线单测。
- **白名单群 + 触发判定**：只有被 @、被引用或被点名才回复，避免打扰群聊。
- **失败不失控**：LLM 超时/空回复/熔断都有固定兜底，进程不掉线。

## 快速开始

1. **Python 3.12**（推荐），创建虚拟环境并安装依赖：
   ```bash
   cd anon            # 或 mutsumi / viola
   python -m venv .venv
   .venv/Scripts/pip install -r requirements.txt   # Windows
   # source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
   ```
2. **准备 OneBot v11 实现**：安装 [NapCat](https://github.com/NapNeko/NapCatQQ)（或 Lagrange 等），登录一个 QQ 小号，开启 WebSocket 服务（默认 `ws://127.0.0.1:3001/3002/3003`，按 bot 区分）。
3. **配置**：
   ```bash
   cp .env.example .env
   ```
   填入 `DEEPSEEK_API_KEY`、`ALLOWED_GROUPS`（你的群号）、`BOT_QQ`（bot 自己的 QQ 号）、`ONEBOT_V11_WS_URLS` 等。
4. **启动**：
   ```bash
   python bot.py
   ```
   看到 `Bot <QQ> connected` 即为上线。

同一台机器跑多个 bot 时，给每个 bot 分配不同的 NapCat 实例与端口（示例：mutsumi 3001 / viola 3002 / anon 3003）。

## 测试

```bash
cd anon && python tests/run_all.py          # ~300 项离线单测（无需网络/QQ）
cd mutsumi && python -m unittest discover -s tests
```

`anon/tests/run_all.py` 会自动收集 `test_*.py`；测试不联网、不启动 NoneBot、不写运行数据（内部有 NoneBot 未初始化守卫）。

## 目录说明（以 anon 为例）

```
anon/
├── bot.py                 NoneBot2 入口（加载 .env、插件）
├── requirements.txt
├── .env.example
├── roles/Anon.md          角色卡（system 消息，人设与边界）
├── src/plugins/
│   ├── anonbot.py         I/O 与调度（触发、发送、记忆落盘、熔断）
│   ├── anon_context.py    纯逻辑：上下文/提示词/清洗/摘要
│   ├── agent_core.py      planner、任务代理、最终出声 voice pass
│   ├── anon_people.py     持久化群友档案（认人/身份锚点）
│   ├── anon_search_policy.py  联网检索判据与黑话
│   ├── anon_experience.py     轻量经验库（L2 群情报 / L3 任务经验）
│   └── ...
└── tests/                 离线单测
```

## 隐私与安全

- **未包含**任何 `.env`、API Key、聊天记录、群友档案、日志、截图、表情包素材、NapCat/QQ 二进制。
- 源码中的真实群号与 QQ 号已统一替换为占位符（`1000000xx` / `2000000xx`），请按需改回你自己的配置（更推荐放在 `.env`）。
- 请不要把你自己的运行数据（`*_memory.json`、`chat_memory.json`、日志等）提交到公开仓库；它们已在 `.gitignore` 中。

## 免责声明

- 非官方同人项目，与 Bushiroad / BanG Dream! 官方无关；角色与设定版权归原作者所有。
- 自动化 QQ 操作可能违反平台条款并带来账号风险，请自行评估并遵守相关规定。
- 本项目仅供学习与个人娱乐使用。

## License

本仓库暂未附带 LICENSE 文件——许可方式由作者决定。若希望他人可以自由使用/修改，建议补充 MIT 或 Apache-2.0。
