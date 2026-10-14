> 面向对象：自建部署者与运维。所有命令只读安全，标注 🔧 的为修复操作。
> 前置：检索 / 索引依赖一个可用的 **embed 端点**（默认 `http://127.0.0.1:18790`），
> 见 §2 渠道七。

---

## 1. 架构速览（排查前必读）

```
┌─ GUI ───────────────────────┐      ┌─ filesearch 服务 (deb, user systemd) ─┐
│ 文件检索视图 → :18700       │─────▶│ python -m bibit_filesearch.main serve  │
└─────────────────────────────┘      │   ↓ create_app (main.py)               │
                                     │   ① SettingsService (settings.py)       │
                                     │      └─ _migrate 老配置一次性迁移       │
                                     │   ② load_config (config.py)             │
                                     │   ③ CORSMiddleware (main.py)            │
                                     │   ④ scanner/watcher/indexer/retriever   │
                                     └─────────────────────────────────────────┘
```

**关键路径表**：

| 项 | 位置 | 代码依据 |
|---|---|---|
| 服务入口 | `create_app()` | main.py |
| **装配顺序**（SettingsService 先于 load_config）| main.py | 迁移必须发生在读配置之前 |
| 用户配置 | `~/.config/bibit-filesearch/config.yaml` | settings.py `_user_config_path` |
| 出厂默认（权威）| 代码 `DEFAULTS`（config.py ⊕ settings.py）| settings.py `_factory_defaults` |
| 出厂模板（参考）| `/opt/bibit-filesearch/config.yaml` | settings.py `_DEFAULT_CONFIG_PATHS` |
| 老配置迁移 | `_migrate()` | settings.py（删化石键 + 补 config_version）|
| 配置加载优先级 | `BIBIT_CONFIG` env > `~/.config` > 代码默认 | config.py `load_config` |
| DB | `~/.local/share/bibit-filesearch/filesearch.db` | main.py `_data_root` |
| 失败审计文件 | `~/.local/share/bibit-filesearch/scan_errors.log` | scanner.py `_record_error`（跨轮追加）|
| 默认索引目录 | `~/Documents`（动态填，存在才给）| settings.py `_default("index_dirs")` |
| systemd unit | `/usr/lib/systemd/user/bibit-filesearch.service` | deb 安装 |
| 排查脚本（装机自带）| `~/.local/bin/fs-search`（命令行检索）、`~/.local/bin/fs-db-report`（库体检）| 见 §2 渠道五 / 六 |

**配置存储语义**：用户文件只保存「与出厂默认不同的键」。首启不复制模板；出厂默认改进
自动触达老用户。想恢复某键的出厂默认 → 从文件里删掉该键（或改回默认值，保存时自动删）。

---

## 2. 排查渠道总览（先看这一节）

出问题时的顺序：**① GUI 能告诉你的 → ② 服务接口 → ③ 服务日志 → ④ 落盘审计文件 →
⑤ 库直查 → ⑥ 命令行工具 → ⑦ 接入端点**。越往后越接近真相，但前几步最快。

### 渠道一：GUI 界面

| 能看到什么 | 位置 | 局限 |
|---|---|---|
| 服务可达性（服务不可用横幅 / 状态灯）| 顶栏状态灯 + 红色横幅 | 它把「任一接口失败」判成服务不可用——**可能是假警报** |
| 索引进度、待向量化、失败数 | 状态栏三格 | 进度只在有轮询成功时更新 |
| **本次扫描的失败明细**（kind / path / reason）| 状态栏「失败文件」链接展开 | 展开按钮的条件是 `vectorizePhase.failed > 0`。**遍历阶段失败**（文件压根没入库）不计入这个数 → 明细里有内容但按钮不出现 |
| 检索结果与相关度名次 `#N` | 检索验证区 | — |
| 生效配置 | 设置页 | 只显示界面可改的 6 个键；`search.*` 等不显示 |

GUI 看不到的东西一律走渠道二 / 三。

### 渠道二：HTTP 接口（curl，最快、最准）

文件检索服务的端口是 **`127.0.0.1:18700`**（只绑回环，必须在本机执行）。

```bash
# 服务信息（uid / 用户名 / home / 端口）——判断服务在给哪个用户干活
curl -s http://127.0.0.1:18700/api/meta

# 健康：浅层（仅进程活着吗）
curl -s http://127.0.0.1:18700/health
# 健康：深层（db / embed 各自状态 + 已索引文件数）
curl -s http://127.0.0.1:18700/health/deep

# 生效配置（权威——用户文件是稀疏的，缺键属正常）
curl -s http://127.0.0.1:18700/api/settings

# 索引进度（含 errorDetails 失败明细 + truncationDetails 内容不完整明细）
curl -s http://127.0.0.1:18700/api/scan/progress | python3 -m json.tool

# 检索冒烟
curl -s -X POST http://127.0.0.1:18700/api/search \
  -H 'Content-Type: application/json' -d '{"query":"测试","pageSize":5}'

# 单个文件的详情（fileId 从上面的检索结果里取）
curl -s http://127.0.0.1:18700/api/files/<fileId>
```

**`/api/scan/progress` 是排查索引问题的核心**，它的 `data` 有两个阶段：

| 字段 | 含义 | 判据 |
|---|---|---|
| `scanPhase.walked` / `total` / `errors` | 遍历阶段（找文件、入库、排进向量化队列）| `errors > 0` 时看 `errorDetails` |
| `scanPhase.errorDetails[]` | 本次遍历的失败明细：`kind`（permission / embed / parse / other）+ `path` + `reason` | **有些失败不进库，只有这里有** |
| `vectorizePhase.{total,queued,vectorizing,completed,failed,unsupported}` | 向量化阶段（六个数，`total` = files 表全部行）| `failed` 是**库里**状态为 failed 的行数，与 `scanPhase.errors` 不是一回事 |
| `vectorizePhase.truncationDetails[]` | 「已索引但内容不完整」（超单文件块数上限）| 不是错误，文件仍可检索 |

> ⚠️ **`scanPhase.errors` 与 `vectorizePhase.failed` 常被混为一谈**：前者是遍历阶段
> 的失败（可能压根没进库），后者是库内状态。GUI 的失败明细面板只看后者。

**响应包装**：所有接口都是 `{code, message, data}` 三层。`code != 0` 时 `message` 是
给人看的说明；检索失败常见 `1004`（内部错误兜底，通常是 embed 侧问题——见 §2 渠道七）。
接口返回非 2xx 或 `code != 0` 时先看服务日志（渠道三）。

### 渠道三：服务日志（journalctl）

```bash
# user 服务必须带 XDG_RUNTIME_DIR，且要以目标用户身份
sudo -u bianbu XDG_RUNTIME_DIR=/run/user/1000 journalctl --user-unit bibit-filesearch -n 100 --no-pager
# 跟时间窗
... --since '17:50' --until '18:00'
# 👆 慎用 --since today：故障发生在昨天时它会返回 0 条（看着像"没这回事"）。
#    用相对窗口最稳：
... --since '3 days ago' | grep -E '遍历失败|向量化失败'
# 👆 也别用 -n 2000 这类小尾巴去捞：日志很啰嗦（每个请求都打 httpx 行），
#    几条错误会被埋在两万行外。
# 只看某类
... | grep -E '遍历失败|向量化失败|ERROR|Traceback'
```

抓日志时直接搜这几个关键字（都是 ERROR / WARNING 级别，会带完整异常）：

| 关键字 | 含义 |
|---|---|
| `遍历失败` | 遍历阶段某个文件进不了库（含非 UTF-8 文件名的 codec 报错）|
| `向量化失败` | embed / 解析阶段失败 |
| `加载配置:` | 启动时读了哪个配置文件——排查 CORS / 配置不生效的第一条 |
| `未找到 config.yaml` | 落到内置默认（属正常）|
| `向量化队列消费完毕，扫描完成` | 扫描正常收尾 |
| `分块规则升级（rev N）` / `解析能力升级（rev N）` | 存量索引重排（只出现一次，正常）|

### 渠道四：落盘审计文件

```
~/.local/share/bibit-filesearch/scan_errors.log
```

逐次**追加**的流水（不像 GUI 明细那样每轮清空），带时间戳 + kind + 路径 + 原因。
**价值在于跨轮、跨重启留存**：GUI 面板只显示本轮，服务重启后内存明细就没了，
而这个文件还在。上面「遍历失败」这类在 GUI 按钮上可能不出现的失败，这里能查到。

```bash
tail -30 /home/<user>/.local/share/bibit-filesearch/scan_errors.log
grep <关键字> /home/<user>/.local/share/bibit-filesearch/scan_errors.log
```

> 库里没有的文件**不会**留下记录（它进不了 `files` 表）。

### 渠道五：库直查（只读）

**首选现成脚本**（只读直连、WAL 安全、不需要 root）：

```bash
fs-db-report              # 总览：各状态条数 + 待处理 + 失败 + 内容不完整 + 目录分布
fs-db-report files        # 每个文件：状态 / 重试次数 / 块数
fs-db-report pending      # 只看待向量化
fs-db-report failed       # 只看失败
fs-db-report truncated    # 只看「已索引但内容不完整」
fs-db-report dirs         # 只看目录分布
fs-db-report --json       # 机器可读
```

**不要用 `cp filesearch.db` 取快照**——库是 WAL 模式，`cp` 会丢掉还没 checkpoint 的提交。
要么用上面这个脚本，要么用 Python 只读 URI 直连：

```bash
/opt/bibit-filesearch/.venv/bin/python -c "
import sqlite3
c = sqlite3.connect('file:/home/<user>/.local/share/bibit-filesearch/filesearch.db?mode=ro', uri=True)
print(c.execute('SELECT vectorize_status, COUNT(*) FROM files GROUP BY vectorize_status').fetchall())
print('chunks:', c.execute('SELECT COUNT(*) FROM chunks').fetchone()[0])
"
```

**「磁盘上有、库里没有」的差集** = 压根没进库的文件（最常见原因是文件名编码）：

```bash
find /home/<user>/test2 -type f | wc -l        # 磁盘上几个
# 库内：用上面脚本的数（或 fs-db-report 末行「共 N 个」），两侧一比即知少了哪些
```

### 渠道六：检索命令行工具

界面上搜得慢（数秒）时，先用它把「服务端耗时」和「界面耗时」分开：

```bash
fs-search <查询词>
fs-search <查询词> --topk 3
fs-search <查询词> --timing       # 逐段计时：/health/deep、/api/scan/progress、/api/search
fs-search <查询词> --raw          # 后端原始 JSON
```

判据：**命令行快、界面慢 → 问题是前端**（轮询间隔 + 渲染）；命令行也慢 → 后端 / embed。

### 渠道七：接入端点（embed 服务）

检索 / 索引必须依赖 `models.embed.url`（默认 `http://127.0.0.1:18790`）。先确认它在听：

```bash
curl -s http://127.0.0.1:18790/healthz
curl -s -X POST http://127.0.0.1:18790/v1/embed/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-0.6b-q4_0","input":["冒烟"]}'
# 期望返回 1024 维向量
```

自建端点时，改 `models.embed.{url,path,model,dim}` 四项即可（见 [`configuration.md`](configuration.md)）。

---

## 3. 排查流程（按现象分支）

### 现象 A：GUI 显示"服务不可用"

> **先排除假警报**：界面把「任一接口失败」判成服务不可用。若 `/health/deep` 与
> `/api/search` 正常、只有 `/api/scan/progress` 报错 → **不是服务挂了**，
> 见现象 I（文件名不是 UTF-8）。

```mermaid
flowchart TD
    A[GUI 服务不可用] --> B{18700 在听?}
    B -- 否 --> C[systemctl --user status bibit-filesearch]
    C --> C1{Active 状态}
    C1 -- "inactive (dead)" --> C2[从未启动 / disabled → is-enabled 查]
    C1 -- "activating auto-restart" --> C3[崩溃循环 → journalctl 看错误]
    C1 -- "active running" --> D
    B -- 是 --> D{CORS 头有吗?}
    D -- 无 --> E[配置缺 cors → 查启动日志读哪个配置]
    D -- 有 --> F{GUI 还能用吗}
    F -- 不能 --> G[查 :18790 embed 端点]
    F -- 能 --> H[OK]
```

**第一步命令**：

```bash
# 以目标用户身份查（user 服务必须带 XDG_RUNTIME_DIR）
sudo -u bianbu XDG_RUNTIME_DIR=/run/user/1000 systemctl --user status bibit-filesearch
ss -tlnp | grep 18700
```

### 现象 B：服务在跑但 GUI 连不上 / CORS 空

**核心判定：服务启动时读了哪个配置？**

```bash
sudo -u bianbu XDG_RUNTIME_DIR=/run/user/1000 journalctl --user -u bibit-filesearch -n 30 --no-pager
# 关注两行：
#   INFO bibit.config: 加载配置: /home/<user>/.config/bibit-filesearch/config.yaml   ← 对
#   WARNING bibit.config: 未找到 config.yaml（...），使用内置默认值                    ← 错！
```

| 日志看到 | 根因 | 修复 🔧 |
|---|---|---|
| `未找到 config.yaml（~/.config/...），使用内置默认值` | 用户配置缺失（**属正常**：首启本就不建文件）| 无需处理——生效值来自代码默认；在 GUI 保存一次即落盘 |
| `加载配置: ~/.config/...` 但 **CORS 仍空** | 用户文件里 `cors_origins` 是空列表 | 🔧 删掉该键或删整个文件重启（自动回落代码默认白名单）|
| 服务读 `/opt/.../config.yaml` 而不是 ~/.config | `BIBIT_CONFIG` env 被显式设置（调试遗留）| 检查 unit / 进程 env 里的 `BIBIT_CONFIG` |

**检查配置是否生效**（用户文件是稀疏的，缺键属正常）：

```bash
curl -s http://127.0.0.1:18700/api/settings      # 看生效值（权威）
grep -c cors_origins ~/.config/bibit-filesearch/config.yaml  # 0 属正常（回落默认）
```

> 判断是否正常请看 `/api/settings` 的生效值，不要数文件里写了几个键。

### 现象 C：索引失败 / 向量化报错

**journal 关键字**：

```bash
journalctl --user -u bibit-filesearch | grep -E "向量化失败|维度|ERROR"
```

| 报错 | 根因 | 修复 🔧 |
|---|---|---|
| `embed 服务返回维度与配置不符：models.embed.dim=512，实际 1024` | 配置里 `dim` 是旧值化石 | 升级会自愈（见**现象 G**）。手改：删掉该键重启 |
| `embed 请求失败（重试 N 次）: connect...` | 端点 `:18790` 没起 | 见 §2 渠道七 |
| `attempt to write a readonly database` | **DB 属主不是服务用户**（曾用 root 手动起服务）| 🔧 `chown -R <user>:<user> ~/.local/share/bibit-filesearch/` |

### 现象 D：重启后服务没了

**本质：服务必须由 systemd user 管理才会自启；手动 nohup / 终端起的进程重启即失。**

```bash
sudo -u bianbu XDG_RUNTIME_DIR=/run/user/1000 systemctl --user is-enabled bibit-filesearch
loginctl show-user bianbu | grep -E "Linger|State"
```

| 现象 | 修复 🔧 |
|---|---|
| disabled | `systemctl --user enable --now bibit-filesearch`（deb postinst 本应自动 enable，若没生效手动补）|
| 服务 enabled 但重启后 inactive | 查 unit 的 KillMode / 孤儿进程（旧进程占 18700 → 新进程 bind 失败循环）；`ps -ef | grep bibit_filesearch` 看有无残留 |
| user manager 没在跑 | 无登录会话时 user 服务不启动；桌面环境会自动拉起 |

### 现象 E：自动索引 / watcher 不生效

| 现象 | 根因 | 修复 🔧 |
|---|---|---|
| 新增文件不被索引 | watcher 停止标志未清（`start()` 补 `_stop.clear()`）| 升级到当前版本 |
| 全取消格式后仍索引 | 空列表被当成「不限」| 升级到当前版本 |
| 改浅深度后深层旧索引还在 | 无深度维度回收 | 升级到当前版本（reconcile 含深度维度）|

### 现象 F：文件明明有字，搜不到（状态 unsupported）

**判据**：文件在库里存在（`/health/deep` 有登记）但检索永远不返回它。

```bash
# 把 FILE 换成实际文件名，查它的状态与块数
FILE='AI.docx'
sudo -u bianbu python3 -c "
import sqlite3, sys
c = sqlite3.connect('file:/home/<user>/.local/share/bibit-filesearch/filesearch.db?mode=ro', uri=True)
fid = c.execute('SELECT file_id FROM files WHERE name=?', (sys.argv[1],)).fetchone()
row = c.execute('SELECT name,file_type,vectorize_status FROM files WHERE file_id=?', (fid[0],)).fetchone()
n = c.execute('SELECT count(*) FROM chunks WHERE file_id=?', (fid[0],)).fetchone()[0]
print(row, 'chunks =', n)" "$FILE"
```

| 状态 / 块数 | 根因 | 修复 🔧 |
|---|---|---|
| `unsupported` + 0 块，类型是 docx/pdf/txt 等**在索引格式内** | 解析器读不到文本。已知一种：**正文全在文本框里**的 docx（WPS / 演示软件导出的版式文档）——`python-docx` 只看主体直属段落，文本框（`w:txbxContent`）里的段落读不到，全篇被判「无文本」| 升级到当前版本（解析文本框），自愈会把它重排一次；**手动救急**见下 |
| `unsupported` + 0 块，类型是 `other`（emf / shaders 等）| 类型不在索引格式列表里 | 设计如此。要索引就先在界面勾上对应格式 |
| `completed` 但搜不到 | 语义门槛挡不住噪声时会被无关结果淹没；或查询词与内容差异过大 | 语义门槛当前标定失效（见 [`configuration.md`](configuration.md) 的 `search.min_similarity`）|

**手动救急**（不想等自愈 / 自愈已跑过）：清掉该文件的 `content_hash` 再重启，即可强制重解析
（**只改状态没用**——hash 未变会被 `indexer` 直接跳过）：

```bash
sudo -u bianbu python3 -c "
import sqlite3; c=sqlite3.connect('/home/<user>/.local/share/bibit-filesearch/filesearch.db', timeout=20)
c.execute(\"UPDATE files SET content_hash='', vectorize_status='queued' WHERE name='<文件名>'\")
c.commit()"
sudo -u bianbu XDG_RUNTIME_DIR=/run/user/1000 systemctl --user restart bibit-filesearch
```

自愈的日志判据：启动日志出现 `解析能力升级（rev N）→ M 个误判为「无文本」的文件重新排队`。
**只出现一次**（跑过就记进 `scan_state.parser_revision`），后续启动不再出现属正常。

### 现象 G：升级后「所有」检索都失败，但服务看着一切正常

**最容易误判的一条**：健康检查全绿，扫描计数在涨，就是搜不出东西。

```bash
curl -s http://127.0.0.1:18700/health/deep         # embed=ok、db=ok、计数正常
curl -s -X POST :18700/api/search -H 'Content-Type: application/json' -d '{"query":"GPU"}'
# → {"code":1004,"message":"..."}  每次检索都这样
```

**根因**：`models.embed.dim` 化石。用户配置里写着旧维度 `512`，而服务实际是 1024 维
（Qwen3-Embedding-0.6B）。配置里的 512 存活下来 → `EmbedClient` 每次请求都撞维度防线
抛错 → 检索统一兜底成 `1004`。

`/health/deep` 为什么查不出来：它只探 embed 服务的健康端点，**不经过维度校验**——所以这个
故障在外观上是完全隐形的。

| 检查 | 结果 |
|---|---|
| 用户配置里的 `models.embed.dim` | `512` |
| embed 服务实际维度 | `1024` |
| `/health`、`/health/deep` | 全 `ok` |
| 每次 `/api/search` | `code=1004` |
| journal | 有 `embed 服务返回维度与配置不符` 行 |

**修复**：维度**以服务探测值为准**（`EmbedClient.probe_dim()`），配置值降级为提示；
存量索引维度不符时清空向量并重排队重扫。日志判据：
`存量索引维度 512 ≠ 当前 embed 维度 1024 → 已清空向量并排队重索引`。

**两个坑**（改这块代码时必须记住）：

- **不能只把旧 `dim` 登记进 `_CHANGED_DEFAULTS` 就完事**。`dim` 与别的键不同：它约束存量
  数据的物理形态，删配置键不等于索引跟着变。只登记化石会让用户从「静默搜不到」变成
  「服务起不来」——更严重。**配置修正 + 索引处理必须成套**。
- **清空向量时必须同时清 `content_hash`**。否则重扫时 `Indexer.index_file` 见到 hash 未变
  直接 `skipped`，而向量刚被清空 → 该文件永远拿不回向量，worker 还会把它改判成
  `unsupported`：界面上「索引完成」，实际搜不到。`Database.purge_vectors` 已同时清 hash，
  同一模式在**解析能力自愈**（现象 F）里也适用。

**「自动丢弃索引」为何可接受**：索引是**派生数据**，内容全部来自知识库目录里的文件，清掉
只损失重建时间。且只在维度**确实不符**时才清（一致时一个块都不动），清空前必打日志。

### 现象 H：文件名不是 UTF-8 → 文件进不了库

**判据**：`/api/scan/progress` 报 500（或返回非法 JSON），而 `/health/deep` 与
`/api/search` 都正常；后台索引其实没停。

```bash
# 三个接口分别测——它们给出矛盾的答案，这本身就是判据
curl -s -o /dev/null -w 'health/deep     %{http_code}\n' http://127.0.0.1:18700/health/deep
curl -s -o /dev/null -w 'scan/progress   %{http_code}\n' http://127.0.0.1:18700/api/scan/progress
curl -s -o /dev/null -w 'search          %{http_code}\n' -X POST http://127.0.0.1:18700/api/search \
  -H 'Content-Type: application/json' -d '{"query":"a"}'
```

| 检查 | 结果（故障时）|
|---|---|
| `/health/deep` | **200**，`{"status":"ok","db":"ok","embed":"ok"}` |
| `/api/search` | **200**，检索结果正常 |
| `/api/scan/progress` | **500**（唯一坏的那个）|
| journal | `UnicodeEncodeError: 'utf-8' codec can't encode ... surrogates not allowed`，位置在响应序列化 |

**根因**：知识库里有**文件名不是 UTF-8** 的文件（典型来源：从 Windows / GBK 环境拷来的
目录，名字的原始字节如 `\xc9\xee\xb6\xc8...`，用 `ls` 看是乱码）。

```
os.walk 把这种名字经 surrogateescape 变成含 \udcc9 的 str
  → 它能碰文件系统，但 .encode("utf-8") 抛错
  → ① 入库时 sqlite3 参数绑定失败 → 该文件永远进不了库
      这条失败带着含 surrogate 的 path 进内存明细
  → ② /api/scan/progress 返回时 JSON 序列化的 .encode("utf-8") 抛错
      → 整个端点 500
  → ③ 前端同一轮轮询里 healthDeep 成功但 scanProgress 抛错，同一个 try/catch
     把 serviceUp 置 false → 界面顶栏「服务不可用」，且检索的门直接静默返回
  → ④ 但索引线程不受影响 → 已索引文件数继续涨
```

**为什么界面看不到那条失败明细**：失败明细面板的展开按钮条件是 `vectorizePhase.failed > 0`，
而这类失败**压根没进库**（卡在 SQLite 前），所以 `failed = 0` → 按钮不出现。

**★ 定位命令**（哪一个文件在捣乱）：

```bash
# ① 最快：服务日志里直接有。用相对时间窗——`--since today` 只在故障当天有效，
#    小 -n 也捞不到（日志每请求一行，错误埋在两万行外）
sudo -u bianbu XDG_RUNTIME_DIR=/run/user/1000 journalctl --user-unit bibit-filesearch \
  --since '3 days ago' --no-pager | grep 遍历失败

# ② 或从接口拿（reason 会说人话）
curl -s http://127.0.0.1:18700/api/scan/progress | python3 -m json.tool | grep -A2 -i gb18030

# ③ 或直接扫目录，把非 UTF-8 文件名揪出来
/opt/bibit-filesearch/.venv/bin/python -c "
import os, sys
for base in ['/home/<user>/test2']:
    for root, _d, files in os.walk(base):
        for n in files:
            try: n.encode('utf-8')
            except UnicodeEncodeError:
                print('非 UTF-8:', os.path.join(root, n).encode('utf-8','replace').decode('utf-8'),
                      '| 原始字节:', os.fsencode(n))
"
```

**修复 🔧**：**把文件改名**（这类文件无论如何都进不了库——见下）。

```bash
# reason 里能直接读到「按 gb18030 读为《深度拷问清单.md》」，用 printf 还原原始字节来 mv；
# 或在文件管理器里右键重命名
cd <目录>
mv "$(printf '\xc9\xee\xb6\xc8\xbf\xbd\xce\xca\xc7\xe5\xb5\xa5.md')" 深度拷问清单.md
```

改完点一次「保存并索引」（或等 inotify 自动收）即入库。

**如果暂时不能改名**（想先把界面救回来）——临时把该目录移出知识库，或重启服务清掉内存
里的脏明细：

```bash
sudo -u bianbu XDG_RUNTIME_DIR=/run/user/1000 systemctl --user restart bibit-filesearch
```

⚠️ 但**下次全盘扫描（改设置 / 点重新扫描 / 重启后首次扫描）会再次踩中** → 又 500。
改名才是根治。

**为什么不能"支持 GBK 文件名"**（这条常被问）：Linux 进程里文件名必须被解码成一个
Python `str`；非 UTF-8 时 Python 走 surrogateescape，得到的 `str` **能还原成磁盘字节**
但**编不出合法 UTF-8**。而 SQLite 的 TEXT 与 JSON 都要求合法 UTF-8 ——
**不存在一个 `str` 能同时满足这两条**。强制假设某个编码（一律按 GBK 解）会让
latin1 / shift_jis / big5 的名字被解成别的名字、混存时必有一方错；存转义串则会让
`os.path.exists` 判「文件不在」→ 索引回收把这些文件当已删除清掉。所以立场是：
**不让它打挂别的功能，并告诉用户去改哪个名字**，而不是去"支持"它。
（这类字节在 Linux 上是历史遗留脏数据，`tar` / `grep` / `rsync` 一样处理不了。）

### 现象 I：图片索引成了 failed

**这是有意引入的可见状态**，不是回归：OCR 失败会落 `failed` 并带原因。
不存在「静默换个引擎接着读」——那会让用户拿到**另一个引擎的结果**却不知道降级发生了。

**判据**：`/api/scan/progress` 的 `errorDetails` 里该文件的 `reason` 是
`ocrFailed:{"kind":"ocrFailed",…}` 结构化标记。
最快的一条：

```bash
curl -s localhost:18700/api/meta | python3 -m json.tool | grep -A9 '"ocr"'
# 期望：engine=ppocr、ready=true、device=npu（或 cpu）、error 为空
# ready=false 或 error 非空 → 见下表
```

| `/api/meta` 的 `ocr` 段 | 根因 | 修复 🔧 |
|---|---|---|
| `ready=false`，`error` 提到 `RapidOCR` / `import` | venv 里 rapidocr 或 onnxruntime 没装成 | `sudo dpkg-reconfigure bibit-filesearch` 或重装 deb；装依赖要 SpaceMiT 私有 PyPI（`PIP_INDEX` 已在 postinst 里配好）|
| `ready=true` 但 `device=cpu`、`npu_ok=false` | NPU 辅助进程或 `libspacemit_ep.so` 缺失 → 自动退到 CPU（**功能正常，只是慢**）| 确认 `/opt/bibit-filesearch/ocr-ep/ep_worker` 存在、`spacemit-onnxruntime` 已装；装好重启服务即可回到 npu |
| `low_conf_count` 持续增长，文本能搜到但明显不对 | 识别质量差（小字 / 低分辨率 / 手写）| 调大 `ocr.max_side_len`（默认 **1400**）让引擎看到更多像素；**服务不会自己重读**——低置信度只记账不改写结果，要重试就清该文件 `content_hash` 重启（见现象 F 的「手动救急」）。⚠️ 调大这个值会**显著**抬高内存，且改完要配 `PARSER_REVISION` `+1` 才会对存量生效 |
| 单张图 OCR 让服务 RSS 涨上去不回落 | ONNX Runtime 的 arena 高水位随**输入面积**上涨且不归还（一张大图能把 det worker 抬到 GB 级，且静置不降、连跑多次不涨——是**高水位不是泄漏**）| 正常现象，由**四条**路径兜住：① `ocr.idle_release_secs` 空闲释放（默认 10s；**运行期唯一**触发点）② `ocr.worker_max_runs` 按次数退休 ③ `ocr.worker_max_rss_mb` 按内存退休（默认 0 = 关；设 1500 只挑大图的 det，因为 rec 自然水位就在 ~1050MB）④ **服务停止**显式释放（lifespan shutdown + 信号兜底，幂等）。**调小 `ocr.max_side_len` 才是唯一降峰值的旋钮**，另几条只降「高水位的持有时间」|
| **watcher 增量（一次只来一张）后内存一直不还** | 增量路径没有批次语义，每次识别的高水位会留到下次全盘扫描 | 由 `ocr.idle_release_secs` 兜住（默认 **10s** 空闲即释放）。仍不还就查 `/api/meta` 的 `ocr.worker_count`（应随空闲归 0）——若长批量扫描期间不还，那是「池一直不空、TTL 也不触发」，需靠 `ocr.worker_max_rss_mb` 兜 |

**为什么不再有 tesseract 兜底**（这条常被问）：双引擎时「PP-OCR 崩了」会静默换成
tesseract 接着读——用户拿到的是**另一个引擎的结果**，既不知道降级发生了，也无从判断
文本为什么和以前不一样。单引擎把这种「悄悄换人」换成一句可查的失败原因。代价说清楚：
**引擎装不成时图片就真的索引不了**（文本类文件不受影响），不存在降级可用。

**「失败」为什么可能比以前多**：判据从「OCR 专用」推广成了「解析器报的任何 error」。
所以损坏文档会显示成「失败」，而不是以前的「未纳入索引 · 无文本」——这是有意的状态迁移。

---

## 4. 状态自愈法（最常用修复）

**配置错乱 → 删文件重启，自动回落完整出厂默认**：

```bash
sudo -u bianbu XDG_RUNTIME_DIR=/run/user/1000 systemctl --user stop bibit-filesearch
rm ~/.config/bibit-filesearch/config.yaml          # 删配置（含自定义设置，先备份）
sudo -u bianbu XDG_RUNTIME_DIR=/run/user/1000 systemctl --user start bibit-filesearch
# 启动后不重建文件（首启不复制模板），生效值全部来自代码默认；
# 首次在 GUI 保存设置时才写回文件。
# 日志应出现：INFO bibit.config: 加载配置: ~/.config/bibit-filesearch/config.yaml
```

**原理**：权威默认值在代码里（`config.DEFAULTS` ⊕ `settings.DEFAULTS`，见 `settings.py` 的
`_factory_defaults`）。`load_config` 按 `BIBIT_CONFIG` env > `~/.config` > 代码默认
分层合并，用户文件缺失的键一律回落到代码默认——不依赖「首启从 /opt 模板复制一份」。

**为什么不复制模板**：把整份出厂模板复制成用户配置，会把当时的默认值**冻结成用户选择**，
此后出厂默认改进永远触达不了该用户。

**只想恢复某一个键**：从文件里删掉该键（或改回出厂默认值，保存时自动删）。
**想看当前所有生效值**：`curl -s http://127.0.0.1:18700/api/settings`。

---

## 5. 验证命令速查

> 排查渠道的分层说明见 §2——本节是「一条一条复制粘贴」的精简版。

```bash
# 服务在给哪个用户干活（uid / home / 端口）
curl -s http://127.0.0.1:18700/api/meta
# 健康（浅 / 深）
curl -s http://127.0.0.1:18700/health                    # {"code":0,...,"status":"ok"}
curl -s http://127.0.0.1:18700/health/deep               # db / embed 各自状态 + 已索引数
# CORS（GUI 能否连的关键）
curl -s -D - -o /dev/null -H "Origin: tauri://localhost" http://127.0.0.1:18700/health | grep -i access-control
# 生效配置
curl -s http://127.0.0.1:18700/api/settings
# 索引进度（★ 含失败明细 errorDetails / 内容不完整 truncationDetails）
curl -s http://127.0.0.1:18700/api/scan/progress | python3 -m json.tool
# 检索冒烟
curl -s -X POST http://127.0.0.1:18700/api/search -H 'Content-Type: application/json' -d '{"query":"测试"}'
# 单个文件详情（fileId 从检索结果取）
curl -s http://127.0.0.1:18700/api/files/<fileId>
# DB 直查（只读，WAL 安全；不要 cp 快照）
fs-db-report        # 总览（failures / pending / truncated / dirs）
/opt/bibit-filesearch/.venv/bin/python -c "import sqlite3;c=sqlite3.connect('file:/home/<user>/.local/share/bibit-filesearch/filesearch.db?mode=ro',uri=True);print(c.execute('SELECT vectorize_status,COUNT(*) FROM files GROUP BY vectorize_status').fetchall())"
# 命令行检索（把界面耗时与服务端耗时分开）
fs-search 测试 --timing
# 失败审计（跨轮留存）
tail -30 /home/<user>/.local/share/bibit-filesearch/scan_errors.log
```

**判别线上代码包含哪些能力**（升级后确认生效）：

```bash
grep -c '\.config/bibit-filesearch/config.yaml' /opt/bibit-filesearch/src/bibit_filesearch/config.py
# ≥1 → 含用户层配置查找
grep -n "SettingsService" /opt/bibit-filesearch/src/bibit_filesearch/main.py | head -1
# 出现在 load_config 之前 → 装配顺序正确（迁移先于读配置）
ls /opt/bibit-filesearch/src/bibit_filesearch/pathutil.py
# 存在 → 含非 UTF-8 文件名处理
grep -c 'min_embed_chars' /opt/bibit-filesearch/src/bibit_filesearch/indexer.py
# ≥1 → 含极短块不入向量
```

---

## 6. 远程排查备忘（SSH 到另一台机器）

```bash
# 以目标用户身份 SSH 进去（用自己的登录方式）
ssh -o StrictHostKeyChecking=accept-new <user>@<IP> '<只读命令>'

# 注意：18700 绑 127.0.0.1，外部 curl 不可达是正常的，必须 SSH 进去 localhost 查
# 排查套路：dpkg 版本 → systemctl --user status → journalctl 加载配置行 → CORS → DB → 依赖 :18790 端点
```
