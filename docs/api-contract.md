# bibit-filesearch API 契约文档

> 本文档是 bibit-filesearch 服务对外接口的**权威基准**。
> 所有上游消费方必须严格按本文档对接。
> 约定：所有响应统一 `{code, message, data}` 三层包装；code=0 成功，非 0 失败。

---

## 0. 基础信息

| 项 | 值 |
|---|---|
| 服务名 | Bibit Filesearch |
| 默认地址 | `http://127.0.0.1:18700`（per-user 端口：`uid<1000 → 18700`，`uid≥1000 → 18700+(uid-1000)`；注意 uid=1000 也落在 18700） |
| 数据目录 | `~/.local/share/bibit-filesearch`（per-user） |
| 配置来源 | `~/.config/bibit-filesearch/config.yaml` |
| 向量化服务 | `http://127.0.0.1:18790` → `POST /v1/embed/embeddings`（须显式传 `model=qwen3-embedding-0.6b-q4_0`；Qwen3-Embedding-0.6B Q4_0，1024 维）。地址与模型均可在 `models.embed.*` 配置 |

---

## 1. 通用响应结构（所有接口统一）

```json
{
  "code": 0,            // 0=成功；非 0=失败
  "message": "success", // 中文或英文提示
  "data": { ... }       // 实际数据，任意结构
}
```

### 错误码（统一）

| code | 含义 | 典型场景 |
|---|---|---|
| 0 | 成功 | — |
| 1001 | 参数错误 | query 为空、分页非法 |
| 1002 | 过滤条件错误 | 未知 fileType、日期区间倒挂 |
| 1004 | 内部错误 / 资源不存在 | 检索异常、服务状态异常；`GET /api/files/{fileId}` 文件不存在也返回 1004（data=null） |

**上游约定**：
- 判断成功只看 `code === 0`，不要用 `message` 或 `data` 是否存在判断。
- 失败时 `data` 可能为 `null`，上游必须先判 code 再访问 data。

---

## 2. 接口清单

| Method | Path | 作用 |
|---|---|---|
| GET | `/health` | 健康检查（轻量） |
| GET | `/health/deep` | 深度健康（db/embed/索引计数） |
| POST | `/api/scan` | 触发全量扫描（幂等） |
| GET | `/api/scan/progress` | 扫描进度 |
| POST | `/api/scan/pause` | 暂停扫描 |
| POST | `/api/scan/resume` | 继续扫描 |
| GET | `/api/settings` | 获取配置 |
| PUT | `/api/settings` | 修改配置（幂等） |
| POST | `/api/search` | 检索（核心接口） |
| GET | `/api/files/{fileId}` | 单文件详情（元数据+分块统计） |

---

## 3. GET /health

轻量健康：服务进程是否存活。

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "ok"
  }
}
```

---

## 3.1 GET /health/deep

深度健康：验证服务是否真正可用（数据库可读 + embed 服务在线 + 索引计数）。供排查/监控用。

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "ok",                // ok / degraded
    "db": "ok",                    // ok / error
    "embed": "ok",                 // ok / unreachable
    "embedUrl": "http://127.0.0.1:18790",  // embed 服务地址（前端降级提示用）
                                              "indexFiles": 30,              // 已向量化文件数
    "indexErrors": 0               // 向量化失败数
  }
}
```

**embed 恢复自动重建**：服务内置看门狗每 30s 探测一次 embed
（`scan.embed_recovery_interval` 可调）。检测到 **不可用 → 可用** 跳变、且库里
存在 `failed`/`queued` 文件时，自动重启扫描重跑这些文件（已 completed 的走
hash 跳过）。仅在跳变时触发：永久解析失败的文件不会被无限重扫。扫描进行中
或暂停时跳过，不打断用户操作。

---

## 4. POST /api/scan

触发全量扫描。**幂等**：已在扫描则忽略，不重复入队。

请求体：无需 body（或 `{}`）。

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "scanning",          // scanning / idle / done / error
    "message": "扫描已启动"         // 人类可读提示
  }
}
```

---

## 5. GET /api/scan/progress

扫描两阶段进度。

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "status": "done",              // idle / scanning / paused / done / error
    "scanPhase": {
      "walked": 168,               // 已遍历文件数（**遍历**阶段，快）
      "total": 168,                // 遍历总数
      "errors": 0,                 // 遍历错误数
      "errorDetails": []           // [{kind, path, reason}]，kind=permission/embed/parse/other
                                   // = 库里 failed 行（持久，带 fail_reason）
                                   //   ∪ 本轮内存流水里「库里没有行」的条目
                                   //   （例如文件名不是 UTF-8 —— 连行都建不出来）
    },
    "vectorizePhase": {
      "total": 168,                // files 表总行数（进度基准的**分母**）
      "queued": 0,                 // 待向量化（队列中）
      "vectorizing": 3,            // 正在向量化（在飞）
      "completed": 30,             // 已成功向量化
      "failed": 0,                 // 向量化失败
      "unsupported": 135,          // 无文本/纯图片 PDF（不产生向量，但已处理完）
      "truncated": 13,             // 内容不完整（被 scan.max_chunks_per_file 截断）；**不是失败**
      "truncationDetails": []      // [{path, indexed, total}]，与 truncated 同源同长
    }
  }
}
```

### 状态机（scan status 值域）

| status | 含义 | 上游展示 |
|---|---|---|
| `idle` | 空闲，未扫描 | 空闲 |
| `scanning` | 扫描中 | 扫描中（蓝） |
| `paused` | 已暂停（用户暂停） | 已暂停 |
| `done` | 扫描完成 | 已完成（绿） |
| `error` | 扫描整体失败 | 失败（红） |

> 注意：`scanPhase.errors`（遍历错误）与 `vectorizePhase.failed`（向量化失败）是**两个不同来源**的失败，上游不要混用。

### 两个阶段：谁是谁的进度

扫描是**两阶段**的，两个 `total` 含义不同，上游别混用：

| 字段 | 阶段 | 什么时候走完 |
|---|---|---|
| `scanPhase.walked` / `total` | **遍历**（os.walk 发现文件并入队）| 很快，与向量化不在一个量级 |
| `vectorizePhase.*` | **向量化**（算向量，真正耗时的）| 队列排空才结束 |

**不要把 `scanPhase.walked/total` 当「索引进度」显示**：遍历很快结束，之后 `status` 仍是
`scanning`（等向量化），界面会长时间停在 `N / N` 而进度条才四成，用户读成「扫完了」。
**索引进度只能用 `vectorizePhase` 算**：

```
已处理 = completed + failed + unsupported
待处理 = queued + vectorizing
进度   = 已处理 / total          （total 即本组各状态之和，闭合）
```

只有遍历**尚未走完**（`scanPhase.walked < scanPhase.total`）时，才适合显示遍历本身。

`total` / `vectorizing` / `unsupported` 三者缺一不可：① `vectorizing` 必须落一格，否则
三数之和 ≠ 文件数；② `unsupported`（无文本 / 纯图 PDF）不可见时，有这类文件的库进度
永远到不了 100%。**总数用 `total`**，不要自行按 `completed + queued + failed` 相加。

### `kind` 取值

| kind | 含义 |
|---|---|
| `permission` | 权限/IO 问题，服务读不到文件 |
| `embed` | **向量化服务侧**失败：过载、超时、拒收、维度不符、重试耗尽 |
| `parse` | 解析失败（文件格式/解析器问题）|
| `other` | 遍历阶段的其它异常 |

`embed` 与 `parse` 必须分开：服务侧失败（过载 / 超时 / 拒收 / 维度不符）与文件解析失败
混用一个标签，会把排查方向带偏。

### `truncated` 与 `failed` 的区别

用途：**仅调试 / 运维取证，界面不展示**——内容不完整不必给最终用户看，只留接口供排查时定位。

| | `failed` | `truncated` |
|---|---|---|
| 语义 | 索引**没成功**，该文件检索不到 | 索引**成功了**，但只收录了前一部分内容 |
| 已入库的块 | 无 | 有（`indexed` 块全部可检索）|
| 生命周期 | 本轮内存流水（下一轮清空）∪ 库里 failed 行（持久） | **持久**属性（按 DB 求值），跨轮次可见 |
| 计数与明细 | `errorDetails` 现在**以库里 failed 行为主干**（与 `failed` 同源），另补内存里没进库的失败 → 条数仍可能不等，但**界面取值一律用 `errorDetails.length`** | `truncated` 与 `truncationDetails` **同源同长** |

`reason` 对**可归类**的失败给**结构化标记**（`filenameUnencodable:` / `parseFailed:` /
`ocrFailed:` + JSON），界面据此按当前语言出词；只有无法翻译的技术细节才原样透传。
判据请匹配结构化前缀（如 `ocrFailed:`），不要匹配旧式 `ocr: ` 前缀。

`errorDetails` 与 `failed` **同源**：明细按 DB 求值（`files.fail_reason` 落库），与
`truncationDetails` 同一范式；内存流水只保留「库里没有行」的条目（例如文件名不是
UTF-8，连行都建不出来）。因此 **`errorDetails` 不是「每轮必清空」**——库里仍失败着的
文件会一直列出。

一条命令列出所有内容不完整的文件（含覆盖比例）：

```bash
curl -s http://127.0.0.1:18700/api/scan/progress | python3 -c \
 "import json,sys; [print(f\"{t['indexed']:>5}/{t['total']:<6} {t['path']}\")
  for t in json.load(sys.stdin)['data']['vectorizePhase']['truncationDetails']]"
```

截断事实在 `vectorizePhase.truncated` / `truncationDetails` 里（**持久**属性，按 DB 求值），
**不在 `errorDetails` 里**——它与「失败」计数不同源，混在一起会出现「失败 1，点开列 12 条」
的矛盾。判据请用 `truncationDetails`，不要过滤 `kind=oversized`（该 kind 不出现）。

---

## 6. POST /api/scan/pause / resume

暂停/继续扫描。请求体无需 body。

```json
// pause
{"code":0, "message":"success", "data": {"status":"paused", "message":"扫描已暂停"}}
// resume
{"code":0, "message":"success", "data": {"status":"scanning", "message":"扫描已继续"}}
```

---

## 7. GET /api/settings

返回当前生效配置。

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "index_dirs": ["/home/<user>/Documents"],                            // 知识库路径列表
    "auto_index": true,                                                  // 自动索引开关
    "indexable_types": ["txt", "md"],                                    // 可索引格式
    "exclude_patterns": [".git", "node_modules"],                        // 排除规则 glob
    "max_depth": 5,                                                      // 索引深度
    "top_k": 5                                                           // 单次检索最多返回条数（1~100）
  }
}
```

---

## 8. PUT /api/settings

修改配置并热生效（写回 config.yaml → 触发扫描/回收）。**幂等**：payload 与当前一致时跳过重应用。

请求体（partial，只传要改的字段；`indexable_types` 传完整列表）：

```json
{
  "max_depth": 5,
  "top_k": 5,
  "auto_index": true
}
```

响应：`data` 为**更新后**的完整配置。幂等时 `data.changed=false`，其余 `true`。

```json
// 真正更新（changed=true）
{"code":0, "message":"success", "data": {"max_depth":5, "top_k":5, "...":"...", "changed": true}}
// 跳过更新（changed=false，payload 与当前一致）
{"code":0, "message":"success", "data": {"...":"...", "changed": false}}
```

> 上游约定：
> - 判断是否真的更新，看 `data.changed`，**不要**靠比较响应值。
> - 幂等跳过返回 `changed:false`，前端应提示"配置未变化，已跳过"而非"保存成功"。

---

## 9. POST /api/search（核心接口）

自然语言/关键词检索。NL 解析由后端统一走规则确定性路径（3ms，不调 LLM）。

### 请求

```json
{
  "query": "帮我找一下 GTK 相关的文档",
  "page": 1,
  "pageSize": 10,
  "filters": {                        // 可选；缺省则由后端 NL 解析自动生成
    "fileTypes": ["md", "docx"],      // 文件类型白名单
    "dateRange": {
      "dateField": "modified",        // modified/created/accessed
      "start": "2026-08-01",
      "end": "2026-08-31"
    },
    "directories": ["/home/<user>/Documents"]
  }
}
```

### 响应

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "total": 13,            // 总命中数
    "page": 1,              // 当前页
    "pageSize": 10,
    "totalPages": 2,
    "results": [
      {
        "fileId": "f_xxx",
        "fileName": "gtk_realize_map_visable.md",
        "filePath": "/home/<user>/Documents/GTK/gtk_realize_map_visable.md",
        "fileType": "md",
        "fileSize": 3462,
        "createdAt": "2026-08-12T10:34:00+0800",
        "modifiedAt": "2026-08-12T10:34:00+0800",
        "author": "",
        "score": 0.5863,             // 未归一化的绝对融合分（0~1）。勿直接展示给用户
        "confidence": 0.83           // 0~1 小数，仅表示**本次结果的相对名次**
      }
    ]
  }
}
```

### 关键约定

1. **`confidence` 只能读相对名次，不能当「匹配度」展示。** 它是对本次结果集做 min-max
   归一化的产物 —— 每次检索**头名恒 1.0、末名恒 0.0**，与绝对相似度无关；同一文件同一
   查询，结果集多一条垃圾就会改变它的显示值。**`Math.round(confidence*100)` 转百分比会
   误导用户**（把「本次排第一」读成「匹配了 100%」，把末名的 `0%` 读成「没索引上」）。
   **推荐做法：展示名次（第 N 位）或按 `score` 分档；若必须给百分比，请在文案里写明它是
   相对排名。** 需要绝对量请用 `score`。
2. **`score` 是未归一化的绝对融合分**（向量/BM25/文件名三路加权后的值，典型 0~1；余弦为
   负时理论上可略低于 0），随结果降序。它的可比性只在**同一 embedding 配置**内成立
   （换模型或换 pooling 后数值尺度会变），可用于排序、分档与调试，**不要**当用户可见的
   精确指标。纯筛选检索（无 query）时恒为 0。
3. **query 为空 + filters 为空** → code=1001（参数错误）。
4. **query 为空 + filters 有值** → 纯筛选检索（按 modifiedAt 降序，confidence=0）。
5. **filters 未传** → 后端 NL 解析自动提取（时间/目录/类型）。
6. `filters.dateRange` 的日期格式为 `YYYY-MM-DD`（本地时区），时区口径与 `createdAt/modifiedAt` 的 `+0800` 后缀一致。
7. **Top-K 是「单次检索最多返回条数」的硬上限**（默认 5，可调范围 1~100）。命中数超过 Top-K 时，结果按 `confidence` 降序截断，`total` 也随之为截断后的条数——即 **`total` ≤ Top-K**，`total` 与 `results` 一致（不会出现「共 N 条匹配」却列不全的情况）。想多看就把 Top-K 调大。
8. **Top-K 只影响「显示几条」，不影响「哪些文件算命中」以及各自的分数**。同一查询在
   Top-K=3 / 10 / 20 下，同一个文件的 `score` 必须完全相同，前几名顺序也必须一致。
   检索深度由 `search.candidate_pool` 独立控制（默认 1000，0 = 不截断；调试旋钮，界面不
   提供）——若把它接到 Top-K 上，块级窗口会随 Top-K 收缩，最佳块排在窗口外的文件分数会
   凭空消失，表现为「改显示条数 → 排序变了」。见 [`configuration.md`](configuration.md)。
9. **语义门槛**：仅向量命中且余弦低于 `search.min_similarity`（默认 0.35）的文件被剔除；
   纯 BM25 命中的文件不受此约束（它们没有 embed 分）。该门槛的**标定当前是失效的**
   （挡不住任何东西）——不要依赖它过滤噪声。
10. **类型词（`python` / `jpg` / `markdown` …）是进 `fileTypes` 还是留在 `keywords`，
    由词性 + 语序决定**：
    - **格式词**（`jpg` `pdf` `docx` `markdown` `excel` `表格` `幻灯片` `配置文件` …）
      ——说文件长什么样。独立出现 / 位于句末 / 后跟「文件 / 文档 / 脚本 / 表格 / 幻灯片 …」
      时进 `fileTypes`（"发票的jpg" → 只要 jpg）；**后面还有别的内容词时留在 `keywords`**
      （"excel公式" / "jpg压缩" → 搜内容）。
    - **语言名 / 数据格式名**（`python` `py` `js` `ts` `java` `cpp` `c` `c++` `cs` `go`
      `rust` `rs` `sql` `sh` `shell` `json` …）——说内容在讲什么。**一律留在 `keywords`**
      （"python" / "python源码剖析" / "java源码" 都搜内容，行为彼此一致）；
      只有后跟「文件 / 脚本」才进 `fileTypes`（"python文件" → py）。
    规则见 `bibit_filesearch.nl_parser`（`_FORMAT_EN_WORDS` / `_CODE_LANG_WORDS` /
    `_TYPE_HEAD_LIMITERS` / `_extract_types`），回归见 `tests/test_type_word_disambiguation.py`。
    补充：**并列类型词**共享一个限定词后缀时全部按中心语收——
    「pdf、md、txt 文档」→ `["pdf","md","txt"]`（限定词只挂在最后一个并列项上，
    不能只收 txt）。判据是 `_only_connectors` 相邻间隙只含连接词，遇到内容词即断段：
    「Python源码剖析和另一个word文件」只收 docx。注意限定词可能落在类型词**跨度内部**
    （中文表条目 `word文件`），也可能隔了空格（`txt 文档`），两者都由
    `_followed_by_limiter` 认；裸并列无限定词（「excel和word」）仍只收句末那个。
    补充（中英对齐）：英文文件性名词（`files` `documents` `scripts`
    `slides` `images` …）同样算限定词，`_TYPE_HEAD_LIMITERS_EN` 按整词匹配，故
    「pdf, md, txt documents」→ `["pdf","md","txt"]`，与中文侧一致；
    `file`/`doc` 不得命中 `filetype`/`profile`。**残留连接词按整词丢弃**：
    类型词净化后句中的「或/及/or/and」等会作为残余关键词把纯筛选查询拖回语义检索
    （「pdf或md文件」kw=`'或'` 时同 filters 下召回 3→1），`_TYPE_CONNECTOR_WORDS`
    在净化收尾整词剔除；只丢独立成词者，`普及率`/`涉及内存`/`world map` 不受影响。

---

## 10. GET /api/files/{fileId}

单文件详情：元数据 + 分块统计。前端点开文件详情用。

### 响应

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "fileId": "f_xxx",
    "fileName": "gtk_realize_map_visable.md",
    "filePath": "/home/<user>/Documents/GTK/gtk_realize_map_visable.md",
    "fileType": "md",
    "fileSize": 3462,
    "createdAt": "2026-08-12T10:34:00+0800",
    "modifiedAt": "2026-08-12T10:34:00+0800",
    "author": "",
    "directory": "/home/<user>/Documents/GTK",
    "chunkCount": 12            // 分块数
  }
}
```

不存在返回 `code=1004, data=null`。

> 本服务**不提供** `summary` / `summarySource` 字段（摘要生成不在范围内）。

---

## 11. 消费方映射说明（供前端 / 插件参考）

| 消费方 | 调用方式 | 读取字段 |
|---|---|---|
| 界面插件（状态 / 进度 / 设置） | `GET /api/scan/progress`、`GET|PUT /api/settings` | 状态机字段、`data.changed` |
| 检索问答编排层 | `POST /api/search {query,page:1,pageSize:10}` | `data.results[].fileName/confidence/filePath` |

> ⚠️ **`confidence` 的常见误用**：把 `confidence` 转成百分比直接拼进回答
> （`f"{fileName}（{conf}%）"`），等于把「相对名次」当成「匹配度」播报给用户。
> 如需向用户展示匹配程度，应使用名次 `#N` 或未归一化的绝对分 `score`（见约定 1）。

> ⚠️ **多层包装陷阱**：调用方若经过一层编排层（如 Agent 转发），拿到的结构可能是
> `{"data": {"code":0, "message":"success", "data":{...}}}`——外层 `success()` 又包了一层。
> 必须 `if (data && data.data) data = data.data` 剥到真正的服务响应，再判 code、再取 data。
> 本服务的四套响应（settings/search/scan/health）结构一致，剥层逻辑可收敛为一个辅助函数。
