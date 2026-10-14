# 扫描/索引状态契约（前端显示口径 + 状态机语义）

> 适用版本：`bibit-filesearch 0.1.17+`
> 消费者：GUI 插件（状态格、进度格）、`fs-search --timing`、`GET /api/scan/progress` 的任何调用方
> 相关代码：`scanner.py::progress()` / `indexer.py::index_file` / 前端 `FilesearchPage.tsx::statusPill`
> 状态真相收归池子，见 [`adr/0001-single-index-pool.md`](adr/0001-single-index-pool.md)。

核心不变式：**对外 `status` 必须回答「还有没有活在干」**。索引有两条路径
（`scanner` 全盘扫描 / `watcher` inotify 增量 + 补扫），两条都经 `indexer.index_file`
这一个入口，故在飞状态由该入口统一登记（见 §3.3）。

## 1. 前端显示的状态（用户可见文案）

状态格（`fs-pill`）是**唯一**的用户可见状态出口。按下列**优先级从高到低**取第一个命中的：

| # | 显示文案（zh-CN） | key | 判据（前端） | 语义 |
|---|---|---|---|---|
| 1 | 服务不可用 | `pill.down` | `serviceDown` | 连不上本机服务（`/health/deep` 与 `/api/scan/progress` 都失败） |
| 2 | 连接中… | `pill.connecting` | `serviceConnecting` | 首次探测未回 |
| 3 | 扫描中 | `pill.scanning` | `progress.status === 'scanning'` | **索引管道还有活在干**（遍历/排队/算向量三者任一） |
| 4 | 已暂停 | `pill.paused` | `progress.status === 'paused'` | 用户暂停，管道停下 |
| 5 | 索引服务不可用 | `pill.embedDown` | `embedUp === false` | 服务在，但 embed 后端不可达 → 索引算不了 |
| 6 | 索引出错 | `pill.error` | `progress.status === 'error'` | 扫描本身出错 |
| 7 | 已完成（N 个失败） | `pill.doneWithFailures` | `status === 'done' && failed > 0` | 管道空了，但有文件永久失败 |
| 8 | 已完成 | `pill.done` | `progress.status === 'done'` | 管道空了，且无失败 |
| 9 | 服务正常 | `pill.idle` | 兜底（`status === 'idle'`） | 尚未扫描过 |

第 3 项与第 7/8 项的分界，就是**「管道里还有没有活」**——增量索引（watcher 路径）期间
必须命中第 3 项，而不是第 8 项。

三格计数（`stats.*`）与状态格**同源同长**，不是独立状态：

| 显示 | key | 取值 |
|---|---|---|
| 已索引文件 | `stats.indexed` | `files` 表中 `vectorize_status='completed'` 的行数 |
| 待向量化 | `stats.queued` | **`pending` = `queued + vectorizing`** —— 池中还没算完的行数 |
| 失败文件 | `stats.failed` | `vectorize_status='failed'` 的行数 |

「待向量化」与进度行文字取**同一个数**（后端 `vectorizePhase.pending`），两处必须一致。

**不要把进程内的在飞计数加进来**（曾写作 `queued + vectorizing + inflight`）：那是重复
计数——同一文件既可能是库里的 `vectorizing` 行、又被计入在飞，实测 8 个文件的库能算出 9。
池子化后 watcher 增量也写池行，故 `pending` 即精确值；`inflight` 退化为纯
「此刻有没有活在干」的信号（界面上是 0 或 1），用于状态派生，不参与这个数。

进度行（`scanProgressView`，纯函数，见 `src/scanProgress.ts`）：

- 遍历未完成 → 「正在遍历文件 x / N」，不画条（遍历期间画条会出现「100% + 扫描中」的自相矛盾）；
- 遍历完成 → 「已处理 x / N 个文件（待 P）」，`P = queued + vectorizing`。

## 2. 状态机（后端）

后端只有一份状态机，落在 `scan_state.scan_status`（持久）+ `ScanService._status`（内存，权威）：

```
         POST /api/scan
              │
   ┌──────────▼──────────┐
   │      scanning       │◄──────┐  resume
   │ （遍历 + 排队 + 算）  │       │
   └──┬───────────┬──────┘   ┌────┴─────┐
      │           │   pause  │  paused  │
      │           └─────────►│          │
      │                      └────┬─────┘
      │ 遍历完成 + 队列排空          │（resume 回 scanning）
      ▼                            
   ┌─────────┐   再 POST /api/scan   ┌────────┐
   │  done   │ ─────────────────────►│scanning│
   └─────────┘                       └────────┘
   ┌────────┐ 服务刚起、尚未扫过
   │  idle  │
   └────────┘
   ┌────────┐ 扫描过程抛异常
   │ error  │
   └────────┘
```

### 2.1 每个状态的语义（**唯一判据 = 管道里有没有活**）

| 状态 | 语义（回答"现在还有文件在算吗"） | 进入条件 | 退出条件 |
|---|---|---|---|
| `idle` | 还没扫过 | 服务启动初值 | 任一 `start()` |
| `scanning` | **有活在干**：正在遍历、或有文件排队、或有文件正在算向量 | `start()` / `restart()` / `resume()`；或"已 done 但检测到在飞索引"（见 §2.2） | 遍历完成 **且** `queued == 0` **且** 无在飞 → `done` |
| `paused` | 用户暂停；管道停下，计数冻结 | `pause()` | `resume()` |
| `done` | **管道空了**：遍历完成、队列排空、无在飞 | 由 scanning 自动转入 | 再次 `start()` |
| `error` | 扫描本身失败 | 遍历异常 | 再次 `start()` |

### 2.2 `done` 判据里的「在飞索引」

worker 的队列判据只看 DB 的 `queued` 行，而 `queued` 只由 `scanner._enqueue_file` 写：
watcher 路径不写 `queued`、直接调 `indexer.index_file()`，于是 watcher 干活期间队列为空、
`done` 判定成立、状态被钉死在 `done`。故 `done` 的判据补上「在飞索引」：

① `indexer.py` 记**在飞索引计数** `inflight_indexes`（进程内整数）：
   `index_file()` 入口 `+1`、`finally` `-1`。它是 scanner worker / watcher 事件 /
   watcher 补扫三条路径的**唯一公共入口**，故一个计数器覆盖全部索引来源。
   用 `finally` 递减（漏计会让状态永久卡在 scanning）。

② `scanner.py::progress()` 出口派生有效状态：

```python
if status == STATUS_DONE and (inflight > 0 or queued > 0):
    status = STATUS_SCANNING        # 还有活在干，别报"已完成"
```

派生放在 `progress()` 出口而不是改 `_status`：`_status` 由 worker 线程写、`progress()` 读，
改 `_status` 要多线程同步；放在出口是**纯读**，无锁、无竞态、无新线程。

### 2.3 同路径单飞

同一文件被 `scanner` 与 `watcher` 各算一遍的根因：两条路径都在索引**开始之前**查库问
「这文件在处理吗」，而库里那行是在 `embed_batch` **完成之后**才 `upsert_file` ——
整个 embed 窗口里该文件在库中不存在，于是两条路径都认为自己是第一个。

`Indexer.index_file` 承担**同路径单飞**（与 §2.2 的 `inflight_indexes` 同属该入口）：

```
_active_paths: dict[path -> {done: threading.Event, owner: tid, result, error}]
第一次来 → 登记占位、inflight+1、真算；算完把结果写进槽 → 摘牌 → 放行等待者
同路径又来 → 不解析，等赢家收尾，**复用它的返回值**（赢家抛异常则转抛给等待者）
同线程重入 → 不等待（等自己 = 自锁），放行重算一遍并留 warning
```

选「等 + 复用结果」而不是「直接丢掉重复请求」：丢掉会让索引窗口内到达的变更事件
**彻底没人处理**（直到下一次事件或全盘扫描才补），而赢家那次解析读的是**旧内容** ——
于是「跳过重复解析」退化成「**跳过更新**」。等待把两件事分开：重复的解析省掉了，结果照样送达。
代价可忽略：embed 是单槽串行的，输的那一方即使不等也拿不到算力。

单飞用进程内集合，不「提前写库占位」：提前 upsert 一行会同时牵动 `progress()` 的
total/queued/unsupported 三格语义、`_run_worker` 的 `skipped` 兜底、`_recover_interrupted`、
`truncated_files`/`requeue_truncated` —— 一处改动、多个消费端变义。单飞只动 `indexer`
一个文件，且**不动任何持久状态**（重启后无残留）。

**覆盖边界**：

- **覆盖**：三条索引路径（scanner worker / watcher 事件 / watcher 补扫），它们共享同一个 `Indexer` 实例。
- **不覆盖**：跨进程 / 重启窗口的重复（那种窗口落在 `_recover_interrupted` 与 hash 快路径之间）。
- **前提**：`Watcher` 与 `ScanService` 必须共享**同一个** `Indexer` 实例。若给 watcher 单独造
  一个 indexer，本判据会**静默失效**。

**连带影响**：

1. 窗口内 `inflight_indexes` 仍计 **1**（不是 2）—— 计数语义与 §4 不变式不变；
2. 重复请求不再产生第二次 `embed_batch`，因此「待向量化」格在窗口内短暂显示 `1` 是**对的**（确实在算）；
3. 兜底路径（等超时 / 同线程重入）返回 `{"status": "inflight"}` 时，`scanner` worker 会把
   `vectorize_status` **还原成进来之前的值** —— 不还原的话那一行会被永久钉在 `vectorizing`
   （既不重试也不修正，界面「待向量化」永远显示 1），只能等下次启动 `_recover_interrupted` 复位。

**测试缺口**：「放行等待者之前必须先摘牌」这条顺序契约**没有测试保护**，
三次构造均为假绿（见 `tests/test_single_flight_index.py` 用例注释），目前只有代码注释 + 变异说明。

## 3. 后端边界（明确不动的部分）

- 不改 `scan_state.scan_status` 的持久值：写进 DB 的仍是 `ScanService` 的真实动作状态，
  派生只发生在 `progress()` 的返回副本上。重启后行为不变。
- 不改 `pause()` 语义：暂停优先于派生（`status != done` 时不派生）。
- 不改 `_status` 的写入时机、不加锁、不加线程。
- 不改 `queued` 的写入者（仍是 `scanner._enqueue_file`）。
- 不改 hash 快路径的判据（仍是「`status='completed'` 且 `content_hash` 相同」）。
- 不改 `upsert_file` 的时机（仍在 `embed_batch` 之后）—— 这是 §2.3 选「进程内单飞」
  而非「提前写库占位」的直接原因。

## 4. 计数一致性不变式

```
files 表总行数 == completed + failed + unsupported + queued + vectorizing
completed + failed + unsupported == 已处理（进度分子）
queued + vectorizing == 待向量化 == 待处理（**统计格与进度行同一口径**）
状态 == scanning  ⟺  遍历进行中 ∨ queued > 0 ∨ 有在飞索引
状态 == done      ⟹  遍历完成 ∧ queued == 0 ∧ 无在飞索引
```

`inflight`（尚未登记进池的文件，界面上 0 或 1）不参与上表的任何一个数：它在库中还没有行，
加进「待向量化」会与库里的 `vectorizing` 行重复计数；它只用于派生状态（`scanning`）。

最后一条是核心不变式：**`done` 保证"真的没有活在干了"**。

## 5. 场景对照

| 场景 | `progress().status` |
|---|---|
| `scanner` 全盘扫描中 | `scanning` |
| watcher 增量索引中（拖文件 / 拷目录） | `scanning` |
| 遍历完成、队列排空 | `done` |
| 用户暂停 | `paused` |
| 服务刚起未扫 | `idle` |

## 6. 验收判据

1. `cp -r ~/pdf ~/test/` 后 5 秒内，`GET /api/scan/progress` 的 `status` 变为 `scanning`，
   GUI 状态格显示「扫描中」。
2. 索引完成后 ≤ 2s（前端轮询间隔），状态自动回 `done` /「已完成」。
3. 大文件（超过 `max_chunks_per_file`）在算期间，状态维持 `scanning` 不掉回 `done`。
4. 异常路径（解析失败 / 权限失败）不留计数残留——之后状态仍能回到 `done`。
5. `paused` 不被派生覆盖。
