# 配置参考

> 权威来源是**代码**：`config.py` 的 `DEFAULTS` ⊕ `settings.py` 的 `DEFAULTS`（由
> `settings._factory_defaults()` 合并）。`/opt/bibit-filesearch/config.yaml` 只是**人类可读
> 的参考模板，运行时不读取**——改它不生效。两者一致性由 `test_config_defaults_parity` 守护。

## 三层与优先级

```
~/.config/bibit-filesearch/config.yaml   ← 运行时实际读的（用户层，稀疏）
        ↓ 缺的键回落
代码里的出厂默认（settings._factory_defaults）
```

- **用户文件是稀疏的**：只写「与出厂默认不同」的键。首次启动**不建文件**，首次在界面
  保存设置时才落盘，且只写改过的键 + `config_version`。
- 所以**文件里没有某个键 = 用出厂默认**，属正常，不是缺配置。
- 判断「实际生效值」永远看 `GET /api/settings`，不要看文件。
- 改完配置文件需**重启服务**；界面保存设置只写它自己那几个键，不会抹掉你手工写的。

```bash
systemctl --user restart bibit-filesearch
curl -s http://127.0.0.1:18700/api/settings      # 生效值（权威）
```

## 界面可改的键（6 个）

界面保存会写回用户文件；「恢复默认」= 把它们全删掉（回落出厂默认）。

| 键 | 默认 | 说明 |
|---|---|---|
| `index_dirs` | `[]` → 首启动态填 `~/Documents` | 知识库目录（绝对路径）|
| `auto_index` | `true` | 关掉则不启动 watcher（**重启才生效**）|
| `indexable_types` | 20 类：`txt md docx pdf xlsx pptx py js java cpp c cs go rs ts sql sh json png jpg` | 可索引扩展名；不在列表里的类型只登记元数据，状态 `unsupported` |
| `exclude_patterns` | `[.git, node_modules, __pycache__]` | 排除规则；匹配路径**任一组成部分**（写目录名即可）|
| `max_depth` | `3` | 扫描深度，1-based：`1` = 仅知识库根目录本身，`0` = 不限 |
| `top_k` | `5` | **最多返回几条**（排序后截断的硬上限）。与检索深度无关，见下 |

## 只改配置文件的调试旋钮（`search.*`，界面不提供）

界面白名单不含这些键 → 界面保存设置不会写它们、也不会抹掉你手工写的。填法：
写进 `~/.config/bibit-filesearch/config.yaml` 后重启服务；**还原就是删掉那几行**。

```yaml
search:
  candidate_pool: 1000     # 检索深度
  min_similarity: 0.35     # 语义门槛
  bm25_weight: 0.230769     # BM25 路权重（0 = 关闭）
  embed_weight: 0.538462    # 语义路权重
  filename_weight: 0.230769 # 文件名/路径路权重
```

| 键 | 默认 | 说明 |
|---|---|---|
| `search.candidate_pool` | `1000` | 检索深度：最多取相似度最高的多少个**块**参与文件级聚合（`≤0` = 不截断）。**与 `top_k` 完全无关**——`top_k` 只管显示几条，本值管找多深。填小会漏掉「最佳块排得靠后」的文件，填大只是变慢（10 万块 ≈ +1.6ms）。默认 1000 的依据：界面 `top_k` 上限 100 × 每文件约 6 块 ≈ 600，留余量 |
| `search.min_similarity` | `0.35` | 语义门槛：文件级最高相似度低于此值即丢弃（豁免纯 BM25 词法命中的文件）。作用在**原始 cos** 上，而展示的是融合分——两者尺度不统一：低 cos 且无关键词命中的文件会整份消失（是"搜不到"而非"排错"）。设 `≤0` 等于关掉 |
| `search.bm25_weight` | `0.230769` | BM25 路权重（0 = 关闭）。技术词/英文 token（`gsettings`、`wl_surface`）靠它一击必中 |
| `search.embed_weight` | `0.538462` | 语义路权重。三路权重直接写**真实比例（和 = 1）**，公式 `0.538462×S_sem + 0.230769×S_name + 0.230769×S_bm25` —— 若写 `0.7` 再靠代码除 ΣW，配置里显示的系数与实际不符 |
| `search.filename_weight` | `0.230769` | 文件名/路径关键词的加权系数。该通道与 BM25 共用**同一套分词（`cut_for_search`）与匹配规则（token 集合相交）**，只有"匹配哪个字段"不同 |

## 其余键（一般不用动）

| 键 | 默认 | 说明 |
|---|---|---|
| `models.embed.url` | `http://127.0.0.1:18790` | embed 端点地址（OpenAI 兼容）|
| `models.embed.path` | `/v1/embed/embeddings` | 推理路径 |
| `models.embed.encoding_format` | `float` | 响应编码；端点契约要求显式声明 |
| `models.embed.load_path` | `/v1/embed/models/load` | 模型加载接口。推理返 `503`（未加载）时先调它再重试一次，不消耗业务重试；`load_timeout` 默认 `150`（读 GGUF + 建 KV 可能耗时百来秒）|
| `models.embed.dim` | `1024` | **只是提示**：启动时以服务探测值（`probe_dim()`）为准。配置里与探测值不符的旧值会被当化石清理（见 troubleshooting 现象 G）|
| `models.embed.model` | `qwen3-embedding-0.6b-q4_0` | 请求体 `model` 字段，**必传**（端点按它选模型）|
| `models.embed.query_instruction` | `""` | 查询指令前缀。默认空：底层 embed 服务用 `--pooling mean`，前缀会淹没用短查询；且索引侧从不加前缀，加了会两侧不对称 |
| `models.embed.batch_size` | `32` | 单请求最大输入数（`embed_batch` 分批上界）。防单个大文件整包一发灌爆 embed 服务（如 minified JS 可达 8 万块）|
| `models.embed.timeout` | `60` | 单请求读超时（秒）。语义是「**一批**（`batch_size` 条）算完的预算」——非流式接口整批算完才返回。超时后按 `32→16→8→4` 缩小工作单元重试（`_MAX_TIMEOUT_SPLIT_DEPTH` 3 层），故 60s 只需覆盖「一个能过的批」|
| `models.embed.connect_timeout` | `5` | 连接超时（秒）。本机回环连不上应秒级失败 |
| `models.embed.probe_timeout` | `5` | 探活超时（秒）。**只在空闲时用得上**：健康判定「被动优先」——最近 `max(2×timeout, 120)` 秒内真的成功产出过向量就直接判健康、不发探针（`EmbedClient.is_healthy()`）。因为探针要与批次抢同一个串行服务，扫描中它排在队尾必然超时，会把满负荷的后端误判成 `unreachable`（界面在「扫描中」与「索引服务不可用」之间抖动）|
| `chunk.size` / `chunk.overlap` | `500` / `100` | 分块大小与重叠（字符）。重叠语义是「边界处的句子在两块里各出现一次」：代价是 `overlap/(size-overlap)` 的重复向量（默认 100/400 = 25%），耗时与向量数同比上升 |
| `scan.io_throttle_ms` | `2` | 遍历节流 |
| `scan.vectorize_parallelism` | `1` | 向量化并发（embed 串行以免抢资源超时）|
| `scan.max_chunks_per_file` | `64` | 单文件最多索引多少块（`0` = 不限）。超限只索引前 N 块，并把**截断前的真实块数**记进 `files.total_chunks`；截断事实经 `/api/scan/progress` 的 `vectorizePhase.truncated` / `truncationDetails` 持久暴露，服务日志另留一条 WARNING。约束是**时间**而非内存（`--cache-ram 0` 下边际内存成本≈0）。改这个值会自动触发一次存量重索引（见下 `CHUNK_REVISION`）|
| `ocr.device` | `auto` | OCR 推理设备：`auto` / `npu` / `cpu`。`auto` = NPU 可用就用（要 `/opt/bibit-filesearch/ocr-ep/ep_worker` 与 `libspacemit_ep.so` 齐），否则 CPU。NPU 比 CPU 快约 1.7~2.9 倍（文本逐字一致），故 `auto` 之外的值一般只在排查时用 |
| `ocr.max_side_len` | `1400` | 送进识别前把长边缩到多少像素（传到 RapidOCR 构造参数）。调小省内存/时间、小字可能丢；调大反之。引擎的 ONNX arena 高水位随**输入面积**上涨且不归还，故它是内存旋钮而非画质选项（见 `troubleshooting` 现象 J）。改它必须配 `PARSER_REVISION` +1 |
| `ocr.idle_release_secs` | `10` | 空闲多久放掉全部 `ep_worker`（秒，`0` = 关）。**运行期唯一的释放触发点**：只有「距上次真正 OCR 满本值」才释放（"池空即释放"不代表"不会再用了"，也与内存贵不贵无关）。定价：释放后第一张 OCR **多付约 1.5s**（重建 det/rec/cls 三个模型），故别设太小。内存真贵时的及时归还有 `worker_max_rss_mb` 兜 |
| `ocr.pdf_max_ocr_pages` | `20` | **单个 PDF 最多对多少页做 OCR**（`0` = 不限）。配合「PDF 逐页判据」：某页文本 < 40 字就对该页提图 OCR，故不设闸时整本扫描书会逐页 OCR（单页数秒，几十页即数十分钟的 NPU 占用）。正常书碰不到它（预算早停通常 10~40 页内就攒够 64 块），它只拦病态输入。闸按「**尝试** OCR 的页数」计，不按「出字的页数」——空白扫描书每页都试、都读不出字，按后者计会永远拦不住 |
| `search.bm25_half_score` | `2.8` | BM25 原始分 → 0~1 的映射常数 K（`S_bm25 = raw/(raw+K)`）。用绝对映射而非批内 min-max：min-max 把「本次命中集」的最小/最大分映成 0/1，命中集小且分数接近时会把「都命中」拉成「一个满分一个零分」（两个原始分仅差 0.0013 的文件可被拉成 1.0 / 0.0 → 排名首尾）。`K` 由**判据法**标定（让可人工验证查询的最差相对裕度最大化）。★ 改 `chunk.size` / `scan.max_chunks_per_file` 或换语料后须重标，见 [`bm25-score-calibration.md`](bm25-score-calibration.md)。`0` = 退化为「命中即 1」的指示函数 |
| `ocr.worker_max_rss_mb` | `0` | 单个 `ep_worker` 常驻内存上限（MB，`0` = 不限）。达到即退休、下次取用时重建。只挑大图的 `det`（`rec` 自然水位 ~1050MB、`cls` ~99MB）。**只降持有时间，不降峰值** |
| `ocr.worker_max_runs` | `0` | 单个 `ep_worker` 跑满多少次 RUN 就退休重建（`0` = 不按次数退休，只靠空闲释放）。防「一次扫描几千张图」时高水位钉在最贵那张的水平 |
| `scan.max_attempts` | `3` | 单文件最多重试几次；达上限即停排（防「每轮全量重灌」）|
| `scan.embed_recovery_cooldown` | `300` | 两次自动重建（看门狗触发）之间的最小间隔秒数，用于打断「embed 被灌爆 OOM → 重启 → 立刻再灌一轮」的正反馈 |
| `server.cors_origins` | 5 项（含 `tauri://localhost`）| GUI 走本地 HTTP，一般无感 |
| `nl.enabled` | `true` | 自然语言查询解析（规则式，不调 LLM）|
| `logging.level` | `INFO` | 日志级别 |

## 配置版本与迁移

`config_version` 记录用户文件的版本（当前 **7**）。迁移做两件事：

1. **删化石**：`_CHANGED_DEFAULTS` 登记「改过的默认值」——用户文件里某键的值**恰好等于**
   旧默认值，说明只是首启复制进来的、用户从未改过 → 删掉，让新默认生效。若值不等于旧默认，
   视为用户真改过 → 保留。
2. **配套索引处理**：个别键改默认不能只删键。典型是 `models.embed.dim`——它约束**存量数据的
   物理形态**，删配置不会让索引跟着变，必须配 `reconcile_embed_dim` 一起用。**只删键会让用户
   从「静默搜不到」掉进「启动即报维度不符」，更严重**。同类见 `parser.PARSER_REVISION`
   （解析能力版本，清 hash 重排）。

## 水位线一览（改这些行为必须一起 +1）

有两类改动**不会**体现在 `content_hash` 里（它是文件**字节**的 hash），所以必须靠独立的
版本号来触发一次性重排，否则「改了却对存量数据不生效」：

| 水位线 | 值 | 位置 | 管什么 | 触发函数 |
|---|---|---|---|---|
| `parser.PARSER_REVISION` | **9** | `parser.py` | 解析器**能不能读到**内容、以及读**到什么内容**。含 PDF 解析策略（两遍 + 逐页薄页判据 + 预算早停 + OCR 页闸 → 影响面是**全部 PDF**，不只图片型）、OCR 引擎与输入上限（改缩放会改检测框与行切分 → 少量小字变化，而 `content_hash` 看不出文本变了）、docx 文本框补扫 | `settings.reconcile_parser_revision` → `db.requeue_unsupported` / `requeue_ocr_text` / `requeue_all_pdfs` / `requeue_all_docx` |
| `indexer.CHUNK_REVISION` | **3** | `indexer.py` | 读到之后**留几块**（`scan.max_chunks_per_file`）| `settings.reconcile_chunk_revision` → `db.requeue_truncated` |

两者的水位线都存在 `scan_state` 表里，**每个版本只跑一次**——所以「文件仍然处于截断态」
不会被反复重灌，那是有意为之。

`CHUNK_REVISION` 的重排有两条判据：

| 判据 | 适用 | 表达式 |
|---|---|---|
| **精确** | `total_chunks` 已知（已被新代码索引过）| `files.total_chunks > 已存块数` |
| **老库近似** | 升级前入库的行（`total_chunks` 刚补列、一律为 `0` = 未知）| `已存块数 >= indexer.CHUNK_REVISION_LEGACY_CAP`（旧默认上限 32）|

老库那条为什么必要：只走精确判据的话，老用户升级后一行都不会被重排（`total_chunks`
全是 0），而水位线在**同一次**就被置成已见——机会一去不回，「抬高上限」等于没做。
近似性：旧上限 32 之下库里不可能有 >32 块的记录，故「存了 32 块」= 「可能被截断」，
本来就是整 32 块的文件会被多索引一次；漏判的代价（那部分内容永久搜不到、界面还显示
正常）远大于多算一次，且水位线保证只跑一次。若你曾自己设过更高的上限，这条会漏掉
一部分——`total_chunks` 一落值即自动转为精确判据。

> 默认值变更前后的完整对照看 `settings._CHANGED_DEFAULTS`（那里逐条写了旧值与对应版本），
> 本页不复制一份以免漂移。

## 改出厂默认值的注意事项

**改出厂默认**（改代码里的值）会自动触达所有**没设过该键**的用户——因为用户文件稀疏、
保存时「值 == 默认」的键会被删。但：

- **老版直升级、文件里冻结着旧值的用户不受影响**（他们的值已被当成「用户选择」）。
  要触达这批人，必须在 `_CHANGED_DEFAULTS` 里登记旧值做迁移。
- 需要「追溯改写老配置」时，先确认值是否可能是用户**真心所选**（与前默认值相同）。
- 三个地方可能各写一份默认值（`config.py` / `settings.py` / `packaging/config.yaml`），
  改之前先确认是否漂移——`test_config_defaults_parity` 会抓，但别等它抓。

## 相关

- 契约与返回字段语义：[`api-contract.md`](api-contract.md)
- 设计取舍（为什么配置是稀疏的、为什么模板不复制）：[`design.md`](design.md)
- 排障（配置读错、CORS 空、门槛失效等）：[`troubleshooting.md`](troubleshooting.md)
