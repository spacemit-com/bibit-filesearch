# bibit-filesearch 设计

> 状态：已实现并发布。本文描述**当前**的设计与参数；对外 HTTP 契约见
> [`api-contract.md`](api-contract.md)，全部配置项见 [`configuration.md`](configuration.md)，
> 排查见 [`troubleshooting.md`](troubleshooting.md)。

## 1. 产品定位

文件检索能力（watcher / indexer / scanner / retriever / nl_parser / settings）独立成服务，
供调用方（桌面界面、Agent 编排层、脚本）通过 HTTP 契约调用。
**不内嵌、不背依赖、独立发布（deb 包）**。

## 2. 用户模型：per-user

- **任意登录用户 X**（桌面用户）装了本服务即可使用
- X 的实例索引 X 的 `~/Documents`，配置在 X 的目录，数据在 X 的目录
- 用户间完全隔离

| 项 | 归属 |
|---|---|
| 服务进程 | user systemd（`systemctl --user`），User=X |
| 配置 | `~/.config/bibit-filesearch/config.yaml`（**只存用户改过的键**，不复制模板——见 §3.1）|
| 数据 | `~/.local/share/bibit-filesearch/filesearch.db` |
| 默认索引 | `~/Documents`（`Path.home()` 动态解析，存在才给，装完即用） |
| 端口 | 18700 段：`uid<1000 → 18700`，`uid≥1000 → 18700+(uid-1000)` |

端口按 uid 派生使多用户各自登录时不抢同一端口；`uid=1000` 与 root 同落在 18700，
故同一时刻只有一个用户实例在跑。

## 3. 配置架构

**yaml 是单一来源**：出厂模板在 `/opt/bibit-filesearch/config.yaml`（仅作参考，运行时不读），
生效值 = 代码默认 ⊕ 用户文件；界面经 `GET`/`PUT /api/settings` 读写。

主要键与出厂默认：

- `index_dirs=[]`（首次启动动态填 `~/Documents`）、`auto_index=true`
- `indexable_types` = 全 20 类（含 png/jpg，图片文字走 PP-OCR）
- `exclude_patterns=[.git, node_modules, __pycache__]`、`max_depth=3`、`top_k=5`
- `scan.max_chunks_per_file`：单文件块数上限（`0` = 不截断；与 `top_k` **无关**）

OCR 侧另有五个键，只在配置文件里改，界面不提供：

| 键 | 默认 | 作用 |
|---|---|---|
| `ocr.device` | — | 推理设备（可指向 NPU）|
| `ocr.max_side_len` | `1400` | 输入长边上限，降**峰值**内存 |
| `ocr.idle_release_secs` | `10` | 空闲多久释放 worker；**运行期唯一的释放触发点** |
| `ocr.worker_max_rss_mb` | `0`（关）| RSS 超限即退休 worker（建议 `1500`，只挑大图的 det）|
| `ocr.worker_max_runs` | `0`（关）| 按使用次数退休 worker |

另有**停机**显式释放（`main._release_on_shutdown`：FastAPI lifespan shutdown + 信号兜底，
幂等），不依赖「父进程退出后子进程读到管道 EOF 自清」那个窗口。

`ocr.idle_release_secs` 只在**孤立**到达的图之间生效：长批量扫描期间池一直不空、TTL 不触发，
那种场景的内存靠 `ocr.worker_max_rss_mb`。

写回 yaml 需加锁（并发写安全），schema 固定。

### 3.1 配置存储语义

用户配置文件**稀疏**存储，不复制出厂模板：

| 时机 | 动作 |
|---|---|
| 首启 | **不写文件**；生效值全部来自代码默认 |
| 首次保存设置 | 落盘：只写「与出厂默认不同」的键 + `config_version` |
| 老文件启动 | 一次性迁移：删「值 == 出厂默认」的化石键 + 补版本号 |
| 改回默认值 | 该键从文件里消失（等价于「恢复默认」） |

`config_version` 标记迁移已完成（幂等：已是当前版本不再改写）。
历史默认值清单在 `settings.py` 的 `_CHANGED_DEFAULTS`——默认值变更时在此登记旧值，
迁移才能区分「用户文件里那条是旧默认残留」与「用户自定义」。

**代价**：用户若想把某键钉在「恰好等于出厂默认」的值上（防止将来默认变化影响自己），
当前无法表达；这类需求出现时再引入显式覆盖标记。

## 4. 生命周期

```
deb 安装 → 服务装上，/opt 出厂模板就位（仅作参考），【不扫描】
用户登录 → user systemd 拉起 X 的实例（读 X 配置；老文件先做一次性迁移）

界面「文件检索」插件页：
  ① 配置表单显示生效值（代码默认 ⊕ 用户文件，此时文件可能还不存在）
  ② 用户设置 → 点「保存并索引」
     - 稀疏写回 X 的 config.yaml（只写改过的键 + config_version）
     - 启动数据准备（watcher 快速阶段扫描 → 符合类型/限制的文件）
     - 面板显示：文件书目 + 「剩余 N 未索引」提示
  ③ 下次修改配置再索引：
     - 与旧配置对比（hash 幂等）
     - 增量更新数据库，内容未变则跳过
```

## 5. 服务组件

| 组件 | 职责 |
|---|---|
| indexer | 文件→分块→embed→入库（hash 幂等） |
| watcher | inotify 增量监听 |
| scanner | 全盘扫描 + 向量化队列（串行限速） |
| retriever | embed + BM25 + 文件名 三路融合 + 结构化筛选 |
| nl_parser | 自然语言→结构化条件（规则确定性，不调 LLM） |
| settings | 配置读写（yaml 单一来源） |

不提供：rerank / summary / mineru（PDF 只用 pymupdf）/ LLM 解析路径。

依赖：jieba / numpy / pymupdf / python-docx / python-pptx / fastapi / uvicorn / pyyaml。

embed：经外部 embed 端点（默认 `http://127.0.0.1:18790`）的 `POST /v1/embed/embeddings`
取向量，须显式传 `model=qwen3-embedding-0.6b-q4_0`。模型为 **Qwen3-Embedding-0.6B Q4_0，
1024 维，`--pooling mean`**；端点负责模型的注册/加载/卸载，底层推理进程端口由它动态分配
（端口不固定，不要写死）。维度以服务探测值为准（`EmbedClient.probe_dim`），配置里的
`dim` 只是提示（见 `troubleshooting.md` 现象 G）。

## 5.1 PDF 解析策略

PDF 走**两遍 + 逐页判据 + 预算早停 + OCR 页闸**：

```
Indexer.index_file
  └─ need_chars_for_chunks(chunk_size, overlap, max_chunks_per_file)   ← 换算住 chunker
       → min_chars（64 块 ⇒ 25301 字符）
  └─ parse_file(path, min_chars=25301)
       └─ _parse_pdf(path, errors, min_chars)
            pass 0  逐页抽文本，只累加字数（4.6ms/页，恒廉价）
                    够预算 ⇒ 直接返回、**0 次 OCR**
            pass 1  不够才逐页：文本厚(>=40 字)直接用；
                    薄(<40 字) → 提该页内嵌图 OCR，结果**内联在该页位置**
                    累进到 >= min_chars 即停（"填满就不再往后看"）
            闸      PDF_MAX_OCR_PAGES（出厂 20）：按"尝试 OCR 的页数"计
```

**分层**：`parser` 不认识"块"（`grep chunk parser.py` 零命中）——传进去的是
"最少需要多少字符"，换算的权威在 `chunker.need_chars_for_chunks`（与 `chunk_text`
同文件、同步演进）。`min_chars=0`（默认）= 不早停，整份解析。

两处工程约束：

- OCR 文本必须**内联**（不是追加到末尾）：混合型 PDF 的正文会先占满前面几十块，
  排尾部的 OCR 文本会落到第 64 块之后 ⇒ 花了 OCR 时间却一块都进不了索引。
- 页闸按「**尝试**页数」计，不按「出字页数」：空白扫描书每页都试、都读不出字
  （出字页数恒 0），按后者计则闸永远不触发、无限往后翻。

**存量的重排面**：`PARSER_REVISION` 变更 + `db.requeue_all_pdfs()`。
影响面是**全部 PDF**，不是「图片型 PDF」——逐页判据影响最大的一类恰恰是混合型
（有文本层、`pdf_images` 为空，按「有无内嵌图 OCR 文本」的旧判据一个都命中不到）。

## 5.2 检索打分融合

### 公式

```
score = (embed_weight·S_sem + filename_weight·S_name + bm25_weight·S_bm25) / ΣW
```

| 路 | 符号 | 绝对分怎么来 | 出厂权重 |
|---|---|---|---|
| 语义 | `S_sem` | 块级 cos → 文件级 **max 池化**（尺度即 cos，0~1） | 0.538462 |
| 文件名 | `S_name` | jieba 分词后，「文件名 + 完整路径」的命中词数 / 总词数 | 0.230769 |
| BM25 | `S_bm25` | `raw / (raw + K)`，K = 2.8（见 [`bm25-score-calibration.md`](bm25-score-calibration.md)） | 0.230769 |

三路**各自先算 0~1 的绝对分，再一次性加权平均**，权重只乘一次，且权重表直接写真实比例
（和 = 1）——不写「0.7」再靠代码除 ΣW，那样配置里显示的系数与实际不符。

两条词法通道（文件名 / BM25）共用同一套分词与匹配规则：`bm25.tokenize`
（中文用 jieba `cut_for_search`）+ `_dedup` + **token 集合相交**，只有"匹配哪个字段"不同。
覆盖两者即可：改分词或换语料后 BM25 或 `K` 需重标。

BM25 绝对分用 `raw / (raw + K)`（而非批内 min-max 归一）：

- 批内归一在「一个满分、其余接近零」的病态输入下会把微小分差拉成 1.0 / 0.0，
  令排名剧烈翻转，未命中 BM25 的文件被挤出窗口；
- 绝对分跨查询可比、且不影响 `score` 的跨查询可比性，是排序所需的性质。

`min_similarity`（默认 0.35）比的是**原始 cos**，不是融合分。语义改写
（如「身份证上的地址」）若没有关键词命中，低 cos 的文件会被整份剔除（是"搜不到"而非"排错"）。
更精确的判据是「`S_sem` 低于阈值 **且** `S_bm25 = 0` **且** `S_name = 0`」——
改它会连带改变结果集大小，需单独一轮标定。

### `score` 与 `confidence` 的分工（别混用）

```
score       未归一化的**绝对**融合分，排序依据。跨查询可比。纯筛选检索恒 0。
confidence  对本次结果集做 min-max（头名恒 1.0、末名恒 0.0）。
            只是**给人看的相对强弱**，不能用来排序 —— 也不是绝对相关度。
```

`_normalize`（min-max）只服务 `confidence` 展示，不用于排序依据。

## 6. API 契约

```
GET  /health                   健康检查
GET  /api/settings             读配置（当前生效值）
PUT  /api/settings             写配置（写回 config.yaml + 热生效 + reconcile 回收）
POST /api/scan                 启动扫描（幂等）
GET  /api/scan/progress        扫描进度（walked/total/vectorize 队列）
POST /api/scan/pause|resume    暂停/继续
POST /api/search               检索（query + filters + page/pageSize）
```

响应契约 `{code, message, data}`，`code=0` 成功。
search results：`{fileId, fileName, filePath, fileType, fileSize, createdAt,
modifiedAt, author, score, confidence}`（无 summary）。完整字段见 [`api-contract.md`](api-contract.md)。

## 7. 调用方接入

```
界面路径（配置 / 状态 / 检索验证）
  桌面 GUI「文件检索」插件页  → HTTP → bibit-filesearch :18700

HTTP 路径
  任何调用方（Agent 编排层 / 脚本 / curl）
    → POST :18700/api/search {query, page, pageSize} → 结果列表
```

## 8. 部署

- deb 包：bibit-filesearch（riscv64）
- 安装：`/opt/bibit-filesearch/config.yaml`（出厂默认，`index_dirs` 首次启动动态填）+
  user service 模板
- postinst：启用**所有登录用户**（unit 装到 `default.target.wants/`，系统账号除外——
  系统账号没有可写的 home），端口按 uid 派生以避免多用户抢同一端口
- 依赖：**外部 embed 端点**（默认 `:18790`），模型 Qwen3-Embedding-0.6B Q4_0 / 1024 维 /
  mean 由该端点管理；缺它时语义路报错，**不降级为纯关键字检索**
