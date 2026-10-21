# bibit-filesearch

本地文件语义检索服务。索引用户知识库（默认 `~/Documents`），用自然语言检索本地文档，
**全程不上云**。

服务以 HTTP 契约对外，可独立部署，也可作为上游 Agent 的数据层：调用方递一句话进来，
本服务定位到本地文件并返回结果。每个登录用户一套独立实例（配置 / 数据 / 索引 / 端口），
互不干扰。

```
调用方「本地搜索一下 GTK 相关的文档」
  → POST /api/search   （端口按 uid 派生，默认 18700 段）
  → 返回 1. gtk_realize_map_visable.md（100%）
```

---

## 功能

| 能力 | 说明 |
|---|---|
| 混合检索 | 语义（embedding 余弦，块级 → 文件级 max 池化）+ BM25 稀疏召回 + 文件名通道，三路按真实比例融合（语义 0.538462 / 文件名 0.230769 / BM25 0.230769，和 = 1） |
| 自然语言查询 | 规则确定性提取时间范围 / 目录 / 类型过滤条件，不调 LLM（毫秒级）；`上周桌面上的 pdf` 直接可用 |
| 多格式解析 | 20 类：文本 `txt md docx pdf`；代码 `py js java cpp c cs go rs ts sql sh json`；表格幻灯片 `xlsx pptx`；图片 `png jpg` |
| 图片与扫描件 | PP-OCR（det / cls / rec 三模型），自带方向分类器，歪斜 / 倒置文字直接识读；推理可下沉 SpaceMIT NPU |
| 自动索引 | inotify 增量监听 + 全盘扫描；扫描可暂停 / 继续 / 重启续跑 |
| 幂等 | `content_hash` 去重，内容未变跳过重算；配置变更自动回收 / 重建（格式取消即回收，重新勾选即重建） |
| per-user 隔离 | 每个登录用户一套配置、数据、索引与端口，多用户同机不冲突 |
| 独立部署 | deb 包 + user systemd；向量化走外部 embed 端点，本服务不重复加载模型 |

检索的其余可调项：`search.candidate_pool`（检索深度，默认 1000 块）、
`search.bm25_half_score`（BM25 原始分 → 0~1 的映射常数 K，默认 2.8）、
`search.min_similarity`（语义门槛 0.35，在当前融合分数尺度下区分力有限，属调试旋钮）、
`top_k`（返回条数，默认 5）。`candidate_pool` 与 `top_k` 互不相干：前者管找多深，
后者管显示几条。

---

## 框架结构

```
src/bibit_filesearch/
├── main.py          # 入口：create_app 组装、serve 子命令、uid 派生端口、信号与停机释放
├── config.py        # 配置加载与出厂默认（config.yaml ⊕ 代码默认）
├── settings.py      # 运行时配置层（写入 / 热生效 / 版本迁移 / 解析与分块重排）
├── db.py            # SQLite WAL 存储层（files / chunks / scan_state）
├── indexer.py       # 执行器：文件 → 解析 → 分块 → 向量化 → 入库 + 文件聚合
├── pool.py          # 索引池 + 常驻消费者（索引状态的唯一真相）
├── scanner.py       # 全盘扫描：批量生产者（遍历 / 向量化两阶段）
├── watcher.py       # inotify 增量监听（ctypes 直调，零依赖）+ 目录补扫
├── retriever.py     # 检索：余弦 + BM25 + 文件名通道，三路融合与分页
├── nl_parser.py     # 自然语言 → 结构化过滤条件（规则确定性）
├── parser.py        # 各格式文本与作者提取；PARSER_REVISION 标记解析能力版本
├── chunker.py       # 文本 / 代码分块（重叠窗口，代码按字符预算不超 embed 上限）
├── bm25.py          # 自写 BM25 倒排表（jieba 分词，无第三方检索依赖）
├── excludes.py      # 排除规则判定的唯一定义（scanner / watcher / 回收共用）
├── pathutil.py      # 路径两种呈现（文件系统原始字节 / 接口可编码字符串）
├── metadata.py      # 文件元数据（大小、时间、作者、目录；创建时间有回退链）
├── state.py         # 扫描状态持久化（scan_state，重启可续跑）
├── recovery.py      # embed 恢复看门狗（掉线 → 恢复后自动重跑失败文件）
├── model_client.py  # embed HTTP 客户端（维度以服务探测为准，超时队列二分重试）
├── ocr_engine.py    # OCR：PP-OCR（唯一引擎），设备 auto / npu / cpu
├── ocr_ep.py        # NPU 侧客户端（常驻 C++ 进程 ep_worker）
└── api/             # FastAPI 路由：search / settings / scan / files / meta
```

### 写入路径（索引）

```
scanner（遍历，秒级）         watcher（inotify，近实时）
        └──── offer / cancel ────┘
                    ↓
             索引池 = files 表（一个文件一行，行上记索引状态）
                    ↓ take / commit
        pool.IndexWorker（全局唯一常驻消费者）
                    ↓
   indexer：parser 取文本 → chunker 切块 → embed 端点算向量 → db 落库
```

### 读取路径（检索）

```
query ─┬─ nl_parser：规则提取时间 / 目录 / 类型过滤（不调 LLM）
       ├─ embed 端点：query 向量 → 块级余弦 → 文件级 max 池化
       ├─ bm25：倒排召回 → raw/(raw+K) 映射到 0~1
       └─ 文件名 / 路径关键词通道
              ↓ 三路加权融合（和 = 1）→ 结构化筛选 → 排序 → 分页
```

### 关键设计

- **状态唯一真相**：索引状态只存在于 `files` 表（池行），由唯一的常驻消费者
  （`pool.IndexWorker`）写入；生产者只 offer / cancel，执行器只算不写状态。
  `files` 表的路径唯一约束是「一个文件一行」的物理保障。
- **两阶段扫描**：遍历（`os.walk`，秒级）与向量化（串行消费，慢）分离，
  暂停 / 继续作用于整个任务，状态落库所以重启可续跑。
- **hash 幂等**：`content_hash` 相同则跳过重算；解析能力变更（`PARSER_REVISION`）
  或分块规则变更（`CHUNK_REVISION`）时，受影响的文件按新规则重排一次。
- **融合排序**：三路权重之和为 1，配置表里写的就是真实比例；BM25 原始分用
  `raw/(raw+K)` 映射，不用批内 min-max（后者会把「都命中」拉成「一个满分一个零分」）。
- **per-user**：端口按 uid 派生（uid < 1000 → 18700，否则 18700 + uid - 1000），
  配置与数据都在用户 home 下，unit 用 `ConditionUser=!@system` 排除系统账号。

---

## 平台与前提

面向 **Bianbu（SpaceMiT / riscv64）设备**。以下是「跑起来需要什么」，不是可选优化：

1. **向量化是外部依赖。** 语义路调 `POST http://127.0.0.1:18790/v1/embed/embeddings`
   （默认 `qwen3-embedding-0.6b-q4_0`，1024 维，`--pooling mean`，故 `query_instruction`
   留空）。没有这个端点时检索会报错，不会降级为纯 BM25。可自建任意 OpenAI 兼容
   embed 服务，改 `models.embed.{url,path,model,dim}` 即可。
2. **Python 3.14**（`requires-python >= 3.14`）。
3. **OCR 依赖的 riscv64 轮子只在 SpaceMiT 私有源。** `onnxruntime` /
   `opencv-python-headless` 在 riscv64 上 PyPI 官方无轮子，`debian/postinst` 默认从该源
   安装，可用环境变量 `PIP_INDEX` 覆盖；外部环境需自备轮子或从源码构建。
4. **NPU 是加速项，不是依赖项。** 有 `/opt/bibit-filesearch/ocr-ep/ep_worker` 与
   `libspacemit_ep.so` 就走 SpaceMIT NPU，否则纯 CPU，识别文本一致。

---

## 编译与安装

### 方式一：源码运行

```bash
pip install -r requirements.txt

# 启动（默认端口按 uid 派生；embed 端点需已在 :18790 监听）
PYTHONPATH=src python -m bibit_filesearch.main serve

# 指定端口 / 配置文件
PYTHONPATH=src python -m bibit_filesearch.main serve --port 18700 --config /tmp/dev.yaml

# 验证
curl -s http://127.0.0.1:18700/health
curl -s http://127.0.0.1:18700/health/deep        # db / embed / 索引计数
curl -s -X POST http://127.0.0.1:18700/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"GTK","page":1,"pageSize":5}'

# 首次索引：触发一次全盘扫描并查进度
curl -s -X POST http://127.0.0.1:18700/api/scan -H 'Content-Type: application/json' -d '{}'
curl -s http://127.0.0.1:18700/api/scan/progress
```

首次启动不写配置文件：生效值全部来自代码默认；`index_dirs` 为空时动态填当前用户
的 `~/Documents`（目录存在才给）。

### 方式二：构建 deb 包

```bash
sudo apt install debhelper g++ python3.14 spacemit-onnxruntime   # 构建依赖
dpkg-buildpackage -b -uc -us
# 产物：../bibit-filesearch_<版本>_riscv64.deb
```

构建过程做两件非默认的事：

1. **编译 NPU 辅助进程** `packaging/ocr-ep/ep_worker.cpp`：
   `g++ -O2 -std=c++17 -lonnxruntime -lspacemit_ep`。
   `-lspacemit_ep` 必须显式链接（该符号只在 `libspacemit_ep.so`，不在
   `libonnxruntime.so`）；Python 侧 `spacemit_ort` 建 EP 会话会段错误，
   C API `OrtSessionOptionsSpaceMITEnvInit()` 是唯一可用入口。
2. **安装 user 级 systemd 单元的全局 enable 链接**（`default.target.wants/` 符号链接，
   由 deb 文件安装而非 `postinst enable`），任何登录用户自动拉起。

### 方式三：安装 deb

```bash
sudo apt install ./bibit-filesearch_<版本>_riscv64.deb
```

安装布局：

```
/opt/bibit-filesearch/src/                 源码（PYTHONPATH 指向这里）
/opt/bibit-filesearch/.venv/               postinst 创建，装 requirements.txt
/opt/bibit-filesearch/config.yaml          出厂模板（人类可读参考，运行时不读）
/opt/bibit-filesearch/requirements.txt
/opt/bibit-filesearch/ocr-ep/ep_worker     NPU 推理辅助进程
/usr/lib/systemd/user/bibit-filesearch.service
/usr/lib/systemd/user/default.target.wants/bibit-filesearch.service -> ../…
/usr/share/doc/bibit-filesearch/           版权与变更记录
```

`postinst` 依次做：建 venv 并从 `PIP_INDEX`（默认 SpaceMiT 私有源）安装
`requirements.txt` → OCR / NPU 自检（真跑一次推理；缺 `libspacemit_ep.so` 时只提示
退回 CPU，不阻断安装）→ 对当前已登录的非系统用户 `daemon-reload` + `restart`。

服务管理：

```bash
systemctl --user status  bibit-filesearch
systemctl --user restart bibit-filesearch
journalctl --user -u bibit-filesearch -n 200 --no-pager
```

服务单元不是 root 服务：它以登录用户身份运行，读该用户的配置、索引该用户的目录。
端口按 uid 派生，多用户同机不冲突。

运行时路径（per-user）：

| 用途 | 路径 |
|---|---|
| 用户配置（稀疏，只存改过的键） | `~/.config/bibit-filesearch/config.yaml` |
| 索引数据库（SQLite WAL） | `~/.local/share/bibit-filesearch/filesearch.db` |
| 扫描错误日志 | `~/.local/share/bibit-filesearch/scan_errors.log` |
| 数据目录覆盖 | 环境变量 `BIBIT_FILESEARCH_DATA_ROOT` |

---

## 配置

权威默认值在代码里（`config.DEFAULTS` 与 `settings` 的出厂默认）；
`~/.config/bibit-filesearch/config.yaml` 只存与出厂默认不同的键（稀疏存储），
首次启动不创建，界面保存设置时才落盘。界面保存的设置经 `PUT /api/settings` 热生效，
直接编辑文件则需重启服务。

界面可改的顶层键：

```yaml
index_dirs: []          # 知识库路径；空 → 动态填 ~/Documents
auto_index: true        # watcher 增量监听
indexable_types: [txt, md, docx, pdf, py, js, java, cpp, c, cs, go, rs, ts,
                  sql, sh, json, xlsx, pptx, png, jpg]     # 全 20 类
exclude_patterns: [.git, node_modules, __pycache__]
max_depth: 3            # 1-based；0 = 不限深度
top_k: 5                # 返回条数（显示口径，不影响检索深度）
```

只改配置文件、界面不提供的调试旋钮：

```yaml
search:
  candidate_pool: 1000    # 检索深度：取相似度最高的多少块参与聚合；0 = 不截断
  embed_weight: 0.538462
  filename_weight: 0.230769
  bm25_weight: 0.230769   # 0 = 关闭 BM25 路
  bm25_half_score: 2.8    # BM25 原始分 → 0~1 的映射常数 K
  min_similarity: 0.35
ocr:
  device: auto            # auto / npu / cpu
  max_side_len: 1400      # 峰值内存的主要旋钮
  idle_release_secs: 10   # 空闲多久释放 OCR worker（0 = 关）
  worker_max_runs: 0      # 单 worker RUN 次数上限（0 = 关）
  worker_max_rss_mb: 0    # 单 worker RSS 上限 MB（0 = 关）
  pdf_max_ocr_pages: 20   # 单个 PDF 最多 OCR 多少页（0 = 不限）
models:
  embed:
    url: http://127.0.0.1:18790
    path: /v1/embed/embeddings
    model: qwen3-embedding-0.6b-q4_0
    dim: 1024
```

---

## HTTP 接口

统一响应 `{code, message, data}`，`code = 0` 为成功。

| Method | Path | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/health/deep` | 深度健康（db / embed / 索引计数） |
| GET | `/api/meta` | 实例归属：uid / userName / homeDir / 派生端口 / OCR 状态 |
| POST | `/api/search` | 检索（query + filters + 分页） |
| GET | `/api/files/{fileId}` | 文件详情（元数据 + 分块统计） |
| POST | `/api/scan` | 触发全盘扫描（幂等） |
| GET | `/api/scan/progress` | 两阶段进度 |
| POST | `/api/scan/pause` · `/api/scan/resume` | 暂停 / 继续 |
| GET | `/api/settings` | 读配置 |
| PUT | `/api/settings` | 改配置（幂等，`data.changed` 标明是否真的变了） |
| POST | `/api/settings/reset` | 恢复默认 |

`/api/search` 返回的 `score`（融合分，可超 1）与 `confidence`（归一化后的展示分）语义不同，
不要混用。

---

## 依赖

运行（`requirements.txt`）：`fastapi` / `uvicorn` / `PyYAML` / `jieba` / `numpy` /
`pymupdf` / `python-docx` / `python-pptx` / `openpyxl` / `rank-bm25` / `httpx` /
`rapidocr` / `onnxruntime` / `opencv-python-headless`。

- 向量化后端：embed 端点 `:18790` → `POST /v1/embed/embeddings`
  （Qwen3-Embedding-0.6B Q4_0，1024 维，mean pooling）。
- OCR：`rapidocr` 自带 PP-OCRv6 det / cls / rec 三个 onnx 模型（约 32MB，随 wheel
  安装，不联网下载），推理走 `onnxruntime`。OCR 引擎唯一：识别失败如实落成
  `failed` 并带原因，不换引擎静默重试。
- 自行验证：`PYTHONPATH=src python -m pytest tests/ -q`（需 `pytest` / `pytest-asyncio`；
  OCR 与 embed 用例自带假引擎，不需要真模型或 NPU）。

---

## 许可

Apache-2.0，见 `LICENSE`。
