# bibit-filesearch

**本地文件语义检索服务**：索引用户知识库（默认 `~/Documents`），
用自然语言检索本地文档，**不上云**。embedding + BM25 混合召回，NL 查询确定性解析为过滤条件。

可独立部署，也可作为上游 Agent 的数据层：上游（任何 HTTP 调用方）把一句话递过来，
本服务定位到本地文件并回结果。

```
调用方「本地搜索一下 GTK 相关的文档」
  → POST /api/search   （本服务默认 :18700 段，端口按 uid 派生）
  → 返回 1. gtk_realize_map_visable.md（100%）
```

---

## 能力

| 能力 | 说明 |
|---|---|
| **混合检索** | embedding 余弦（块级 → 文件级 max 池化）+ BM25 稀疏召回 + 文件名通道，三路按真实比例融合（语义 ≈54% / 词法 ≈46%） |
| **自然语言查询** | 规则确定性提取时间范围 / 目录 / 类型过滤条件，**不调 LLM**（约 3ms）；`上周桌面上的 pdf` 直接可用 |
| **多格式** | 20 类：`txt md docx pdf` + 代码 `py js java cpp c cs go rs ts sql sh json` + 表格幻灯片 `xlsx pptx` + 图片 `png jpg`（走 OCR） |
| **图片与扫描件** | PP-OCR（det / cls / rec 三个模型），自带方向分类器，歪斜/倒置文字直接识读；推理可下沉 SpaceMIT NPU |
| **自动索引** | inotify 增量监听 + 全盘扫描（遍历快、向量化串行限速），暂停 / 继续 / 重启续跑 |
| **幂等** | `content_hash` 去重；配置变更自动回收 / 重建（类型取消即回收，重新勾选即重建） |
| **per-user 隔离** | 每个登录用户一套配置 / 数据 / 索引 / 端口，互不干扰 |
| **独立部署** | deb 包 + user systemd；向量化经外部 embed 端点，不重复起模型 |

---

## 平台与前提

本组件面向 **Bianbu（SpaceMiT / riscv64）设备**。搬到其他平台前，先看这一节 ——
它列的是「跑起来需要什么」，不是「可选优化」。

1. **向量化不是本仓的能力，是外部依赖。**
   `POST /api/search` 的语义路调 `POST http://127.0.0.1:18790/v1/embed/embeddings`
   （默认 `http://127.0.0.1:18790`），模型 `qwen3-embedding-0.6b-q4_0`、1024 维。
   没有这个端点时检索会**报错**（embed 路在 BM25 之前且无降级），不是「退回纯 BM25」。
   替代方案：自建任何 OpenAI 兼容的 embed 服务，并改 `models.embed.{url,path,model,dim}`。

2. **OCR 依赖里有只有 SpaceMiT 私有源才有的 riscv64 轮子。**
   `onnxruntime` / `opencv-python-headless` 在 riscv64 上 PyPI 官方无 linux_riscv64 轮子，
   `debian/postinst` 默认从 SpaceMiT 的私有源安装，该源可用 `PIP_INDEX` 环境变量覆盖；
   外部环境需自备这些轮子（或从源码构建）。

3. **NPU 是加速项，不是依赖项。** `ocr.device=auto`：有 `/opt/bibit-filesearch/ocr-ep/ep_worker`
   与 `libspacemit_ep.so` 就用 SpaceMIT NPU，否则纯 CPU。实测差距 1.71~2.93×（均值 2.26×，
   四张真图三次取中位数），**识别文本逐字一致**。

4. **Python 3.14**（`requires-python >= 3.11`，打包与实测均为 3.14）。

---

## 快速开始（源码）

```bash
# 1) 依赖（语义检索需 embed 端点已在 :18790 监听）
pip install -r requirements.txt

# 2) 启动（端口按 uid 派生：uid<1000 → 18700，否则 18700+(uid-1000)）
PYTHONPATH=src python -m bibit_filesearch.main serve
# 指定端口/配置：
PYTHONPATH=src python -m bibit_filesearch.main serve --port 18700 --config /tmp/dev.yaml

# 3) 验证
curl -s http://127.0.0.1:18700/health
curl -s http://127.0.0.1:18700/health/deep          # db / embed / 索引计数
curl -s -X POST http://127.0.0.1:18700/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"GTK","page":1,"pageSize":5}'

# 4) 触发一次全盘扫描（首次索引）
curl -s -X POST http://127.0.0.1:18700/api/scan -H 'Content-Type: application/json' -d '{}'
curl -s http://127.0.0.1:18700/api/scan/progress     # 两阶段进度
```

首次启动**不写配置文件**：生效值全部来自代码默认，`index_dirs` 为空时动态填当前用户的
`~/Documents`（目录存在才给）。

---

## 安装与部署（deb）

```bash
sudo apt install ./bibit-filesearch_*_riscv64.deb
# 或： sudo dpkg -i bibit-filesearch_<版本>_riscv64.deb
```

安装布局：

```
/usr/bin/bibit-filesearch                                   CLI 入口（serve 子命令）
/opt/bibit-filesearch/{src,config.yaml,ocr-ep/ep_worker,.venv}
/usr/lib/systemd/user/bibit-filesearch.service              user 单元
/usr/lib/systemd/user/default.target.wants/…                ＠ 全局启用（所有登录用户）
```

`postinst` 实际做的事：

1. 在 `/opt/bibit-filesearch/.venv` 建 venv 并从**私有 PyPI 源**安装 `requirements.txt`
   （见「平台与前提」第 2 条 —— 装机必须能访问该源）。
2. NPU 自检：确认 `ep_worker` 可执行且依赖库齐全，缺 `libspacemit_ep.so` 则只能跑 CPU、不阻断服务。
3. `systemctl --user daemon-reload` + `restart --no-block`。

服务范围：unit 装在 `default.target.wants/`（全局），**任何登录用户**的 user manager 自动拉起；
**系统账号除外**（`ConditionUser=!@system` —— 系统账号没有可写的 home 与桌面目录）。
端口按 uid 派生，多用户同机不冲突。

```bash
systemctl --user status  bibit-filesearch
systemctl --user restart bibit-filesearch
journalctl --user -u bibit-filesearch -n 200 --no-pager
```

运行时路径（per-user）：

| 用途 | 路径 |
|---|---|
| 用户配置（稀疏，只存改过的键） | `~/.config/bibit-filesearch/config.yaml` |
| 索引数据库（SQLite WAL） | `~/.local/share/bibit-filesearch/filesearch.db` |
| 扫描错误日志 | `~/.local/share/bibit-filesearch/scan_errors.log` |
| 出厂模板（**仅人类可读参考，运行时不读**） | `/opt/bibit-filesearch/config.yaml` |

---

## 配置

权威默认值在**代码**里（`config.DEFAULTS` ⊕ `settings._factory_defaults`）。
`~/.config/bibit-filesearch/config.yaml` **只存与出厂默认不同的键**（稀疏存储）——
首次启动不创建，首次在界面保存设置时才落盘。改完需重启服务生效
（界面保存的设置经 `PUT /api/settings` 热生效）。

界面可改的键（顶层）：

```yaml
index_dirs: []          # 知识库路径；空 → 动态填 ~/Documents
auto_index: true        # watcher 增量监听
indexable_types: [txt, md, docx, pdf, py, js, java, cpp, c, cs, go, rs, ts,
                  sql, sh, json, xlsx, pptx, png, jpg]     # 全 20 类
exclude_patterns: [.git, node_modules, __pycache__]
max_depth: 3
top_k: 5                # 检索返回条数（显示口径，不影响检索深度）
```

只改配置文件、界面不提供的调试旋钮：

```yaml
search:
  candidate_pool: 1000    # 检索深度：取相似度最高的多少块参与聚合。0 = 不截断
  embed_weight: 0.538462  # 三路融合的真实比例（和 = 1）
  filename_weight: 0.230769
  bm25_weight: 0.230769   # 0 = 关闭 BM25 路
  bm25_half_score: 2.8    # BM25 原始分 → 0~1 的映射常数 K（S = raw/(raw+K)）
  min_similarity: 0.35    # 语义门槛。⚠️ 当前分数尺度下已无区分力
ocr:
  device: auto            # auto / npu / cpu
  max_side_len: 1400      # 峰值内存的主要旋钮（2000→4916MB / 1400→2983MB）
  idle_release_secs: 10   # 空闲多久释放 OCR worker（0 = 关）；运行期唯一释放触发点
  worker_max_runs: 0      # 单个 worker 的 RUN 次数上限（0 = 关）
  worker_max_rss_mb: 0    # 单个 worker 的 RSS 上限 MB（0 = 关；内存紧张时建议 1500）
  pdf_max_ocr_pages: 20   # 单个 PDF 最多 OCR 多少页（0 = 不限）
models:
  embed:
    url: http://127.0.0.1:18790
    path: /v1/embed/embeddings
    model: qwen3-embedding-0.6b-q4_0
    dim: 1024
```

> ⚠️ `candidate_pool` 与 `top_k` **无关**：前者管「找多深」，后者管「显示几条」。
> 两者曾接到同一个值上，把 Top-K 从 10 调到 5 会连带改排序（已修）。
>
> 完整键表、三层优先级与配置迁移见 [`docs/configuration.md`](docs/configuration.md)。

---

## API

统一响应 `{code, message, data}`，`code=0` 为成功。

| Method | Path | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/health/deep` | 深度健康（db / embed / 索引计数） |
| GET | `/api/meta` | 实例归属：uid / userName / homeDir / 派生端口 / OCR 状态 |
| POST | `/api/search` | 语义检索（query + filters + 分页） |
| GET | `/api/files/{fileId}` | 文件详情 |
| POST | `/api/scan` | 触发全盘扫描（幂等） |
| GET | `/api/scan/progress` | 两阶段进度 |
| POST | `/api/scan/pause` · `/api/scan/resume` | 暂停 / 继续 |
| GET | `/api/settings` | 读配置 |
| PUT | `/api/settings` | 改配置（幂等，`data.changed` 标明是否真的变了） |
| POST | `/api/settings/reset` | 恢复默认 |

`/api/search` 的 `score`（融合分，可超 1）与 `confidence` 语义不同，别混用。
完整字段、错误码与状态机见 [`docs/api-contract.md`](docs/api-contract.md)。

---

## 架构

```
src/bibit_filesearch/
├── main.py          # 入口：create_app 组装 + serve 子命令 + uid 派生端口
├── config.py        # 配置加载与出厂默认（config.yaml 2-tier）
├── settings.py      # 运行时配置层（config.yaml 单一来源，热生效）
├── db.py            # SQLite WAL 存储层（files / chunks）
├── indexer.py       # 文件 → 分块 → 向量化 → 入库 + 文件聚合
├── pool.py          # 索引池 + 常驻消费者（状态唯一真相）
├── scanner.py       # 全盘扫描：批量生产者（遍历 / 向量化两阶段）
├── watcher.py       # inotify 增量监听 + 目录补扫
├── retriever.py     # 检索：余弦 + BM25 + 文件名通道，三路融合
├── nl_parser.py     # 自然语言 → 结构化过滤条件（规则确定性，不调 LLM）
├── parser.py        # 各格式文本与作者提取
├── chunker.py       # 文本 / 代码重叠分块
├── bm25.py          # 自写 BM25 倒排表（无第三方依赖）
├── excludes.py      # 排除规则判定的唯一定义（scanner / watcher / 回收共用）
├── pathutil.py      # 路径两种呈现（文件系统原始字节 / 接口可编码字符串）
├── metadata.py      # 文件元数据提取
├── state.py         # 扫描状态持久化（重启可继续）
├── recovery.py      # embed 恢复看门狗
├── model_client.py  # embed HTTP 客户端（维度以服务探测为准）
├── ocr_engine.py    # OCR：PP-OCR（唯一引擎）
├── ocr_ep.py        # NPU 侧：EP worker 客户端（常驻 C++ 进程）
└── api/             # FastAPI 路由（search / settings / scan / files / meta）
```

关键设计：

- **状态唯一真相**：索引状态只存 `files` 表（池行），由一个常驻消费者（`pool.IndexWorker`）写；
  执行器只算不写状态。设计记录见 [`docs/adr/0001-single-index-pool.md`](docs/adr/0001-single-index-pool.md)。
- **两阶段扫描**：遍历（`os.walk`，秒级）→ 向量化（串行消费，慢）；可暂停、可续跑。
- **hash 幂等**：`content_hash` 相同跳过重算；解析能力变更时按 `PARSER_REVISION` 重排一次。
- **融合排序**：三路权重之和为 1，表里写的就是真实比例；BM25 原始分经 `raw/(raw+K)` 映射到 0~1。

---

## 测试

```bash
PYTHONPATH=src python -m pytest tests/ -q     # 789 passed, 2 xfailed
```

覆盖：BM25 / 混合检索与打分融合 / NL 解析 / 各格式解析 / OCR / 索引 / 扫描状态机 /
watcher / 配置与迁移 / 池子单飞 / 失败恢复。OCR 相关用例自带假引擎（`tests/fake_embed.py` 等），
不需要真模型或 NPU。

---

## 文档

入口是 **[`docs/README.md`](docs/README.md)** —— 它说明哪份是权威、哪份是历史、按什么顺序读。

常用几份：

| 文档 | 内容 |
|---|---|
| [`docs/api-contract.md`](docs/api-contract.md) | 对外 HTTP 接口的**权威基准**（端点 / 字段 / 错误码 / score vs confidence） |
| [`docs/configuration.md`](docs/configuration.md) | 配置参考（三层优先级、全部键、调试旋钮、版本迁移） |
| [`docs/design.md`](docs/design.md) | 设计与取舍（定位、per-user 模型、配置架构、PDF 与打分策略） |
| [`docs/bibit-filesearch-intro.md`](docs/bibit-filesearch-intro.md) | 实现全貌（服务组成、索引写路径、检索读路径、运维） |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | 按现象分支的排查指南 + 版本速查 |
| [`CONTEXT.md`](CONTEXT.md) | 领域术语表 |

---

## 依赖

运行（`requirements.txt`）：`fastapi` / `uvicorn==0.52.1` / `pymupdf` / `python-docx` /
`python-pptx` / `openpyxl` / `jieba` / `numpy` / `rank-bm25` / `PyYAML` / `httpx` /
`rapidocr` / `onnxruntime` / `opencv-python-headless`。

向量化后端：embed 端点 `:18790` → `POST /v1/embed/embeddings`
（`model=qwen3-embedding-0.6b-q4_0`，Qwen3-Embedding-0.6B Q4_0，1024 维，`--pooling mean`）。

OCR：`rapidocr` 自带 PP-OCRv6 det/cls/rec 三个模型（约 32MB，随 wheel 安装，**不联网下载**）。
OCR 引擎唯一：识别失败如实落成 `failed` 并带原因，**不换引擎静默重试**。

---

## 许可

Apache-2.0，见顶层 [`LICENSE`](LICENSE)；版权与打包声明见 [`debian/copyright`](debian/copyright)。
