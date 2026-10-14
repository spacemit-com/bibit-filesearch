# bibit-filesearch 文档索引

**先读这一页**——它说明哪份文档是权威、出问题该翻哪份。

## 📌 权威（与实现同步维护；改代码时同步改这里）

| 文档 | 内容 | 什么时候读 |
|---|---|---|
| [`api-contract.md`](api-contract.md) | 对外 HTTP 接口的**权威基准**：端点、请求/响应字段、错误码，以及 `score` 与 `confidence` 的语义差异 | 对接本服务、改接口、改返回字段 |
| [`configuration.md`](configuration.md) | **配置参考**：三层优先级、界面可改的键、`search.*` / `ocr.*` 调试旋钮、`candidate_pool` 与 `top_k` 的区别、配置版本与迁移 | 想改任何设置、想知道「该写在哪」 |
| [`design.md`](design.md) | 设计：产品定位、per-user 用户模型、配置架构、PDF 解析策略、检索打分融合、OCR 内存与释放、生命周期、部署 | 想知道「为什么这样设计」 |
| [`bibit-filesearch-intro.md`](bibit-filesearch-intro.md) | 实现全貌：服务组成、索引写路径、检索读路径、全链路时序、运维要点 | 新同学上手；想理解一次检索怎么走完 |
| [`troubleshooting.md`](troubleshooting.md) | **按现象分支的排查指南** + 排查渠道总览 + 版本对应速查 | 部署后出问题——先到这里找现象 |

## 📐 契约与设计记录（实现依据，改这些区域前必读）

| 文档 | 内容 | 状态 |
|---|---|---|
| [`scan-status-contract.md`](scan-status-contract.md) | 扫描/索引状态的**对外口径与状态机语义**（状态格优先级、三格计数、两阶段进度、计数不变式） | **现行契约** |
| [`adr/0001-single-index-pool.md`](adr/0001-single-index-pool.md) | **ADR**：索引状态收归单一池子、执行路径收敛为一条 | 已采纳 |
| [`bm25-score-calibration.md`](bm25-score-calibration.md) | BM25 得分标定（K 值）的可复现步骤与记录表 | 现役（`search.bm25_half_score=2.8` 的依据） |

## 约定

- **代码引用只写文件 / 函数 / 类名，不标行号**（行号随重构漂移，核对成本高于收益）。
- **一个主题只有一份权威文档**，其余位置只写指针。写重复的表格最容易两边一起过时。
- 新增文档后**回来登记本索引**。
