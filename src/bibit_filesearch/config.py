"""配置加载：config.yaml（2-tier：文件 + 默认值）。"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("bibit.config")

# 出厂 CORS 白名单（GUI tauri webview + 本地调试来源）。
# 2026-09-07 教训：此前 DEFAULTS 里是空列表，而 /opt 出厂模板有 5 条——用户
# 删掉 ~/.config 副本后 load_config 只读 DEFAULTS（不读模板）→ 白名单为空
# → GUI 跨域「服务不可用」。模板与代码默认值必须同源，否则删配置就炸。
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "tauri://localhost",
    "http://tauri.localhost",
    "http://localhost",
]

DEFAULTS = {
    "server": {"host": "127.0.0.1", "port": 8050, "cors_origins": list(CORS_ORIGINS)},
    "index_dirs": [],
    "exclude_patterns": [".git", "node_modules", "__pycache__"],
    # models.embed：embed 服务连接 + 模型语义，全部配置化（2026-09-03 解耦，切换模型不改代码）：
    #   model              —— 请求体 "model" 字段透传（服务端不校验，仅需与运行服务一致）
    #   dim                —— 向量维度；与存量索引实际维度不符时启动即报错，提示清索引重索引
    #   query_instruction  —— 查询指令前缀（instruction-based 模型官方模板
    #                          "Instruct: <任务描述>\nQuery: <query>"）；留空 = 裸 query。
    #                          2026-09-09 默认改为空：:8082 用 --pooling mean，指令前缀会
    #                          淹没用短查询（见 packaging/config.yaml 注释），且索引侧
    #                          embed_batch 从不加前缀，两侧不对称。仅当 embed 服务改用
    #                          last-token pooling 时才考虑重新开启。
    "models": {
        # 2026-09-20：后端从「直连推理进程」改为统一走一个 OpenAI 兼容端点
        # （模型由该端点统一加载/卸载、端口动态分配）。两者的差异只在下面几个键，代码里没有分支：
        #   url / path  —— 推理入口（默认 :18790 + /v1/embed/embeddings）
        #   model       —— 端点要求显式传模型名 "qwen3-embedding-0.6b-q4_0"
        #   load_path   —— 端点提供的模型加载接口；模型未加载时推理返 503，
        #                  客户端会先调它再重试（直连模式留空 = 不适用）
        # 若要退回直连推理进程：url 改回 http://127.0.0.1:8082，
        # path=/v1/embeddings，model 填那个进程实际加载的模型名，load_path 留空。
        "embed": {"url": "http://127.0.0.1:18790", "dim": 1024, "model": "qwen3-embedding-0.6b-q4_0",
                  "query_instruction": "",
                  "path": "/v1/embed/embeddings",
                  # 端点契约要求显式声明返回格式为 float（响应形状同 OpenAI：
                  # data[i].embedding）。直连模式时该字段留空不发送。
                  "encoding_format": "float",
                  "load_path": "/v1/embed/models/load",
                  # 模型加载可能真的耗时百来秒（读 GGUF + 建 KV），与业务超时分开口径
                  "load_timeout": 150,
                  # 2026-09-18：单请求最大输入数。原先把整个文件的全部块塞进一次请求，
                  # 实测一个 24.8MB 的 minified JS = 83,782 块 = 一个请求携带 83,782 条
                  # 输入，直接把 embed 服务内存顶起来。32 与 embed 服务 4 个 slot 匹配，
                  # 单请求体积有上界、失败重试也只重发一小批（但不降低总内存）。
                  "batch_size": 32,
                  # 2026-09-18：真实探活超时（秒）。必须远小于业务超时 timeout——
                  # 探活要快速失败，否则看门狗一个节拍就被拖住。
                  "probe_timeout": 5,
                  # 2026-09-20：业务请求超时（秒）。它是「**一批** 32 条算完的预算」，
                  # 不是「多久没响应算卡死」——该接口非流式，
                  # 算完整批才返回。实测单批 32 块：markdown 45.8s、压缩 JS 124~150s
                  # （--cache-ram 0），所以没有一个常数能覆盖所有内容。
                  # 因此它不再需要覆盖最坏内容：超时后客户端会把该批**队列二分**
                  # （32→16→8→4）再试，只要有一个规模能在 60s 内算完就能完成索引。
                  "timeout": 60,
                  # 2026-09-20：连接超时（秒），与 read 分开。本机回环，服务不在听时
                  # 应秒级失败，不该白等整个 timeout。
                  "connect_timeout": 5},
    },
    "chunk": {"size": 500, "overlap": 100},
    # 顶层键（与 yaml config 对齐；config.py 顶层另有 index_dirs/top_k/auto_index/
    # indexable_types 等运行时字段——此前缺省由 settings_svc.get_all() 在 main.py
    # 覆盖，纯 load_config 无 env 且无 yaml 时可能缺 → 补齐默认，避免裸跑缺字段）
    "top_k": 5,
    "auto_index": True,
    "indexable_types": ["txt", "md", "docx", "pdf", "py", "js", "java", "cpp", "c", "cs",
                        "go", "rs", "ts", "sql", "sh", "json", "xlsx", "pptx"],
    # search.bm25_weight：稀疏关键词路权重（0 = 关闭）。D-BM25 设计 7:3 融合，
    # retriever/测试/文档均按 0.3 就绪；2026-09-04 起默认开启——技术词/英文 token
    # （gsettings、wl_surface 等）靠 BM25 一击必中，纯语义路对稀有词召回不足。
    # search.candidate_pool：检索深度——最多取相似度最高的多少个**块**参与文件级
    # 聚合（BM25 路按**文件**数同名处理）。**与「返回条数」top_k 无关**：top_k 只管
    # 显示几条，本值管找多深。
    #
    # 2026-09-11 曾误把两者当同一概念、让池取用户的 top_k（进了已推的 0.1.8），后果：
    # 把 Top-K 从 10 调到 5，块级窗口从「前 10 块」缩到「前 5 块」，某文件的最佳块
    # 排第 8 就被切掉、它的分数凭空消失（实测同一文件同一查询 score 0.6168 → 0.30），
    # 表现为「改显示条数 → 排序变了」。故本值独立于 top_k，且不进界面——它是调试
    # 旋钮，用户不知道该填几：填小重现上述 bug，填大只是变慢。
    #
    # 取值：0 或负数 = 不截断（全部候选块参与）。默认 1000 的推导——界面 top_k 上限
    # 100、每文件平均约 6 块，保证「显示 100 条时每条的块都在池内」需约 600，留余量。
    # 成本：1 万块 ≈0.2ms、10 万块（max_chunks 上限）≈1.6ms。
    # 调试改法：编辑 ~/.config/bibit-filesearch/config.yaml 的 search.candidate_pool
    # 后重启服务。界面保存设置**不会**抹掉它（只写它自己那几个键，其余原样保留）。
    # search.bm25_half_score：BM25 原始分 → 0~1 的映射常数 K（S = raw/(raw+K)）。
    #   2026-09-29 新增，**替换掉原来的批内 min-max 归一化**。min-max 把"此次命中集"
    #   的最小分映 0、最大分映 1，于是命中集小且分数接近时会把"都命中"拉成"一个满分
    #   一个零分"。实测（真机 query「身份」）：id_card_1 原始分 2.3033、id_rotate 2.3020
    #   （同一张身份证的两个方向版本，OCR 文本几乎相同），min-max 后是 1.0 与 0.0，
    #   最终 score 0.3920 vs 0.1145 —— 用户看到"一张第一、一张最后"。
    #   绝对映射下两者都是 ≈0.500，并列。
    #   K 的取值：2026-09-29 **判据法**标定 2.8（同日三次重标：分词从精确模式 →
    #     双字兜底 → 搜索模式 cut_for_search，每次 token 规模变化都必须重标）。
    #     判据法：让两个可人工验证的查询（「身份」三张身份证都要排在无关文档前、
    #     「身份证上的地址」id_card_2 必须第一）的**最差相对裕度**最大化。
    #     "取好命中原始分的中位数"的旧规则在原始分跨度大时不成立（曾给出 9.66
    #     并把相关文件排到无关文档之后，已否）。标定全过程含被否取值与原因，
    #     见 docs/bm25-score-calibration.md。
    #   ★ 改 chunk.size / max_chunks_per_file / 换语料后要重标（原始分随 avg_len 与 IDF 漂移）。
    #     0 = 退化为"命中即 1"的指示函数。
    # search.embed_weight / filename_weight / bm25_weight：三路融合的**真实比例**
    #   （和 = 1，见 retriever 的加权平均；代码仍除以 ΣW，故历史值 0.7/0.3/0.3 也不会变分）。
    #   2026-09-29 由 0.7 / 0.3 / 0.3 改为此值：那组数字的和是 **1.3**，除下来实际比例是
    #   0.538462 / 0.230769 / 0.230769 —— 写 0.7 会让人以为 cos 占七成，实际 53.8%。
    #   现在配置里写的数就是真实比例（保留 6 位小数：和恰为 1，与旧值系数偏差 <5e-7，
    #   远低于 score 显示的 4 位小数）。历史意图「语义 ≈54% / 词法 ≈46%」，此值即照此固化。
    #   （P4，见 docs/bm25-score-calibration.md 的"未做/已做"记录）
    "search": {"filename_weight": 0.230769, "embed_weight": 0.538462,
               "min_similarity": 0.35, "bm25_weight": 0.230769, "candidate_pool": 1000,
               "bm25_half_score": 2.8},
    # scan.max_chunks_per_file：单文件最多索引多少块（0 = 不限）。超限只索引前 N 块，
    #   并在失败明细里记一条 kind=oversized（扫描期间可在 /api/scan/progress 的
    #   errorDetails 看到，另落 data/scan_errors.log）。
    #   默认 64。**旧依据已作废**：设为 32 时的推导是「每 600 字符稠密内容占 embed 服务
    #   55~65MB 且不归还（~112KB/token），故 32 块 ≈ 1.9GB」——那是 embed 用**默认
    #   cache-ram（8192）**时的数。2026-09-21 实测（`--cache-ram 0` 已进网关
    #   default_args，见 06-给上游的embed参数反馈.md）：
    #     · 256 个全新块走生产 embed_batch → RSS **−1MB**（边际成本≈0，8 批全 +0）；
    #     · 同一 286KB markdown 全量 599 块 → RSS **+13MB**、耗时 521s（-t 4）。
    #   即内存几乎与块数无关（一次性台阶 + 平台），真正的约束是**时间**。
    #   64 的选择：当前索引范围内真实文档最多 31 块（仍零截断），翻倍后能覆盖
    #   「中等偏大」的文档；单文件最坏耗时约 64/32×46s≈1.5min（-t 4）。仍设上限而非
    #   不限，是为了给「异常文件」留下界——但下界由 oversized 记录兜底，不再假装省内存。
    #   只约束单文件冲击，不约束总量。
    # scan.max_attempts：单文件最多重试几次；达上限即停止重排（防「每轮全量重灌」）。
    # scan.min_embed_chars：短于此**字符数**的块不做向量化（0 = 关闭）。2026-09-21 新增。
    #   为什么按字符而不按 token：实测同一 tokenizer 下连续 ASCII 词是 1 token
    #   （'flutter'/'test'/'README'），数字串是逐字符（'123'→3、'12345'→5），两类无法
    #   用同一条字符启发式兼顾，任何估算式都必然在一边误判。字符数是最保守且跨模型稳定的判据。
    #   默认 3：挡住最脏的一档（单字符块，实测 24 个查询里 1~2 token 查询 12/12 由它占第一）。
    #   若要连 'liuyingjie'（10 字符 4 token）这类短标识符也挡在语义路外，调到 21 以上——
    #   但那样文件只在 BM25 里命中，且已有的短块需清 scan_state 的 chunk_revision 触发重排。
    # scan.embed_recovery_cooldown：两次自动重建（看门狗触发）之间的最小间隔秒数，
    #   用于打断「embed 被灌爆 OOM → 重启 → 立刻再灌一轮」的正反馈。
    # scan.ocr_preprocess（2026-09-24 新增，rev 5 已删）：它的作用是「跳过 tesseract 的
    #   方向补救」。rev 5 移除 tesseract 与那套补救后，这个开关没有作用对象了，故删键。
    "scan": {"vectorize_parallelism": 1, "io_throttle_ms": 2, "max_depth": 3,
             "max_chunks_per_file": 64, "max_attempts": 3,
             "min_embed_chars": 3,
             "embed_recovery_cooldown": 300},
    # OCR（rev 5，2026-09-28 引擎唯一化后可配的只剩三个旋钮）：
    #   device —— auto / npu / cpu。**设备**级选择，不是引擎选择：换的是算力不是识别器，
    #     文本不变（实测 CPU 与 NPU 四张真图逐字一致）。auto = 有 ep_worker 与
    #     libspacemit_ep.so 就用 NPU（实测快 1.7~2.9×，均值 2.26×），否则纯 CPU。
    #   max_side_len —— rapidocr 的 Global.max_side_len（出厂 1400，2026-09-28 由 2000 改）。
    #     内存代价近平方级，见 ocr_engine.MAX_SIDE_LEN 的逐档实测表（4 图 × 8 档）：
    #     2000→4916MB / 1600→3438 / 1400→2983 / 1200→2648 触底，再往下只损文本。
    #     <=0 表示不覆盖。★ 改它必须配 PARSER_REVISION +1：存量图片文本会变。
    "ocr": {"device": "auto",
            "max_side_len": 1400, "worker_max_runs": 0,
            # worker_max_rss_mb —— 单个 ep_worker 的 RSS 上限（MB，0 = 不限制）。
            # 超了即退休，下次 OCR 时重建。实测依据见 ocr_ep.WORKER_MAX_RSS_MB。
            "worker_max_rss_mb": 0,
            # idle_release_secs —— 空闲多久放掉全部 OCR worker（秒，0 = 关，出厂 10）。
            # 运行期**唯一**的释放触发点（2026-10-10 方案 A 收敛，见 pool._settle 的说明）：
            # 原「索引池排空即释放」那条已删 —— 池空不等于「不会再用了」，实测会让零星
            # 到达的相邻图每张都重建一遍 worker。内存真贵时的及时归还有 worker_max_rss_mb
            # 兜。定价：释放后第一张 OCR 多付 1.5s（实测 3.56s vs 2.06s），所以别设太小。
            # 见 ocr_ep.WORKER_IDLE_RELEASE_SECS。
            "idle_release_secs": 10,
            # pdf_max_ocr_pages —— 单个 PDF 最多 OCR 多少页（0 = 不限，出厂 20）。
            # 2026-09-29：PDF 改为逐页判据后，"整本扫描书"从"不 OCR"变成"逐页 OCR"。
            # 实测单页 5.2~7.5s，不设闸就是 43~64 分钟的 NPU 占用。
            # 正常书根本碰不到这个闸（预算早停通常 10~40 页内就攒够 64 块）；
            # 它只在病态输入（大量无文字图页要一直往后翻）上生效。
            # 见 parser.PDF_MAX_OCR_PAGES 的标定说明。
            "pdf_max_ocr_pages": 20},
    "storage": {"db_path": "data/filesearch.db"},
    "logging": {"level": "INFO", "file": ""},
    "nl": {"enabled": True},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 覆盖 base。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """点号访问的配置对象（嵌套 dict 自动包装）。

    2026-08-26：__getattr__ 拼错字段时给出相近字段提示（原 AttributeError 不直观，
    且 IDE/静态检查对动态属性失效——这是权衡，文档化使用约定）。
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            v = self._data[name]
            return Config(v) if isinstance(v, dict) else v
        # 相近字段提示（拼写错误快速定位）
        close = [k for k in self._data if k.startswith(name[:2]) or name.startswith(k[:2])]
        hint = f"，相近字段: {close}" if close else ""
        raise AttributeError(f"config 无字段: {name}{hint}")

    def __getitem__(self, key: str) -> Any:
        v = self._data[key]
        return Config(v) if isinstance(v, dict) else v

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            v = self._data[key]
            # 2026-08-26：显式 null 值视为不存在（返回 default）——避免调用方拿到 None
            # 后 bool(None)/if None 判断不一致
            if v is None:
                return default
            return Config(v) if isinstance(v, dict) else v
        return default

    def set(self, key: str, value: Any) -> None:
        """设置字段值（2026-08-26：settings API 运行时更新 config 用，替代直接改 _data）。"""
        self._data[key] = value

    def to_dict(self) -> dict:
        return self._data


def load_config(config_path: str | None = None) -> Config:
    """加载配置：默认值 <- config.yaml（若存在）。

    2026-08-26：候选路径不再硬编码 __file__ 推导——优先环境变量 BIBIT_CONFIG，
    其次显式传参，最后默认 backend/config.yaml。
    2026-09-07：默认路径补上用户层——启动不再需要显式 env 也能读到
    ~/.config/bibit-filesearch/config.yaml（GUI/设置保存的配置单一来源）。此前
    systemd 自启（无 BIBIT_CONFIG env）落到包内默认（/opt/.../src/config.yaml
    不存在）→ 内置默认 dim=512/cors 空 → 启动即错。
    优先级：BIBIT_CONFIG env > ~/.config/bibit-filesearch/config.yaml > 包内默认。
    """
    data = _deep_merge(DEFAULTS, {})
    if config_path is None:
        # 环境变量 > 用户配置层（~/.config） > 包内默认（源码旁 config.yaml）
        config_path = (
            os.environ.get("BIBIT_CONFIG")
            or str(Path.home() / ".config/bibit-filesearch/config.yaml")
            or str(Path(__file__).resolve().parent.parent / "config.yaml")
        )
    p = Path(config_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        data = _deep_merge(data, file_cfg)
        logger.info("加载配置: %s", p)
    else:
        logger.warning("未找到 config.yaml（%s），使用内置默认值", p)
    return Config(data)
