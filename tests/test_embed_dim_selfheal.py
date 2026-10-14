"""向量维度的自愈：以服务为准 + 存量索引不可复用时清空重建。

背景（`docs/troubleshooting.md` 现象 G）：`models.embed.dim` 是唯一漏掉的
老化石键——0.1.1~0.1.3 出厂模板把它冻成 512，服务换成 Qwen3（1024 维）后：

* 配置 512 + 服务 1024 → 每次 `embed_batch` 撞维度防线抛错 → 检索全返 1004。
  界面只显示「检索失败」，而 `/health/deep` 探的是 `:8082/health`、不经过维度校验，
  所以**看不出任何异常**（这才是严重之处）；
* 只把配置键删掉、不管存量索引 → 变成「启动即报维度不符」，比原来更严重。

本文件钉三件事：维度以服务为准、旧向量在维度不符时被清空并重排队、配置里的化石被迁移删掉。
"""
from pathlib import Path

import yaml

from bibit_filesearch.db import Database, _pack_vec
from bibit_filesearch.settings import (
    CONFIG_VERSION,
    _CHANGED_DEFAULTS,
    SettingsService,
    reconcile_embed_dim,
)

# ─────────────────────────────────────────────────────────────
# 一、DB 侧：读存量维度 / 清空向量
# ─────────────────────────────────────────────────────────────

def _db(tmp_path) -> Database:
    return Database(str(tmp_path / "dim.db"))


def _add_file(db: Database, fid: str, path: str, status: str):
    db.conn.execute(
        "INSERT INTO files (file_id, path, name, file_type, size, modified_at, content_hash, "
        "vectorize_status) VALUES (?,?,?,?,?,?,?,?)",
        (fid, path, Path(path).name, path.rsplit(".", 1)[-1], 1, "2026-09-15T10:00:00",
         "hash-" + fid, status),
    )
    db.conn.commit()


def _add_vector(db: Database, fid: str, dim: int, idx: int = 0):
    db.conn.execute(
        "INSERT INTO chunks (file_id, chunk_index, text, vector) VALUES (?,?,?,?)",
        (fid, idx, "text", _pack_vec([0.1] * dim)),
    )
    db.conn.commit()


def test_stored_vector_dim_none_when_no_vectors(tmp_path):
    db = _db(tmp_path)
    _add_file(db, "f1", str(tmp_path / "a.md"), "completed")
    assert db.stored_vector_dim() is None, "无向量时应为 None（不是 0，0 会被当成有效维度）"


def test_stored_vector_dim_is_bytes_over_four(tmp_path):
    db = _db(tmp_path)
    _add_file(db, "f1", str(tmp_path / "a.md"), "completed")
    _add_vector(db, "f1", 512)
    assert db.stored_vector_dim() == 512
    _add_vector(db, "f1", 1024, idx=1)
    # 取任一条即可（同库内混杂属数据损坏，由 retriever 的整除校验兜）
    assert db.stored_vector_dim() in (512, 1024)


def test_purge_vectors_clears_chunks_and_requeues_processed_files(tmp_path):
    db = _db(tmp_path)
    for fid, status in (("done", "completed"), ("fail", "failed"),
                        ("unsup", "unsupported"), ("idle", "idle")):
        _add_file(db, fid, str(tmp_path / f"{fid}.md"), status)
        _add_vector(db, fid, 512)

    n = db.purge_vectors()

    assert db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0, "向量须全部清空"
    assert n == 2, f"只应重置 completed/failed 两个文件，实际 {n}"
    left = dict(db.conn.execute(
        "SELECT file_id, vectorize_status FROM files").fetchall())
    assert left["done"] == "queued" and left["fail"] == "queued", "已处理的须重新排队"
    assert left["unsup"] == "unsupported", "unsupported 不是「待向量化」，不该被重置"
    assert left["idle"] == "idle"


def test_purge_vectors_clears_content_hash(tmp_path):
    """清空向量时必须**同时**清空 content_hash。

    否则重扫时 index_file 看到 hash 未变 → 直接返回 skipped → 向量永远补不回来，
    worker 还会把状态改判成 unsupported，表现为「索引完成但搜不到」。
    这条测试守的就是这个坑：只清向量不清 hash，等于白清。
    """
    db = _db(tmp_path)
    _add_file(db, "f1", str(tmp_path / "a.md"), "completed")
    _add_vector(db, "f1", 512)
    assert db.get_file("f1")["content_hash"] == "hash-f1", "前置：播种时 hash 非空"

    db.purge_vectors()

    assert db.get_file("f1")["content_hash"] == "", \
        "hash 未清 → 重扫会被 hash-skip 跳过，索引永远建不回来"


# ─────────────────────────────────────────────────────────────
# 二、探测：维度以服务为准
# ─────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _StubClient:
    def __init__(self, payload=None, exc=None):
        self.payload, self.exc = payload, exc

    def post(self, url, json=None, timeout=None):
        if self.exc:
            raise self.exc
        return _Resp(self.payload)


def _embed_with(payload=None, exc=None):
    from bibit_filesearch.model_client import EmbedClient

    c = EmbedClient("http://127.0.0.1:9", dim=512, model="m", query_instruction="")
    c._client = _StubClient(payload, exc)
    return c


def test_probe_dim_returns_service_dim_even_when_config_disagrees():
    """探测必须返回**服务**的维度——它存在的意义就是发现配置与服务不符。"""
    c = _embed_with({"data": [{"embedding": [0.0] * 1024}]})
    assert c.probe_dim() == 1024, "配置写 512 也要如实返回服务实际维度"


def test_probe_dim_returns_none_on_error():
    c = _embed_with(exc=RuntimeError("connection refused"))
    assert c.probe_dim() is None, "探测失败返回 None（让调用方退回配置值），不得抛"


def test_probe_dim_returns_none_on_empty_payload():
    assert _embed_with({"data": []}).probe_dim() is None
    assert _embed_with({}).probe_dim() is None
    assert _embed_with({"data": [{}]}).probe_dim() is None


# ─────────────────────────────────────────────────────────────
# 三、启动决策：生效维度 + 清空不可复用的旧索引
# ─────────────────────────────────────────────────────────────

def test_reconcile_probed_dim_wins_over_fossil_config(tmp_path):
    """主场景①：配置是化石 512、服务是 1024、存量索引 512 → 按 1024 生效并清空旧索引。"""
    db = _db(tmp_path)
    _add_file(db, "f1", str(tmp_path / "a.md"), "completed")
    _add_vector(db, "f1", 512)

    effective, purged = reconcile_embed_dim(db, configured_dim=512, probed_dim=1024)

    assert effective == 1024, "以服务为准"
    assert purged == 1
    assert db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_reconcile_keeps_index_when_dim_matches(tmp_path):
    """维度一致（正常用户）时**不得**动索引——一次误清就是全库重建。"""
    db = _db(tmp_path)
    _add_file(db, "f1", str(tmp_path / "a.md"), "completed")
    _add_vector(db, "f1", 1024)

    effective, purged = reconcile_embed_dim(db, configured_dim=1024, probed_dim=1024)

    assert (effective, purged) == (1024, 0)
    assert db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1, "索引须原封不动"


def test_reconcile_falls_back_to_config_when_probe_failed(tmp_path):
    """服务未起时不能凭猜测改行为：退回配置值，且不动索引。"""
    db = _db(tmp_path)
    _add_file(db, "f1", str(tmp_path / "a.md"), "completed")
    _add_vector(db, "f1", 512)

    effective, purged = reconcile_embed_dim(db, configured_dim=512, probed_dim=None)

    assert (effective, purged) == (512, 0), "探不到 → 配置为准且不清索引"


def test_reconcile_purges_when_only_config_changed(tmp_path):
    """场景③：配置已改对（1024）但存量索引还是旧的 512 → 必须清，否则启动即报维度不符。"""
    db = _db(tmp_path)
    _add_file(db, "f1", str(tmp_path / "a.md"), "completed")
    _add_vector(db, "f1", 512)

    effective, purged = reconcile_embed_dim(db, configured_dim=1024, probed_dim=1024)

    assert (effective, purged) == (1024, 1)


def test_reconcile_no_index_is_noop(tmp_path):
    """空库首次启动：无向量可清，不能报错。（purge 也返回 0）"""
    db = _db(tmp_path)
    effective, purged = reconcile_embed_dim(db, configured_dim=1024, probed_dim=1024)
    assert (effective, purged) == (1024, 0)


# ─────────────────────────────────────────────────────────────
# 四、配置迁移：化石 dim 必须被删掉
# ─────────────────────────────────────────────────────────────

def _migrate(tmp_path, cfg: dict) -> dict:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    SettingsService(config_path=p)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def test_migration_strips_dim_fossil(tmp_path):
    """老用户配置里的 dim: 512（== 老出厂默认）须被迁移删除。"""
    out = _migrate(tmp_path, {
        "config_version": 2,
        "index_dirs": ["/tmp/kb"],
        "models": {"embed": {"dim": 512, "url": "http://127.0.0.1:8082"}},
    })
    assert 512 not in [v for v in _flatten(out).values()], \
        f"dim 化石仍在配置里：{out}"
    assert (out.get("models") or {}).get("embed", {}).get("dim") is None, "dim 键应被删"
    assert out["config_version"] == CONFIG_VERSION


def test_migration_keeps_user_chosen_dim(tmp_path):
    """用户为自己的 embed 服务显式声明的维度（≠ 老默认）不得被删。"""
    out = _migrate(tmp_path, {
        "config_version": 2,
        "models": {"embed": {"dim": 768}},
    })
    assert (out.get("models") or {}).get("embed", {}).get("dim") == 768, \
        "非化石的 dim 是用户选择，保留（真正的正确性由探测保证）"


def test_dim_fossil_registered_in_changed_defaults():
    """注册表守卫：dim 化石登记项与配置版本号被改动时报警。"""
    assert _CHANGED_DEFAULTS.get("models.embed.dim") == 512, \
        "dim 的老出厂默认（bge 512 维）未登记 → 老用户升级后检索会全返 1004"
    assert CONFIG_VERSION >= 3, "登记新化石必须同时升版本号，否则已迁到旧版本的用户不会重跑迁移"


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out
