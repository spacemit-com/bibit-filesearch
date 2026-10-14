"""抬高块数上限后，存量被截断的文件必须真的重索引（2026-09-21 新增）。

背景：`content_hash` 是文件**字节**的 hash，截断不体现在里面。所以把
`scan.max_chunks_per_file` 从 32 调到 64 之后，重扫会判「内容未变 → skipped」，
那些文件永远停在 32 块上——**改了配置等于没改**。这与 0.1.9（purge_vectors 没清
hash）、0.1.10（requeue_unsupported）是同一类陷阱：派生数据的语义前提变了，而
幂等判据看不出来。

修法：`files.total_chunks` 记录截断前的真实块数 + `indexer.CHUNK_REVISION` 水位线，
启动时由 `settings.reconcile_chunk_revision` 触发一次精确重排。
"""
from bibit_filesearch.db import Database
from bibit_filesearch.indexer import CHUNK_REVISION, Indexer
from bibit_filesearch.settings import reconcile_chunk_revision

CHUNK = 10          # chunk_size：10 字符/块、无重叠 → 块数 == 字符数/10


class _FakeState:
    """scan_state 的最小替身（key-value）。"""

    def __init__(self):
        self.d = {}

    def get(self, k, default=None):
        return self.d.get(k, default)

    def set(self, k, v):
        self.d[k] = v


class _Embed:
    """确定性伪 embed（维度 4，内容无关）。"""

    dim = 4

    def embed_batch(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_one(self, text):
        return [0.1, 0.2, 0.3, 0.4]


def _db(tmp_path) -> Database:
    return Database(str(tmp_path / "c.db"))


def _txt(tmp_path, name: str, chars: int) -> str:
    """写一个 chars 字符的 .txt（内容无关，只需块数可控）。"""
    p = tmp_path / name
    p.write_text("x" * chars, encoding="utf-8")
    return str(p)


def _indexer(db, cap: int) -> Indexer:
    # 用 500/100 之外的参数让块数 == 字符数/10，断言好读
    return Indexer(db, _Embed(), chunk_size=CHUNK, chunk_overlap=0, max_chunks_per_file=cap)


def test_total_chunks_column_exists_on_old_db(tmp_path):
    """老库（无该列）打开后被补列，默认 0。"""
    db = _db(tmp_path)
    cols = [r[1] for r in db.conn.execute("PRAGMA table_info(files)").fetchall()]
    assert "total_chunks" in cols


def test_truncated_file_records_real_total(tmp_path):
    """被截断的文件：存 3 块，但 total_chunks 记下真实的 10 块。"""
    db = _db(tmp_path)
    idx = _indexer(db, cap=3)
    r = idx.index_file(_txt(tmp_path, "big.txt", 100))     # 10 块 → 截到 3
    assert r["chunks"] == 3
    assert r["total_chunks"] == 10
    assert r["oversized"] is True
    row = db.conn.execute("SELECT total_chunks FROM files").fetchone()
    assert row[0] == 10


def test_untruncated_file_total_equals_indexed(tmp_path):
    """未截断的文件 total_chunks == 已索引块数（不变式，供 requeue 判据使用）。"""
    db = _db(tmp_path)
    idx = _indexer(db, cap=100)
    r = idx.index_file(_txt(tmp_path, "small.txt", 50))    # 5 块，未截断
    assert r["chunks"] == 5
    assert r["total_chunks"] == 5
    assert r["oversized"] is False


def test_requeue_truncated_only_picks_truncated(tmp_path):
    """★ 核心：只挑「已存块数 < 真实块数」的文件，不误伤本来就短的文件。"""
    db = _db(tmp_path)
    idx = _indexer(db, cap=3)
    big_path = _txt(tmp_path, "big.txt", 100)      # 10 块 → 截到 3
    small_path = _txt(tmp_path, "small.txt", 30)   # 3 块 → 完整
    idx.index_file(big_path)
    idx.index_file(small_path)

    assert db.requeue_truncated() == 1
    big = db.get_file_by_path(big_path)
    small = db.get_file_by_path(small_path)
    # 被截断的：清 hash + 重排（否则幂等会 skipped）
    assert big["vectorize_status"] == "queued" and big["content_hash"] == ""
    # 完整的：原样不动
    assert small["vectorize_status"] == "completed" and small["content_hash"] != ""


def test_raising_cap_reindexes_the_truncated_file(tmp_path):
    """★ 端到端：上限 32→64 后，同一个文件重排出 64 块（修的是「改配置不生效」）。"""
    db = _db(tmp_path)
    path = _txt(tmp_path, "mid.txt", 640)          # 64 块

    r1 = _indexer(db, cap=32).index_file(path)
    assert r1["chunks"] == 32 and r1["total_chunks"] == 64

    # 只改上限重跑：没有重排机制的话，这里会因 content_hash 幂等直接 skipped
    same = _indexer(db, cap=64)
    assert same.index_file(path)["status"] == "skipped"     # ← 这正是 bug 的形态

    # 水位线触发重排后才真正重算
    assert db.requeue_truncated() == 1
    r2 = same.index_file(path)
    assert r2["status"] in ("indexed", "updated")
    assert r2["chunks"] == 64 and r2["oversized"] is False


def test_reconcile_runs_once_per_revision(tmp_path):
    """★ 水位线：只跑一次，之后即使文件仍处截断态也不重复重灌。"""
    db = _db(tmp_path)
    path = _txt(tmp_path, "big.txt", 100)
    _indexer(db, cap=3).index_file(path)
    state = _FakeState()

    assert reconcile_chunk_revision(db, state) == 1        # 第一次：重排
    assert state.get("chunk_revision") == CHUNK_REVISION
    # 再跑（模拟服务重启）：不再重排。文件仍是截断态（3 < 10），但水位线已到位——
    # 这是有意为之，不是漏判（否则每次启动都重灌一遍）。
    assert reconcile_chunk_revision(db, state) == 0


def test_reconcile_noop_when_already_at_revision(tmp_path):
    """水位线已到位 → 不碰数据库（每次启动零代价）。"""
    db = _db(tmp_path)
    path = _txt(tmp_path, "big.txt", 100)
    _indexer(db, cap=3).index_file(path)
    state = _FakeState()
    state.set("chunk_revision", CHUNK_REVISION)

    assert reconcile_chunk_revision(db, state) == 0
    assert db.get_file_by_path(path)["vectorize_status"] == "completed"   # 未被重排


def test_old_db_without_total_chunks_is_not_requeued(tmp_path):
    """精确判据（legacy_cap=0）下老库不参与重排，避免把整库误清一遍。"""
    db = _db(tmp_path)
    db.conn.execute(
        "INSERT INTO files (file_id, path, name, file_type, content_hash, vectorize_status)"
        " VALUES ('f1','/kb/old.md','old.md','md','deadbeef','completed')")
    db.conn.commit()
    assert db.requeue_truncated() == 0


# ---------------------------------------------------------------- 老库路径
def _legacy_row(db, path: str, n_chunks: int, status: str = "completed") -> str:
    """造一行「老库」记录：total_chunks=0（该列刚补上、值为未知），但 chunks 表里有数据。"""
    db.conn.execute(
        "INSERT INTO files (file_id, path, name, file_type, content_hash, vectorize_status)"
        " VALUES (?,?,?,'md','HASHED',?)",
        (path, path, path.rsplit("/", 1)[-1], status))
    fid = db.conn.execute("SELECT file_id FROM files WHERE path=?", (path,)).fetchone()[0]
    for i in range(n_chunks):
        db.conn.execute(
            "INSERT INTO chunks (file_id, chunk_index, text, vector) VALUES (?,?,?,?)",
            (fid, i, f"t{i}", b"\x00\x01"))
    db.conn.commit()
    return fid


def test_legacy_db_requeues_files_sitting_at_old_cap(tmp_path):
    """★ 老库关键路径：升级前入库的文件 total_chunks 未知，但块数顶在旧上限 32 上 →
    必须重排。否则老用户装了新版本，上限改了却一个文件都不动，而水位线本次就被
    置成已见、机会一去不回。"""
    db = _db(tmp_path)
    at_cap = _legacy_row(db, "/kb/truncated.md", 32)     # 旧上限 32 → 可能被截断
    below = _legacy_row(db, "/kb/complete.md", 5)        # 明确没被截断

    assert db.requeue_truncated(legacy_cap=32) == 1
    a = db.get_file_by_path("/kb/truncated.md")
    b = db.get_file_by_path("/kb/complete.md")
    assert a["vectorize_status"] == "queued" and a["content_hash"] == ""
    assert b["vectorize_status"] == "completed" and b["content_hash"] == "HASHED"
    assert at_cap != below


def test_legacy_requeue_is_off_by_default(tmp_path):
    """不传 legacy_cap 时老库判据不生效（只有水位线调用点显式开启）。"""
    db = _db(tmp_path)
    _legacy_row(db, "/kb/truncated.md", 32)
    assert db.requeue_truncated() == 0


def test_legacy_clause_ignores_rows_with_known_total(tmp_path):
    """已被新代码索引过的行（total_chunks > 0）不受老库判据影响：
    恰好 32 块且 total==32 的完整文件不会被误排。"""
    db = _db(tmp_path)
    path = _txt(tmp_path, "exactly32.txt", 32 * CHUNK)
    r = _indexer(db, cap=64).index_file(path)          # 新代码：total_chunks=32
    assert r["total_chunks"] == 32 and r["oversized"] is False

    assert db.requeue_truncated(legacy_cap=32) == 0
    assert db.get_file_by_path(path)["vectorize_status"] == "completed"


def test_reconcile_passes_legacy_cap(tmp_path):
    """水位线入口把 legacy_cap 透传下去（老库路径真的会走到）。"""
    db = _db(tmp_path)
    _legacy_row(db, "/kb/truncated.md", 32)
    state = _FakeState()

    assert reconcile_chunk_revision(db, state, legacy_cap=32) == 1
    assert state.get("chunk_revision") == CHUNK_REVISION


def test_reconcile_without_legacy_cap_skips_old_rows(tmp_path):
    """不传 legacy_cap：老库不动（默认行为，保证既有调用方语义不变）。"""
    db = _db(tmp_path)
    _legacy_row(db, "/kb/truncated.md", 32)
    assert reconcile_chunk_revision(db, _FakeState()) == 0
