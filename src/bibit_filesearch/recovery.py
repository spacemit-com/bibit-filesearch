"""embed 恢复看门狗（2026-09-09）。

问题：embed 后端（默认 127.0.0.1:8082）掉线时，
索引与检索全部失败，文件被标 failed；embed 恢复后**没有任何触发点**重跑——
watcher 只监听文件变化事件，POST /api/settings 在配置未变时幂等早退，
用户不打开 GUI 就永远停在 failed。

方案：周期性探测 embed 的 /health，检测到 **down → up 跳变** 且库里存在
failed/queued 文件时，重启扫描（hash 跳过已完成文件，只重算失败/待处理项）。

只在跳变时触发（不是每次探测都触发）：永久解析失败的文件不会被无限重扫，
只会在 embed 真正重启一次后重试一次。扫描进行中/暂停时跳过，不打断用户操作。
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("bibit.recovery")

# 与 scanner 状态机对齐（不 import scanner，避免循环依赖）
_STATUS_SCANNING = "scanning"
_STATUS_PAUSED = "paused"


class EmbedRecovery:
    """embed 掉线→恢复后自动重建索引。"""

    def __init__(self, embed, db, scanner, interval: float = 5.0,
                 max_attempts: int = 3, cooldown: float = 300.0):
        self.embed = embed
        self.db = db
        self.scanner = scanner
        # 下限 0.05s：仅测试需要短间隔；生产 5s。
        # 2026-09-09：生产间隔 30s → 5s。实测「start embed 后 GUI 归零」要等 25s，
        # 其中 16s 是白等下一次探测节拍（探测是固定节拍，与 embed 何时就绪无关；
        # 恢复判定只在 down→up 跳变发生的那一次探测里做）。5s 把这段抖动压到 ≤5s，
        # 代价是每 5s 一次回环 /health GET（开销可忽略）。
        # 原先写死 max(1.0) 让 interval=0.2 的测试退化成 1s，与 sleep(1.0) 竞态
        # → 变异测试失效，故下限保持 0.05。
        self.interval = max(0.05, float(interval))
        # 2026-09-18：只把「还有重试预算」的文件算作待重跑。达上限的永久失败文件
        # 不再参与，避免每次 embed 重启都触发一轮注定失败的全量重扫。
        self.max_attempts = int(max_attempts or 0) or 3
        # 2026-09-18：两次自动重建之间的最小间隔（秒）。重建 = 一次全盘扫描，
        # 而 embed 崩溃的常见原因正是被这一波灌爆（OOM）——不加冷却会形成
        # 「OOM → systemd 拉起 → 看门狗探到 up → 再灌一轮 → 再 OOM」正反馈。
        self.cooldown = max(0.0, float(cooldown))
        self._last_rebuild = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_ok: bool | None = None
        self.probes = 0  # 探测次数（测试用来确定性等待，避免 sleep 竞态）

    # ---------- 生命周期 ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # 首次探测同步执行：确立基线（否则第一轮 prev=None 会把"一直是 up"
        # 误当成跳变，或像测试里那样永远等不到跳变）。
        self._prev_ok = self._probe()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="embed-recovery")
        self._thread.start()
        logger.info("embed 恢复看门狗启动（间隔 %.0fs，当前 %s）",
                    self.interval, "ok" if self._prev_ok else "unreachable")

    def stop(self):
        self._stop.set()

    # ---------- 内部 ----------
    def _probe(self) -> bool:
        try:
            # 2026-09-21：与 /health/deep 用**同一**判定——忙时用被动信号自证，
            # 闲时才真发探针。否则扫描期间探针排在批次队尾会假超时，看门狗把
            # 好后端判成 down、下一拍又判成 up，凭空造出一次 down→up 跳变
            # （跳变正是触发自动重建的条件）。
            # getattr 兜底：测试桩常只实现 ping()，缺 is_healthy 时退回真探针，
            # 不因探针接口不全而把桩误判成 down。
            healthy = getattr(self.embed, "is_healthy", None)
            if callable(healthy) and healthy():
                return True
            return bool(self.embed.ping())
        except Exception:  # noqa: BLE001
            return False

    def _pending_count(self) -> int:
        """还剩重试预算的 failed + queued 文件数（需要重跑的）。

        2026-09-18：加上 `attempt_count < max_attempts` 条件。此前把永久失败的
        文件也算作「待重跑」，配合「embed 反复崩溃重启」的现实，等于每次重启都
        触发一轮注定的全量重扫（这正是 OOM 正反馈的一环）。
        """
        try:
            row = self.db.conn.execute(
                "SELECT COUNT(*) FROM files WHERE vectorize_status IN ('failed','queued')"
                " AND attempt_count < ?",
                (self.max_attempts,),
            ).fetchone()
            return int(row[0]) if row else 0
        except Exception:  # noqa: BLE001
            logger.warning("查询待重跑文件数失败", exc_info=True)
            return 0

    def _loop(self):
        while not self._stop.wait(self.interval):
            ok = self._probe()
            recovered = self._prev_ok is False and ok is True
            self._prev_ok = ok
            self.probes += 1
            if not recovered:
                continue
            pending = self._pending_count()
            if pending <= 0:
                logger.info("embed 已恢复，无失败/待处理文件，跳过重建")
                continue
            status = getattr(self.scanner, "status", "idle")
            if status in (_STATUS_SCANNING, _STATUS_PAUSED):
                logger.info("embed 已恢复，但扫描状态为 %s，跳过自动重建", status)
                continue
            # 2026-09-18：冷却——两次自动重建之间至少间隔 cooldown 秒。
            # 不设冷却时，embed 因被灌爆而反复 OOM/重启，每次重启都会立刻触发
            # 新的一轮全量重扫，形成自我加速的闭环。冷却给系统留出消化时间。
            now = time.monotonic()
            waited = now - self._last_rebuild
            if self.cooldown > 0 and waited < self.cooldown:
                logger.warning(
                    "embed 已恢复，但距上次自动重建仅 %.0fs（冷却 %.0fs），跳过本轮"
                    "（待重跑 %d 个文件）",
                    waited, self.cooldown, pending,
                )
                continue
            logger.info("embed 已恢复，自动重建索引（待重跑 %d 个文件）", pending)
            try:
                self.scanner.start()
                self._last_rebuild = time.monotonic()
            except Exception:  # noqa: BLE001
                logger.exception("自动重建索引失败")
