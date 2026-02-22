"""
LotusAI Heartbeat Motoru
Versiyon: 1.0.0
Açıklama: OpenClaw tarzı proaktif arka plan görev yöneticisi.
           Sistem, kullanıcı komutu beklemeden arka planda yaşar ve
           kritik olayları kendiliğinden bildirir.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional

logger = logging.getLogger("LotusAI.Heartbeat")


# ═══════════════════════════════════════════════════════════════
# GÖREV TANIMI
# ═══════════════════════════════════════════════════════════════
@dataclass
class HeartbeatTask:
    """Tek bir heartbeat görevinin tanımı"""
    name: str
    interval: float           # Saniye cinsinden çalışma aralığı
    handler: Callable         # async def handler() -> Optional[str]
    agent: str = "ATLAS"      # Bildirimi hangi ajan seslendirir?
    priority: int = 5         # 1=en kritik, 10=en düşük
    enabled: bool = True
    # Çalışma istatistikleri
    last_run: Optional[datetime] = field(default=None, repr=False)
    run_count: int = field(default=0, repr=False)
    error_count: int = field(default=0, repr=False)


# ═══════════════════════════════════════════════════════════════
# HEARTBEAT MOTORU
# ═══════════════════════════════════════════════════════════════
class HeartbeatEngine:
    """
    Proaktif heartbeat motoru.

    Sistemi kullanıcı komutunu beklemeksizin arka planda yaşatır.
    Her kayıtlı görev kendi aralığında bağımsız olarak çalışır.
    Görev bir bildirim metni döndürürse, kayıtlı `on_notify`
    callback'i aracılığıyla seslendirme sağlanır.

    Kullanım:
        engine = HeartbeatEngine(on_notify=my_async_fn)
        engine.register("sistem_kontrol", interval=300, handler=fn)
        await engine.start()   # asyncio.create_task ile çağırın
        engine.stop()          # Kapatma sırasında
    """

    def __init__(self, on_notify: Optional[Callable] = None) -> None:
        """
        Args:
            on_notify: async def on_notify(message: str, agent: str) -> None
                       Heartbeat bildirimi geldiğinde çağrılır.
        """
        self._tasks: List[HeartbeatTask] = []
        self._running: bool = False
        self._on_notify = on_notify

    # ───────────────────────────────────────────────────────────
    # GÖREV KAYDETTIRME
    # ───────────────────────────────────────────────────────────
    def register(
        self,
        name: str,
        interval: float,
        handler: Callable,
        agent: str = "ATLAS",
        priority: int = 5,
    ) -> None:
        """
        Yeni bir heartbeat görevi kaydet.

        Args:
            name:     Görevin benzersiz adı
            interval: Tekrarlama aralığı (saniye)
            handler:  async callable — Optional[str] döndürür
            agent:    Bildirimi seslendirecek ajan adı
            priority: Öncelik (1=kritik, 10=düşük)
        """
        task = HeartbeatTask(
            name=name,
            interval=interval,
            handler=handler,
            agent=agent,
            priority=priority,
        )
        self._tasks.append(task)
        logger.info(
            f"[💓 HEARTBEAT] Görev kayıt edildi → {name} "
            f"| {interval}s aralık | Ajan: {agent}"
        )

    def register_skill(self, skill: Any) -> None:
        """
        Bir BaseSkill nesnesinden otomatik heartbeat görevi oluştur.
        Skill'in `heartbeat_interval > 0` olması gerekir.
        """
        if not (hasattr(skill, "heartbeat_interval") and skill.heartbeat_interval > 0):
            return

        self.register(
            name=skill.name,
            interval=skill.heartbeat_interval,
            handler=skill.on_heartbeat,
            agent=getattr(skill, "agent", "ATLAS"),
            priority=getattr(skill, "priority", 5),
        )

    # ───────────────────────────────────────────────────────────
    # BAŞLAT / DURDUR
    # ───────────────────────────────────────────────────────────
    async def start(self) -> None:
        """
        Tüm etkin görevleri paralel olarak başlat.
        Bu coroutine, `stop()` çağrılana kadar çalışmaya devam eder.
        """
        active = [t for t in self._tasks if t.enabled]

        if not active:
            logger.warning("[💓 HEARTBEAT] Kayıtlı görev yok, başlatılmıyor")
            return

        self._running = True
        sorted_tasks = sorted(active, key=lambda t: t.priority)
        logger.info(
            f"[💓 HEARTBEAT] Motor başlatıldı — {len(sorted_tasks)} görev aktif"
        )

        await asyncio.gather(
            *[self._run_task(task) for task in sorted_tasks],
            return_exceptions=True,
        )

    def stop(self) -> None:
        """Heartbeat motorunu durdur."""
        self._running = False
        logger.info("[💓 HEARTBEAT] Motor durduruldu")

    # ───────────────────────────────────────────────────────────
    # İÇ ÇALIŞMA DÖNGÜSÜ
    # ───────────────────────────────────────────────────────────
    async def _run_task(self, task: HeartbeatTask) -> None:
        """Tek bir görevi aralıklı olarak çalıştır."""
        # İlk çalışmayı hemen yapmak yerine aralık kadar bekle
        await asyncio.sleep(task.interval)

        while self._running:
            try:
                task.last_run = datetime.now()
                task.run_count += 1

                result: Optional[str] = await task.handler()

                if result and self._on_notify:
                    await self._on_notify(result, task.agent)
                    logger.info(
                        f"[💓 HEARTBEAT] {task.name} → bildirim gönderildi (#{task.run_count})"
                    )
                else:
                    logger.debug(
                        f"[💓 HEARTBEAT] {task.name} ✓ (#{task.run_count})"
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                task.error_count += 1
                logger.error(
                    f"[💓 HEARTBEAT] {task.name} hata: {exc} "
                    f"(toplam hata: {task.error_count})"
                )
                # Hata durumunda bekle, tekrar dene
                await asyncio.sleep(min(task.interval * 2, 120))
                continue

            await asyncio.sleep(task.interval)

    # ───────────────────────────────────────────────────────────
    # DURUM SORGULAMA
    # ───────────────────────────────────────────────────────────
    def get_status(self) -> List[Dict[str, Any]]:
        """Tüm görevlerin anlık durumunu döndür."""
        return [
            {
                "name": t.name,
                "agent": t.agent,
                "interval_sec": t.interval,
                "priority": t.priority,
                "enabled": t.enabled,
                "run_count": t.run_count,
                "error_count": t.error_count,
                "last_run": (
                    t.last_run.strftime("%H:%M:%S") if t.last_run else "—"
                ),
            }
            for t in self._tasks
        ]

    @property
    def task_count(self) -> int:
        """Kayıtlı görev sayısı"""
        return len(self._tasks)

    @property
    def is_running(self) -> bool:
        return self._running
