"""
Daily Brief Skill
Versiyon: 1.0.0
Açıklama: Her sabah 09:00-10:00 arası günün sistem özetini ATLAS aracılığıyla sunar.
           Bir gün içinde yalnızca bir kez çalışır.
"""

from datetime import datetime
from typing import Optional

from skills.base import BaseSkill


class DailyBriefSkill(BaseSkill):
    """
    Günlük brifing — sabah saatinde otomatik sistem özeti.

    Davranış:
        - Her saat kontrol eder
        - Saat 09:00-10:00 aralığındaysa özet bildirir
        - Aynı gün ikinci kez tetiklenmez
    """

    name = "daily_brief"
    description = "Her sabah sistem özetini proaktif olarak seslendirir"
    version = "1.0.0"
    agent = "ATLAS"
    priority = 3
    heartbeat_interval = 3600  # Her saat kontrol et

    # Hangi saatte bildirim yapılsın?
    BRIEF_HOUR: int = 9   # 09:xx

    def __init__(self, tools=None) -> None:
        super().__init__(tools)
        self._last_briefed_date: Optional[str] = None

    async def on_heartbeat(self) -> Optional[str]:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Sadece belirlenen saatte ve günde bir kez çalış
        if now.hour != self.BRIEF_HOUR:
            return None
        if self._last_briefed_date == today:
            return None

        self._last_briefed_date = today

        health = self.tools.get("system")
        summary = ""

        if health:
            try:
                summary = health.get_status_summary()
                # Çok uzunsa kısalt
                if len(summary) > 200:
                    summary = summary[:197] + "..."
            except Exception:
                summary = "Sistem durumu alınamadı."

        return (
            f"Günaydın! {now.strftime('%d %B %Y')} sabahı sistem özeti: {summary}"
            if summary
            else f"Günaydın! {now.strftime('%d %B %Y')} tarihli yeni bir güne başlıyoruz."
        )
