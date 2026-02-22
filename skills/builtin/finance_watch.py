"""
Finance Watch Skill
Versiyon: 1.0.0
Açıklama: FINANCE_MODE aktifken her 10 dakikada bir piyasa uyarılarını kontrol eder.
           FinanceManager'da `get_alerts()` metodu varsa kullanır.
           KURT ajanı aracılığıyla bildirim yapar.
"""

from typing import Optional

from skills.base import BaseSkill


class FinanceWatchSkill(BaseSkill):
    """
    Finansal piyasa izleyici.

    Yalnızca Config.FINANCE_MODE = True ise aktif olur.
    FinanceManager'ın `get_alerts()` metodunu çağırır;
    uyarı varsa KURT ajanı seslendirme yapar.
    """

    name = "finance_watch"
    description = "Borsa/kripto piyasasını proaktif izler ve önemli hareketleri bildirir"
    version = "1.0.0"
    agent = "KURT"
    priority = 4
    heartbeat_interval = 600  # 10 dakika

    async def initialize(self) -> bool:
        # Sadece FINANCE_MODE aktifse çalışsın
        try:
            from config import Config
            if not Config.FINANCE_MODE:
                return False  # Registry bu skill'i devre dışı bırakır
        except ImportError:
            return False

        finance = self.tools.get("finance")
        if finance is None:
            return False

        self._initialized = True
        return True

    async def on_heartbeat(self) -> Optional[str]:
        finance = self.tools.get("finance")
        if not finance:
            return None

        # FinanceManager'da get_alerts() varsa kullan
        if not hasattr(finance, "get_alerts"):
            return None

        try:
            alerts = finance.get_alerts()
            if alerts:
                first = alerts[0] if isinstance(alerts, list) else str(alerts)
                return f"Piyasa uyarısı: {first}"
        except Exception:
            pass

        return None
