"""
System Pulse Skill
Versiyon: 1.0.0
Açıklama: Her 5 dakikada bir CPU/RAM/GPU durumunu izler.
           Kritik eşik aşılırsa ATLAS aracılığıyla sesli uyarı verir.
"""

from typing import Optional

from skills.base import BaseSkill


class SystemPulseSkill(BaseSkill):
    """
    Sistem Nabzı — proaktif donanım izleyici.

    Eşikler:
        CPU  > %85 → kritik uyarı
        RAM  > %90 → kritik uyarı
    """

    name = "system_pulse"
    description = "Sistem sağlığını proaktif olarak izler ve kritik durumlarda uyarır"
    version = "1.0.0"
    agent = "ATLAS"
    priority = 2
    heartbeat_interval = 300  # 5 dakika

    # Uyarı eşikleri
    CPU_CRITICAL: float = 85.0
    RAM_CRITICAL: float = 90.0

    async def initialize(self) -> bool:
        health = self.tools.get("system")
        if health is None:
            return False  # SystemHealthManager yoksa bu skill işlevsiz
        self._initialized = True
        return True

    async def on_heartbeat(self) -> Optional[str]:
        health = self.tools.get("system")
        if not health:
            return None

        try:
            metrics: dict = health.get_metrics()
        except Exception:
            return None

        cpu: float = metrics.get("cpu_percent", 0.0)
        ram: float = metrics.get("ram_percent", 0.0)

        alerts = []
        if cpu >= self.CPU_CRITICAL:
            alerts.append(f"İşlemci kullanımı kritik seviyede, yüzde {cpu:.0f}")
        if ram >= self.RAM_CRITICAL:
            alerts.append(f"Bellek kullanımı kritik seviyede, yüzde {ram:.0f}")

        if alerts:
            return "Sistem uyarısı: " + ". ".join(alerts)

        return None
