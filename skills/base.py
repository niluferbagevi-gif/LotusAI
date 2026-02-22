"""
LotusAI BaseSkill
Versiyon: 1.0.0
Açıklama: Tüm LotusAI skill'lerinin miras aldığı taban sınıf.

Yeni bir skill yazmak için:
    1. Bu sınıfı miras alın
    2. Sınıf değişkenlerini doldurun (name, description, agent, ...)
    3. Dosyayı skills/custom/ dizinine koyun
    4. Sistem yeniden başladığında otomatik yüklenir
"""

from abc import ABC
from typing import Any, Dict, Optional


class BaseSkill(ABC):
    """
    LotusAI dinamik skill taban sınıfı.

    Sınıf Değişkenleri (alt sınıfta tanımlanmalı):
        name              : Skill'in benzersiz kısa adı ("system_pulse")
        description       : İnsanın okuyabileceği açıklama
        version           : Semantik versiyon ("1.0.0")
        agent             : Bildirimi seslendirecek ajan ("ATLAS", "KURT" ...)
        priority          : 1=en kritik, 10=en düşük öncelik
        heartbeat_interval: Saniye cinsinden heartbeat aralığı (0 = heartbeat yok)
        enabled           : False ise SkillRegistry bu skill'i yüklemez

    Örnek:
        class MySkill(BaseSkill):
            name = "my_skill"
            description = "Özel bir işlem yapar"
            agent = "ATLAS"
            heartbeat_interval = 600  # 10 dakikada bir çalış

            async def on_heartbeat(self) -> Optional[str]:
                return "Kontrol tamamlandı"
    """

    # ── Kimlik ──────────────────────────────────────────────────
    name: str = "base_skill"
    description: str = "Açıklama tanımlanmamış"
    version: str = "1.0.0"

    # ── Davranış ────────────────────────────────────────────────
    agent: str = "ATLAS"
    priority: int = 5
    heartbeat_interval: float = 0   # 0 = proaktif heartbeat yok
    enabled: bool = True

    def __init__(self, tools: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            tools: lotus_system.py'deki manager sözlüğü.
                   {"system": SystemHealthManager, "finance": FinanceManager, ...}
        """
        self.tools: Dict[str, Any] = tools or {}
        self._initialized: bool = False

    # ── Yaşam döngüsü ───────────────────────────────────────────
    async def initialize(self) -> bool:
        """
        Skill başlangıç kurulumu.
        Gerekli kaynakların hazır olduğunu doğrulayın.

        Returns:
            True  → başarılı, False → başlatma başarısız (devre dışı kalır)
        """
        self._initialized = True
        return True

    # ── Proaktif heartbeat ──────────────────────────────────────
    async def on_heartbeat(self) -> Optional[str]:
        """
        `heartbeat_interval` saniyede bir otomatik çağrılır.

        Returns:
            None  → bildirim yok
            str   → bu metin ilgili ajan tarafından seslendirilir
        """
        return None

    # ── Komut işleme (opsiyonel) ─────────────────────────────────
    def can_handle(self, text: str) -> bool:
        """Bu skill verilen kullanıcı metnini işleyebilir mi?"""
        return False

    async def on_command(
        self, command: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Kullanıcı komutuna yanıt üret.
        `can_handle()` True döndürdüğünde AgentEngine bu metodu çağırır.
        """
        return None

    def __repr__(self) -> str:
        return (
            f"<Skill:{self.name} v{self.version} "
            f"agent={self.agent} hb={self.heartbeat_interval}s>"
        )
