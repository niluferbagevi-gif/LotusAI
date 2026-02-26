"""
Sidar Project - Güvenlik Yöneticisi
OpenClaw erişim kontrol sistemi.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Erişim seviyesi sabitleri
RESTRICTED = 0   # Yalnızca okuma, denetim ve analiz
SANDBOX = 1      # Okuma + yalnızca /temp dizinine yazma
FULL = 2         # Tam erişim (terminal dahil)

LEVEL_NAMES = {
    "restricted": RESTRICTED,
    "sandbox": SANDBOX,
    "full": FULL,
}


class SecurityManager:
    """
    OpenClaw erişim kontrol sistemi.
    Sidar'ın dosya/sistem işlemlerine yetki verir veya reddeder.
    """

    def __init__(self, access_level: str, base_dir: Path) -> None:
        self.level: int = LEVEL_NAMES.get(access_level.lower(), SANDBOX)
        self.level_name: str = access_level.lower()
        self.base_dir: Path = base_dir
        self.temp_dir: Path = base_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        logger.info("SecurityManager başlatıldı — seviye: %s (%d)", self.level_name, self.level)

    # ─────────────────────────────────────────────
    #  OKUMA YETKİSİ
    # ─────────────────────────────────────────────

    def can_read(self, path: Optional[str] = None) -> bool:
        """Her erişim seviyesinde okuma serbesttir."""
        return True

    # ─────────────────────────────────────────────
    #  YAZMA YETKİSİ
    # ─────────────────────────────────────────────

    def can_write(self, path: str) -> bool:
        """
        Yazma iznini kontrol et.
        - RESTRICTED: hiçbir zaman
        - SANDBOX: yalnızca temp/ dizini
        - FULL: her yere
        """
        if self.level == RESTRICTED:
            return False
        if self.level == FULL:
            return True
        # SANDBOX — yalnızca temp dizinine
        try:
            target = Path(path).resolve()
            return str(target).startswith(str(self.temp_dir.resolve()))
        except Exception:
            return False

    # ─────────────────────────────────────────────
    #  TERMİNAL YETKİSİ
    # ─────────────────────────────────────────────

    def can_execute(self) -> bool:
        """Terminal komutları yalnızca FULL modda çalışır."""
        return self.level == FULL

    # ─────────────────────────────────────────────
    #  YARDIMCILAR
    # ─────────────────────────────────────────────

    def get_safe_write_path(self, filename: str) -> Path:
        """Sandbox modunda güvenli yazma yolu döndürür."""
        return self.temp_dir / filename

    def status_report(self) -> str:
        perms = []
        perms.append("Okuma: ✓")
        perms.append(f"Yazma: {'✓ (tam)' if self.level == FULL else ('✓ (yalnızca /temp)' if self.level == SANDBOX else '✗')}")
        perms.append(f"Terminal: {'✓' if self.level == FULL else '✗'}")
        return (
            f"[OpenClaw] Erişim Seviyesi: {self.level_name.upper()}\n"
            + "\n".join(f"  {p}" for p in perms)
        )

    def __repr__(self) -> str:
        return f"<SecurityManager level={self.level_name}>"
