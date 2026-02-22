"""
LotusAI Skill Registry
Versiyon: 1.0.0
Açıklama: Skill'leri dinamik olarak keşfeder, yükler ve yönetir.
           skills/builtin/ ve skills/custom/ dizinleri otomatik taranır.
           Yeni bir skill eklemek için core koda dokunmak gerekmez.
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from skills.base import BaseSkill

logger = logging.getLogger("LotusAI.SkillRegistry")


class SkillRegistry:
    """
    Dinamik skill yöneticisi.

    Tarama sırası:
        1. skills/builtin/  — Sistem tarafından gelen yerleşik skill'ler
        2. skills/custom/   — Kullanıcının kendi yazdığı skill'ler

    Kullanım:
        registry = SkillRegistry()
        registry.load_all(tools=tools_dict)
        await registry.initialize_all()
        heartbeat_skills = registry.get_heartbeat_skills()
    """

    # Otomatik taranan paket yolları (öncelik sırasına göre)
    _PACKAGES = [
        "skills.builtin",
        "skills.custom",
    ]

    def __init__(self) -> None:
        self._skills: Dict[str, BaseSkill] = {}

    # ───────────────────────────────────────────────────────────
    # YÜKLEME
    # ───────────────────────────────────────────────────────────
    def load_all(self, tools: Optional[Dict[str, Any]] = None) -> int:
        """
        Tüm skill paketlerini tara ve yükle.

        Args:
            tools: Skill'lere aktarılacak manager sözlüğü

        Returns:
            Başarıyla yüklenen skill sayısı
        """
        tools = tools or {}
        total = 0

        for package in self._PACKAGES:
            total += self._load_package(package, tools)

        logger.info(f"[SKILLS] Yükleme tamamlandı → {total} skill aktif")
        return total

    def _load_package(self, package_name: str, tools: Dict[str, Any]) -> int:
        """Tek bir Python paketindeki tüm skill sınıflarını yükle."""
        loaded = 0

        try:
            pkg = importlib.import_module(package_name)
            pkg_path = Path(pkg.__file__).parent  # type: ignore[arg-type]
        except ImportError:
            logger.debug(f"[SKILLS] {package_name} paketi bulunamadı, atlanıyor")
            return 0

        for _, module_name, is_pkg in pkgutil.iter_modules([str(pkg_path)]):
            if is_pkg:
                continue  # Alt paketleri atla

            full_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_name)
            except Exception as exc:
                logger.error(f"[SKILLS] {full_name} import hatası: {exc}")
                continue

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if not (
                    inspect.isclass(attr)
                    and issubclass(attr, BaseSkill)
                    and attr is not BaseSkill
                    and getattr(attr, "enabled", True)
                ):
                    continue

                # Aynı isimde skill varsa custom olan override eder
                skill_name = getattr(attr, "name", attr.__name__)
                try:
                    skill_instance = attr(tools=tools)
                    self._skills[skill_name] = skill_instance
                    logger.info(
                        f"[SKILLS] ✓ {skill_name} yüklendi "
                        f"({full_name})"
                    )
                    loaded += 1
                except Exception as exc:
                    logger.error(
                        f"[SKILLS] {skill_name} örneklenemedi: {exc}"
                    )

        return loaded

    # ───────────────────────────────────────────────────────────
    # BAŞLATMA
    # ───────────────────────────────────────────────────────────
    async def initialize_all(self) -> None:
        """Yüklü tüm skill'lerin `initialize()` metodunu çağır."""
        failed: List[str] = []

        for name, skill in self._skills.items():
            try:
                ok = await skill.initialize()
                if not ok:
                    logger.warning(f"[SKILLS] {name} başlatılamadı, devre dışı")
                    failed.append(name)
            except Exception as exc:
                logger.error(f"[SKILLS] {name} başlatma hatası: {exc}")
                failed.append(name)

        for name in failed:
            del self._skills[name]

        logger.info(
            f"[SKILLS] Başlatma tamamlandı → {len(self._skills)} skill hazır"
        )

    # ───────────────────────────────────────────────────────────
    # SORGULAMA
    # ───────────────────────────────────────────────────────────
    def get_heartbeat_skills(self) -> List[BaseSkill]:
        """heartbeat_interval > 0 olan skill'leri döndür."""
        return [
            s for s in self._skills.values()
            if s.heartbeat_interval > 0
        ]

    def get_command_skills(self) -> List[BaseSkill]:
        """Komut işleyebilen skill'leri döndür (can_handle desteği)."""
        return [
            s for s in self._skills.values()
            if type(s).can_handle is not BaseSkill.can_handle
        ]

    def get_all(self) -> List[BaseSkill]:
        return list(self._skills.values())

    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name)

    @property
    def count(self) -> int:
        return len(self._skills)

    def summary(self) -> str:
        """Yüklü skill listesini tek satır özet olarak döndür."""
        if not self._skills:
            return "Skill yok"
        names = ", ".join(self._skills.keys())
        return f"{self.count} skill: {names}"
