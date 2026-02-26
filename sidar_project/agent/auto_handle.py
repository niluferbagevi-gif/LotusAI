"""
Sidar Project - Otomatik Komut İşleyici
Kullanıcı girdisindeki ortak kalıpları otomatik olarak tanır ve işler.
"""

import re
from typing import Optional, Tuple

from managers.code_manager import CodeManager
from managers.system_health import SystemHealthManager
from managers.github_manager import GitHubManager
from core.memory import ConversationMemory


class AutoHandle:
    """
    Kullanıcı mesajlarını anahtar kelime örüntülerine göre analiz eder
    ve uygun manager metodunu çağırır.

    Dönen değer: (işlendi_mi: bool, yanıt: str)
    """

    def __init__(
        self,
        code: CodeManager,
        health: SystemHealthManager,
        github: GitHubManager,
        memory: ConversationMemory,
    ) -> None:
        self.code = code
        self.health = health
        self.github = github
        self.memory = memory

    # ─────────────────────────────────────────────
    #  ANA GİRİŞ NOKTASI
    # ─────────────────────────────────────────────

    def handle(self, text: str) -> Tuple[bool, str]:
        """
        text: kullanıcı mesajı (küçük harf + şeritlenmiş)

        Returns:
            (True, yanıt)  — otomatik işlendiyse
            (False, "")    — LLM'e ilet
        """
        t = text.lower().strip()

        # Dizin listeleme
        result = self._try_list_directory(t, text)
        if result[0]:
            return result

        # Dosya okuma
        result = self._try_read_file(t, text)
        if result[0]:
            return result

        # Sistem denetimi
        result = self._try_audit(t)
        if result[0]:
            return result

        # Sistem sağlığı raporu
        result = self._try_health(t)
        if result[0]:
            return result

        # GPU optimizasyonu
        result = self._try_gpu_optimize(t)
        if result[0]:
            return result

        # Sözdizimi doğrulama
        result = self._try_validate_file(t, text)
        if result[0]:
            return result

        # GitHub commit listesi
        result = self._try_github_commits(t)
        if result[0]:
            return result

        # GitHub depo bilgisi
        result = self._try_github_info(t)
        if result[0]:
            return result

        # GitHub uzak dosya okuma
        result = self._try_github_read(t, text)
        if result[0]:
            return result

        # Güvenlik durumu
        result = self._try_security_status(t)
        if result[0]:
            return result

        return False, ""

    # ─────────────────────────────────────────────
    #  ÖZEL ÖRÜNTÜ İŞLEYİCİLERİ
    # ─────────────────────────────────────────────

    def _try_list_directory(self, t: str, raw: str) -> Tuple[bool, str]:
        patterns = [
            r"listele|dosyaları\s+göster|klasör.*içer|dizin.*listele|ls\b",
        ]
        if any(re.search(p, t) for p in patterns):
            # Yol çıkarmaya çalış
            path = self._extract_path(raw) or "."
            ok, result = self.code.list_directory(path)
            return True, result
        return False, ""

    def _try_read_file(self, t: str, raw: str) -> Tuple[bool, str]:
        patterns = [
            r"(dosyayı?\s+oku|incele|göster|içeriğ|cat\b)",
        ]
        if any(re.search(p, t) for p in patterns):
            path = self._extract_path(raw) or self.memory.get_last_file()
            if not path:
                return True, "⚠ Hangi dosyayı okumamı istiyorsunuz? Lütfen dosya yolunu belirtin."
            ok, content = self.code.read_file(path)
            if ok:
                self.memory.set_last_file(path)
                # Büyük dosyalar için kısalt
                lines = content.splitlines()
                preview = "\n".join(lines[:80])
                suffix = f"\n... ({len(lines) - 80} satır daha)" if len(lines) > 80 else ""
                return True, f"[{path}]\n```\n{preview}{suffix}\n```"
            return True, f"✗ {content}"
        return False, ""

    def _try_audit(self, t: str) -> Tuple[bool, str]:
        if re.search(r"denetle|sistemi\s+tara|audit|teknik\s+rapor|kod.*kontrol", t):
            report = self.code.audit_project(".")
            return True, report
        return False, ""

    def _try_health(self, t: str) -> Tuple[bool, str]:
        if re.search(r"sistem.*sağlık|donanım|hardware|cpu|ram|memory.*report|sağlık.*rapor", t):
            return True, self.health.full_report()
        return False, ""

    def _try_gpu_optimize(self, t: str) -> Tuple[bool, str]:
        if re.search(r"gpu.*(optimize|temizle|boşalt|clear)|vram", t):
            return True, self.health.optimize_gpu_memory()
        return False, ""

    def _try_validate_file(self, t: str, raw: str) -> Tuple[bool, str]:
        if re.search(r"sözdizimi|syntax|doğrula|validate|kontrol\s+et", t):
            path = self._extract_path(raw) or self.memory.get_last_file()
            if not path:
                return True, "⚠ Doğrulanacak dosya yolunu belirtin."
            ok, content = self.code.read_file(path)
            if not ok:
                return True, f"✗ Dosya okunamadı: {content}"
            if path.endswith(".py"):
                ok, msg = self.code.validate_python_syntax(content)
            elif path.endswith(".json"):
                ok, msg = self.code.validate_json(content)
            else:
                return True, f"⚠ {path} için sözdizimi doğrulama desteklenmiyor."
            icon = "✓" if ok else "✗"
            return True, f"{icon} {msg}"
        return False, ""

    def _try_github_commits(self, t: str) -> Tuple[bool, str]:
        if re.search(r"(github|commit).*(listele|göster|son|last)", t) or re.search(r"son.*commit", t):
            if not self.github.is_available():
                return True, "⚠ GitHub token ayarlanmamış."
            # Commit sayısı çıkarmaya çalış
            m = re.search(r"(\d+)\s*commit", t)
            n = int(m.group(1)) if m else 10
            ok, result = self.github.list_commits(n=n)
            return True, result
        return False, ""

    def _try_github_info(self, t: str) -> Tuple[bool, str]:
        if re.search(r"github.*(bilgi|info|repo|depo)|(depo|repo).*(bilgi|info)", t):
            if not self.github.is_available():
                return True, "⚠ GitHub token ayarlanmamış."
            ok, result = self.github.get_repo_info()
            return True, result
        return False, ""

    def _try_github_read(self, t: str, raw: str) -> Tuple[bool, str]:
        if re.search(r"github.*(oku|dosya|file)|uzak.*dosya", t):
            if not self.github.is_available():
                return True, "⚠ GitHub token ayarlanmamış."
            path = self._extract_path(raw)
            if not path:
                return True, "⚠ Okunacak GitHub dosya yolunu belirtin."
            ok, content = self.github.read_remote_file(path)
            return True, content if ok else f"✗ {content}"
        return False, ""

    def _try_security_status(self, t: str) -> Tuple[bool, str]:
        if re.search(r"erişim|güvenlik|openclaw|access.*level|yetki", t):
            # SecurityManager'a referans gerekmez — CodeManager üzerinden alınır
            return True, self.code.security.status_report()
        return False, ""

    # ─────────────────────────────────────────────
    #  YARDIMCILAR
    # ─────────────────────────────────────────────

    def _extract_path(self, text: str) -> Optional[str]:
        """Metinden dosya/dizin yolu çıkar."""
        # Tırnak içindeki yol
        m = re.search(r'["\']([^"\']+\.[a-zA-Z]{1,6})["\']', text)
        if m:
            return m.group(1)
        # .py / .json / .md / .txt gibi uzantılı token
        m = re.search(r'\b([\w/\\.\-]+\.(?:py|json|md|txt|yaml|yml|js|ts|sh))\b', text)
        if m:
            return m.group(1)
        return None
