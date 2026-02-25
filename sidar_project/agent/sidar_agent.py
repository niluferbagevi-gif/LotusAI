"""
Sidar Project - Ana Ajan
ReAct (Reason + Act) döngüsü ile çalışan yazılım mühendisi AI asistanı.
"""

import logging
import threading
from typing import Optional

from config import Config
from core.memory import ConversationMemory
from core.llm_client import LLMClient
from managers.code_manager import CodeManager
from managers.system_health import SystemHealthManager
from managers.github_manager import GitHubManager
from managers.security import SecurityManager
from agent.auto_handle import AutoHandle
from agent.definitions import SIDAR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class SidarAgent:
    """
    Sidar — Yazılım Mimarı ve Baş Mühendis AI Asistanı.

    Mimari:
    1. Kullanıcı girdisi → AutoHandle (hızlı örüntü eşleme)
    2. Eşleşme yoksa → ReAct döngüsü (LLM + araç çağrısı)
    3. LLM yanıtı → Hafızaya kaydet → Kullanıcıya sun
    """

    VERSION = "1.0.0"

    def __init__(self, cfg: Config = None) -> None:
        self.cfg = cfg or Config()
        self._lock = threading.RLock()

        # Alt sistemler
        self.security = SecurityManager(self.cfg.ACCESS_LEVEL, self.cfg.BASE_DIR)
        self.code = CodeManager(self.security, self.cfg.BASE_DIR)
        self.health = SystemHealthManager(self.cfg.USE_GPU)
        self.github = GitHubManager(self.cfg.GITHUB_TOKEN, self.cfg.GITHUB_REPO)
        self.memory = ConversationMemory(self.cfg.MAX_MEMORY_TURNS)
        self.llm = LLMClient(self.cfg.AI_PROVIDER, self.cfg)
        self.auto = AutoHandle(self.code, self.health, self.github, self.memory)

        logger.info(
            "SidarAgent v%s başlatıldı — sağlayıcı=%s model=%s erişim=%s",
            self.VERSION, self.cfg.AI_PROVIDER, self.cfg.CODING_MODEL, self.cfg.ACCESS_LEVEL,
        )

    # ─────────────────────────────────────────────
    #  ANA YANIT METODU
    # ─────────────────────────────────────────────

    def respond(self, user_input: str) -> str:
        """
        Kullanıcı girdisini işle ve yanıt üret.

        1. AutoHandle → hızlı yol
        2. ReAct loop → LLM ile araç çağrısı
        """
        user_input = user_input.strip()
        if not user_input:
            return "⚠ Boş girdi."

        with self._lock:
            # Belleğe kaydet
            self.memory.add("user", user_input)

            # Hızlı yol: otomatik örüntü eşleme
            handled, quick_response = self.auto.handle(user_input)
            if handled:
                self.memory.add("assistant", quick_response)
                return quick_response

            # Yavaş yol: LLM + ReAct
            response = self._react_loop(user_input)
            self.memory.add("assistant", response)
            return response

    # ─────────────────────────────────────────────
    #  ReAct DÖNGÜSÜ
    # ─────────────────────────────────────────────

    def _react_loop(self, user_input: str) -> str:
        """
        LLM'e araç talimatları içeren bir sistem talimatıyla istek gönder.
        LLM, gerekirse araç çağrısı yapar; bu döngü sonuçları toplayıp son yanıtı üretir.
        """
        messages = self.memory.get_messages_for_llm()

        # Araç durumu bağlamını sistem talimatına ekle
        context = self._build_context()
        full_system = SIDAR_SYSTEM_PROMPT + "\n\n" + context

        for step in range(self.cfg.MAX_REACT_STEPS):
            response = self.llm.chat(
                messages=messages,
                model=self.cfg.CODING_MODEL,
                system_prompt=full_system,
                temperature=0.3,
            )

            # Araç çağrısı var mı kontrol et
            tool_result = self._check_and_execute_tools(response)
            if tool_result is None:
                # Araç çağrısı yok → son yanıt
                return response

            # Araç sonucunu bağlama ekle ve döngüye devam et
            messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": f"[Araç Sonucu]\n{tool_result}"},
            ]

        return response  # Maksimum adım aşıldı

    def _check_and_execute_tools(self, llm_response: str) -> Optional[str]:
        """
        LLM yanıtında araç çağrısı direktifi var mı kontrol et ve çalıştır.

        Desteklenen direktifler (basit metin tabanlı):
        TOOL:list_dir:<path>
        TOOL:read_file:<path>
        TOOL:audit
        TOOL:health
        TOOL:gpu_optimize
        TOOL:github_commits:<n>
        TOOL:github_info
        """
        import re

        # Basit metin-tabanlı araç ayrıştırma
        m = re.search(r"TOOL:(\w+)(?::(.+))?", llm_response)
        if not m:
            return None

        tool_name = m.group(1)
        tool_arg = (m.group(2) or "").strip()

        if tool_name == "list_dir":
            ok, result = self.code.list_directory(tool_arg or ".")
            return result
        elif tool_name == "read_file":
            if not tool_arg:
                return "Dosya yolu belirtilmedi."
            ok, result = self.code.read_file(tool_arg)
            if ok:
                self.memory.set_last_file(tool_arg)
            return result
        elif tool_name == "audit":
            return self.code.audit_project(tool_arg or ".")
        elif tool_name == "health":
            return self.health.full_report()
        elif tool_name == "gpu_optimize":
            return self.health.optimize_gpu_memory()
        elif tool_name == "github_commits":
            try:
                n = int(tool_arg) if tool_arg else 10
            except ValueError:
                n = 10
            ok, result = self.github.list_commits(n=n)
            return result
        elif tool_name == "github_info":
            ok, result = self.github.get_repo_info()
            return result

        return None

    # ─────────────────────────────────────────────
    #  BAĞLAM OLUŞTURMA
    # ─────────────────────────────────────────────

    def _build_context(self) -> str:
        """Araç durumlarını özetleyen bağlam dizesi."""
        lines = ["[Araç Durumu]"]
        lines.append(f"  Güvenlik: {self.security.level_name.upper()}")
        lines.append(f"  GitHub  : {'Bağlı' if self.github.is_available() else 'Bağlı değil'}")
        lines.append(f"  GPU     : {'Mevcut' if self.health._gpu_available else 'Yok'}")
        m = self.code.get_metrics()
        lines.append(f"  Okunan  : {m['files_read']} dosya | Yazılan: {m['files_written']}")

        last_file = self.memory.get_last_file()
        if last_file:
            lines.append(f"  Son dosya: {last_file}")

        lines.append("")
        lines.append("[Araç Direktifleri]")
        lines.append("Araç çağırmak için: TOOL:<isim>:<argüman>")
        lines.append("  TOOL:list_dir:<yol>")
        lines.append("  TOOL:read_file:<yol>")
        lines.append("  TOOL:audit")
        lines.append("  TOOL:health")
        lines.append("  TOOL:gpu_optimize")
        lines.append("  TOOL:github_commits:<n>")
        lines.append("  TOOL:github_info")

        return "\n".join(lines)

    # ─────────────────────────────────────────────
    #  YARDIMCI METODLAR
    # ─────────────────────────────────────────────

    def clear_memory(self) -> str:
        self.memory.clear()
        return "Konuşma belleği temizlendi. ✓"

    def status(self) -> str:
        lines = [
            f"[SidarAgent v{self.VERSION}]",
            f"  Sağlayıcı    : {self.cfg.AI_PROVIDER}",
            f"  Model        : {self.cfg.CODING_MODEL}",
            f"  Erişim       : {self.cfg.ACCESS_LEVEL}",
            f"  Bellek       : {len(self.memory)} mesaj",
            f"  {self.github.status()}",
            self.health.full_report(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<SidarAgent v{self.VERSION} provider={self.cfg.AI_PROVIDER}>"
