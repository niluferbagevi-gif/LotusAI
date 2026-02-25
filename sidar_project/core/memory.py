"""
Sidar Project - Konuşma Belleği
Çoklu tur konuşma geçmişini yönetir.
"""

import time
import threading
from typing import List, Dict, Optional


class ConversationMemory:
    """Thread-safe konuşma belleği yöneticisi."""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._turns: List[Dict] = []
        self._lock = threading.RLock()
        self._last_file: Optional[str] = None   # Son okunan/yazılan dosya yolu

    # ─────────────────────────────────────────────
    #  EKLEME & OKUMA
    # ─────────────────────────────────────────────

    def add(self, role: str, content: str) -> None:
        """Yeni bir mesaj turu ekle. 'role' değeri 'user' veya 'assistant' olmalı."""
        with self._lock:
            self._turns.append({
                "role": role,
                "content": content,
                "timestamp": time.time(),
            })
            # Pencere boyutunu koru
            if len(self._turns) > self.max_turns * 2:
                self._turns = self._turns[-(self.max_turns * 2):]

    def get_history(self, n_last: Optional[int] = None) -> List[Dict]:
        """Son n_last turu döndür. None ise tamamını döndür."""
        with self._lock:
            turns = list(self._turns)
        return turns if n_last is None else turns[-n_last:]

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """LLM API çağrısı için mesaj listesi döndür."""
        with self._lock:
            return [{"role": t["role"], "content": t["content"]} for t in self._turns]

    # ─────────────────────────────────────────────
    #  DOSYA TAKİBİ
    # ─────────────────────────────────────────────

    def set_last_file(self, path: str) -> None:
        with self._lock:
            self._last_file = path

    def get_last_file(self) -> Optional[str]:
        with self._lock:
            return self._last_file

    # ─────────────────────────────────────────────
    #  YARDIMCILAR
    # ─────────────────────────────────────────────

    def clear(self) -> None:
        """Belleği temizle."""
        with self._lock:
            self._turns.clear()
            self._last_file = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    def __repr__(self) -> str:
        return f"<ConversationMemory turns={len(self._turns)} max={self.max_turns}>"
