"""
Sidar Project - LLM İstemcisi
Ollama ve Google Gemini API entegrasyonu.
"""

import json
import logging
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """Ollama veya Gemini üzerinden LLM çağrıları yapar."""

    def __init__(self, provider: str, config) -> None:
        """
        provider: "ollama" | "gemini"
        config  : Config nesnesi
        """
        self.provider = provider.lower()
        self.config = config

    # ─────────────────────────────────────────────
    #  ANA ÇAĞRI NOKTASI
    # ─────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """
        Sohbet tamamlama isteği gönder.

        Args:
            messages      : [{"role": "user", "content": "..."}, ...]
            model         : None ise Config.CODING_MODEL kullanılır
            system_prompt : Varsa başa eklenir
            temperature   : Yanıt yaratıcılığı (0-1)

        Returns:
            LLM yanıtı (str)
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        if self.provider == "ollama":
            return self._ollama_chat(messages, model or self.config.CODING_MODEL, temperature)
        elif self.provider == "gemini":
            return self._gemini_chat(messages, temperature)
        else:
            raise ValueError(f"Bilinmeyen AI sağlayıcısı: {self.provider}")

    # ─────────────────────────────────────────────
    #  OLLAMA
    # ─────────────────────────────────────────────

    def _ollama_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        url = f"{self.config.OLLAMA_URL.rstrip('/api')}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            logger.error("Ollama bağlantı hatası. Ollama çalışıyor mu?")
            return "[HATA] Ollama'ya bağlanılamadı. 'ollama serve' komutunu çalıştırın."
        except Exception as exc:
            logger.error("Ollama hata: %s", exc)
            return f"[HATA] Ollama: {exc}"

    # ─────────────────────────────────────────────
    #  GEMINI
    # ─────────────────────────────────────────────

    def _gemini_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            return "[HATA] 'google-generativeai' paketi kurulu değil. pip install google-generativeai"

        if not self.config.GEMINI_API_KEY:
            return "[HATA] GEMINI_API_KEY ayarlanmamış."

        genai.configure(api_key=self.config.GEMINI_API_KEY)

        # Sistem mesajını ayır
        system_text = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                chat_messages.append(m)

        model = genai.GenerativeModel(
            model_name=self.config.GEMINI_MODEL,
            system_instruction=system_text or None,
            generation_config={"temperature": temperature},
        )

        # Mesajları Gemini formatına dönüştür
        history = []
        last_user = None
        for m in chat_messages:
            role = "user" if m["role"] == "user" else "model"
            if role == "user":
                last_user = m["content"]
                if history or last_user:
                    history.append({"role": role, "parts": [m["content"]]})
            else:
                history.append({"role": role, "parts": [m["content"]]})

        if not last_user and chat_messages:
            last_user = chat_messages[-1]["content"]

        try:
            chat_session = model.start_chat(history=history[:-1] if history else [])
            response = chat_session.send_message(last_user or "Merhaba")
            return response.text
        except Exception as exc:
            logger.error("Gemini hata: %s", exc)
            return f"[HATA] Gemini: {exc}"

    # ─────────────────────────────────────────────
    #  YARDIMCILAR
    # ─────────────────────────────────────────────

    def list_ollama_models(self) -> List[str]:
        """Mevcut Ollama modellerini listele."""
        url = f"{self.config.OLLAMA_URL.rstrip('/api')}/api/tags"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        except Exception:
            return []

    def is_ollama_available(self) -> bool:
        """Ollama sunucusunun çalışıp çalışmadığını kontrol et."""
        try:
            url = f"{self.config.OLLAMA_URL.rstrip('/api')}/api/tags"
            requests.get(url, timeout=5)
            return True
        except Exception:
            return False
