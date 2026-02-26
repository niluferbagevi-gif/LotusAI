"""
Sidar Project - Konfigürasyon Modülü
Yazılım Mühendisi AI Asistanı
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  PROJE KÖK DİZİNİ
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class Config:
    # ─────────────────────────────────────────────
    #  AI SAĞLAYICI
    # ─────────────────────────────────────────────
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "ollama")   # "ollama" | "gemini"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    # ─────────────────────────────────────────────
    #  OLLAMA
    # ─────────────────────────────────────────────
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
    CODING_MODEL: str = os.getenv("CODING_MODEL", "qwen2.5-coder:7b")
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "gemma2:9b")

    # ─────────────────────────────────────────────
    #  ERİŞİM SEVİYESİ (OpenClaw Sistemi)
    # ─────────────────────────────────────────────
    # restricted : Salt okunur, denetim ve analiz; yazma işlemi yok
    # sandbox    : Serbest okuma; yalnızca /temp dizinine yazma (varsayılan güvenli mod)
    # full       : Terminal komutları dahil tam erişim
    ACCESS_LEVEL: str = os.getenv("ACCESS_LEVEL", "sandbox")

    # ─────────────────────────────────────────────
    #  GITHUB
    # ─────────────────────────────────────────────
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "")          # örn: "kullanici/repo"

    # ─────────────────────────────────────────────
    #  DONANIM
    # ─────────────────────────────────────────────
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"

    # ─────────────────────────────────────────────
    #  UYGULAMA
    # ─────────────────────────────────────────────
    MAX_MEMORY_TURNS: int = int(os.getenv("MAX_MEMORY_TURNS", "20"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    RESPONSE_LANGUAGE: str = os.getenv("RESPONSE_LANGUAGE", "tr")  # "tr" | "en"

    # ─────────────────────────────────────────────
    #  DIZINLER
    # ─────────────────────────────────────────────
    BASE_DIR: Path = BASE_DIR
    TEMP_DIR: Path = TEMP_DIR
    LOGS_DIR: Path = LOGS_DIR

    # ─────────────────────────────────────────────
    #  ReAct DÖNGÜSÜ
    # ─────────────────────────────────────────────
    MAX_REACT_STEPS: int = int(os.getenv("MAX_REACT_STEPS", "10"))
    REACT_TIMEOUT: int = int(os.getenv("REACT_TIMEOUT", "60"))   # saniye
