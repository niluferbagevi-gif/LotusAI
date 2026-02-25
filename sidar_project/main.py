"""
Sidar Project - Giriş Noktası
Yazılım Mühendisi AI Asistanı — CLI Arayüzü

Kullanım:
    python main.py                  # interaktif mod
    python main.py --status         # sistem durumunu göster
    python main.py -c "komut"       # tek komut çalıştır
    python main.py --level full     # erişim seviyesini geçici olarak ayarla
"""

import argparse
import logging
import os
import sys

# Proje kökünü sys.path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from agent.sidar_agent import SidarAgent


# ─────────────────────────────────────────────
#  LOGLAMA
# ─────────────────────────────────────────────

def _setup_logging(level: str) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(Config.LOGS_DIR / "sidar.log"), encoding="utf-8"),
        ],
    )


# ─────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────

BANNER = r"""
 ╔══════════════════════════════════════════════╗
 ║  ███████╗██╗██████╗  █████╗ ██████╗          ║
 ║  ██╔════╝██║██╔══██╗██╔══██╗██╔══██╗         ║
 ║  ███████╗██║██║  ██║███████║██████╔╝         ║
 ║  ╚════██║██║██║  ██║██╔══██║██╔══██╗         ║
 ║  ███████║██║██████╔╝██║  ██║██║  ██║         ║
 ║  ╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝         ║
 ║  Yazılım Mimarı & Baş Mühendis AI  v1.0.0   ║
 ╚══════════════════════════════════════════════╝
"""

HELP_TEXT = """
Komutlar:
  .status     — Sistem durumunu göster
  .clear      — Konuşma belleğini temizle
  .audit      — Proje denetimini çalıştır
  .health     — Sistem sağlık raporu
  .gpu        — GPU belleğini optimize et
  .github     — GitHub bağlantı durumu
  .level      — Mevcut erişim seviyesini göster
  .help       — Bu yardım mesajını göster
  .exit / .q  — Çıkış
"""


# ─────────────────────────────────────────────
#  İNTERAKTİF DÖNGÜ
# ─────────────────────────────────────────────

def interactive_loop(agent: SidarAgent) -> None:
    print(BANNER)
    print(f"  Erişim Seviyesi : {agent.cfg.ACCESS_LEVEL.upper()}")
    print(f"  AI Sağlayıcı    : {agent.cfg.AI_PROVIDER} ({agent.cfg.CODING_MODEL})")
    print(f"  GitHub          : {'Bağlı' if agent.github.is_available() else 'Bağlı değil'}")
    print(f"\n  '.help' yazarak komut listesini görebilirsiniz.\n")

    while True:
        try:
            user_input = input("Sen  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSidar > Görüşürüz. ✓")
            break

        if not user_input:
            continue

        # Dahili komutlar
        if user_input.lower() in (".exit", ".q", "exit", "quit", "çıkış"):
            print("Sidar > Görüşürüz. ✓")
            break
        elif user_input.lower() == ".help":
            print(HELP_TEXT)
            continue
        elif user_input.lower() == ".status":
            print(agent.status())
            continue
        elif user_input.lower() == ".clear":
            print(agent.clear_memory())
            continue
        elif user_input.lower() == ".audit":
            print(agent.code.audit_project("."))
            continue
        elif user_input.lower() == ".health":
            print(agent.health.full_report())
            continue
        elif user_input.lower() == ".gpu":
            print(agent.health.optimize_gpu_memory())
            continue
        elif user_input.lower() == ".github":
            print(agent.github.status())
            continue
        elif user_input.lower() == ".level":
            print(agent.security.status_report())
            continue

        # Ajan yanıtı
        try:
            response = agent.respond(user_input)
            print(f"Sidar > {response}\n")
        except Exception as exc:
            print(f"Sidar > ✗ Hata: {exc}\n")
            logging.exception("Ajan yanıt hatası")


# ─────────────────────────────────────────────
#  GİRİŞ NOKTASI
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sidar — Yazılım Mühendisi AI Asistanı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--command", help="Tek komut çalıştır ve çık")
    parser.add_argument("--status", action="store_true", help="Sistem durumunu göster ve çık")
    parser.add_argument(
        "--level",
        choices=["restricted", "sandbox", "full"],
        help="Erişim seviyesini geçici olarak ayarla",
    )
    parser.add_argument("--provider", choices=["ollama", "gemini"], help="AI sağlayıcısı")
    parser.add_argument("--model", help="Ollama model adı")
    parser.add_argument("--log", default="INFO", help="Log seviyesi (DEBUG/INFO/WARNING)")
    args = parser.parse_args()

    _setup_logging(args.log)

    # Config override
    if args.level:
        os.environ["ACCESS_LEVEL"] = args.level
    if args.provider:
        os.environ["AI_PROVIDER"] = args.provider
    if args.model:
        os.environ["CODING_MODEL"] = args.model

    agent = SidarAgent()

    if args.status:
        print(agent.status())
        return

    if args.command:
        response = agent.respond(args.command)
        print(f"Sidar > {response}")
        return

    interactive_loop(agent)


if __name__ == "__main__":
    main()
