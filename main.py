"""
LotusAI Launcher — OpenClaw Tarzı Başlatıcı
Versiyon: 2.6.0
Python: 3.11+
Açıklama: Erişim seviyesi seçimi, mod seçimi ve sistem başlatma.
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import traceback
import logging
import requests
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════
# TERMINAL RENK KODLARI
# ═══════════════════════════════════════════════════════════════
class Colors:
    """Terminal çıktıları için ANSI renk kodları"""
    HEADER  = '\033[95m'
    OKBLUE  = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    ENDC    = '\033[0m'
    BOLD    = '\033[1m'
    CYAN    = '\033[96m'

# ═══════════════════════════════════════════════════════════════
# PROJE YOL AYARLARI
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# ═══════════════════════════════════════════════════════════════
# EKRAN YÖNETİCİSİ (ÇOKLU EKRAN DESTEĞİ)
# ═══════════════════════════════════════════════════════════════
try:
    from dotenv import load_dotenv
    load_dotenv()
    target_screen = int(os.getenv("TARGET_SCREEN", 0))
    if target_screen > 0:
        from screen_manager import set_target_screen
        set_target_screen(target_screen)
except Exception as e:
    print(f"Ekran ayarlama atlandı: {e}")

# ═══════════════════════════════════════════════════════════════
# CONFIG IMPORT VE FALLBACK
# ═══════════════════════════════════════════════════════════════
try:
    from config import Config
except ImportError:
    @dataclass
    class Config:
        PROJECT_NAME: str = "LotusAI"
        VERSION: str = "2.6.0"
        LOG_DIR: Path = Path("logs")
        USE_GPU: bool = False
        GPU_INFO: str = "N/A"
        ACCESS_LEVEL: str = "readonly"

        @staticmethod
        def set_provider_mode(mode: str) -> None:
            pass

        @staticmethod
        def validate_critical_settings() -> bool:
            return True

# ═══════════════════════════════════════════════════════════════
# LOGLAMA SİSTEMİ
# ═══════════════════════════════════════════════════════════════
LOG_DIR = getattr(Config, "LOG_DIR", Path("logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "launcher.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger("LotusLauncher")

# ═══════════════════════════════════════════════════════════════
# LOTUS SİSTEM IMPORT
# ═══════════════════════════════════════════════════════════════
start_lotus_system: Optional[Callable] = None
import_error_message: str = ""

try:
    from lotus_system import start_lotus_system
    logger.info("lotus_system modülü başarıyla yüklendi")
except ImportError as e:
    import_error_message = (
        f"Bağımlılık eksik: {str(e)}\n\n"
        "Lütfen şu komutu çalıştırın:\n"
        "conda activate lts\n"
        "pip install -r requirements.txt"
    )
    logger.error(import_error_message)
except Exception as e:
    import_error_message = f"Sistem dosyası hatası: {str(e)}"
    logger.error(f"{import_error_message}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════
# ERİŞİM SEVİYELERİ
# ═══════════════════════════════════════════════════════════════
class AccessLevel:
    """Ajan yetki seviyeleri — OpenClaw modeli"""

    READONLY = "readonly"
    SANDBOX  = "sandbox"
    FULL     = "full"

    # Radio buton etiketleri (sabit genişlik için boşluk hizalaması)
    OPTIONS = [
        (READONLY, "Kısıtlı      (Sadece Bilgi Alma)"),
        (SANDBOX,  "Sandbox      (Güvenli Dosya Yazma)"),
        (FULL,     "Tam Erişim   (Terminal & Komut)"),
    ]

    # Terminal banner'ında gösterilecek açıklamalar
    DESCRIPTIONS = {
        READONLY: "Ajanlar yalnızca okuma yapabilir",
        SANDBOX:  "Ajanlar sandbox dizinine yazabilir",
        FULL:     "Ajanlar tam terminal yetkisine sahip",
    }


# ═══════════════════════════════════════════════════════════════
# OPENCLAW TEMA
# ═══════════════════════════════════════════════════════════════
class Theme:
    # Arka planlar
    BG_MAIN    = "#0d0d0d"
    BG_PANEL   = "#080f1c"
    BG_FRAME   = "#0f1622"
    BG_BUTTON  = "#0b1d38"
    BG_STATUS  = "#060c16"
    BG_HOVER   = "#112244"

    # Vurgular
    RED        = "#c01010"
    RED_BRIGHT = "#e02020"
    GREEN      = "#00cc44"
    GREEN_DIM  = "#009933"
    BORDER     = "#1a3a5c"
    BORDER_DIM = "#0f2235"

    # Yazı renkleri
    TEXT_WHITE = "#e0e8f0"
    TEXT_GRAY  = "#6a8099"
    TEXT_GREEN = "#00ee55"
    TEXT_DIM   = "#3a5068"
    TEXT_WARN  = "#ddaa00"

    # Font — başlangıç değerleri, apply_fonts() ile güncellenir
    F_TITLE  = ("DejaVu Sans Mono", 22, "bold")
    F_SUB    = ("DejaVu Sans Mono",  9)
    F_LABEL  = ("DejaVu Sans Mono", 10, "bold")
    F_RADIO  = ("DejaVu Sans Mono",  9)
    F_BUTTON = ("DejaVu Sans Mono", 10, "bold")
    F_STATUS = ("DejaVu Sans Mono",  8)
    F_GPU    = ("DejaVu Sans Mono",  9, "bold")
    F_SMALL  = ("DejaVu Sans Mono",  8)

    @classmethod
    def apply_fonts(cls) -> None:
        """
        Tk başlatıldıktan sonra Türkçe Unicode desteği olan
        monospace fontu otomatik seç ve tüm font tanımlarını güncelle.
        """
        import tkinter.font as tkfont
        available = set(tkfont.families())

        # Türkçe (ı ş ğ ç ö ü) destekleyen monospace fontlar — öncelik sırası
        candidates = [
            "DejaVu Sans Mono",   # Ubuntu / Debian (varsayılan)
            "Liberation Mono",    # RHEL / Fedora
            "Noto Mono",          # Geniş Unicode desteği
            "Ubuntu Mono",        # Ubuntu
            "Consolas",           # Windows (iyi Unicode desteği)
            "Courier New",        # Windows / macOS fallback
            "Menlo",              # macOS
            "Courier",            # Son çare
        ]
        mono = next((f for f in candidates if f in available), "TkFixedFont")

        cls.F_TITLE  = (mono, 22, "bold")
        cls.F_SUB    = (mono,  9)
        cls.F_LABEL  = (mono, 10, "bold")
        cls.F_RADIO  = (mono,  9)
        cls.F_BUTTON = (mono, 10, "bold")
        cls.F_STATUS = (mono,  8)
        cls.F_GPU    = (mono,  9, "bold")
        cls.F_SMALL  = (mono,  8)
        logger.debug(f"Launcher fontu: {mono}")


# ═══════════════════════════════════════════════════════════════
# LAUNCHER UYGULAMASI
# ═══════════════════════════════════════════════════════════════
class LauncherApp:
    """
    LotusAI OpenClaw Tarzı Görsel Başlatıcı

    Özellikler:
        - Terminal / hacker estetiği (koyu arka plan, monospace)
        - Ajan erişim seviyesi seçimi (Kısıtlı / Sandbox / Tam)
        - Online (Gemini) ve Local (Ollama) çalışma modu
        - Ollama servis sağlık kontrolü
        - Seçilen yetki Config üzerinden sisteme aktarılır
    """

    WIN_W = 460
    WIN_H = 530

    OLLAMA_URLS = [
        "http://127.0.0.1:11434/api/tags",
        "http://localhost:11434/api/tags",
    ]
    OLLAMA_TIMEOUT = 2.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        Theme.apply_fonts()   # Tk başladıktan sonra Türkçe uyumlu font seç
        self._access_var = tk.StringVar(value=AccessLevel.READONLY)
        self._setup_window()
        self._setup_ui()
        logger.info("Launcher başlatıldı (OpenClaw UI)")

    # ───────────────────────────────────────────────────────────
    # PENCERE
    # ───────────────────────────────────────────────────────────
    def _setup_window(self) -> None:
        self.root.title(f"LotusAI OS  v{Config.VERSION}")
        self.root.configure(bg=Theme.BG_MAIN)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x  = (sw - self.WIN_W) // 2
        y  = (sh - self.WIN_H) // 2
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}+{x}+{y}")

    # ───────────────────────────────────────────────────────────
    # UI İSKELETİ
    # ───────────────────────────────────────────────────────────
    def _setup_ui(self) -> None:
        self._create_header()
        self._create_access_frame()
        self._create_hardware_panel()
        self._create_mode_section()
        self._create_status_bar()

    # ── 1. Başlık ───────────────────────────────────────────────
    def _create_header(self) -> None:
        tk.Label(
            self.root,
            text=Config.PROJECT_NAME.upper(),
            font=Theme.F_TITLE,
            bg=Theme.BG_MAIN,
            fg=Theme.RED,
        ).pack(pady=(24, 3))

        tk.Label(
            self.root,
            text="Çok Ajanlı Yapay Zeka İşletim Sistemi",
            font=Theme.F_SUB,
            bg=Theme.BG_MAIN,
            fg=Theme.TEXT_GRAY,
        ).pack(pady=(0, 14))

    # ── 2. Erişim Seviyesi Frame ────────────────────────────────
    def _create_access_frame(self) -> None:
        # Dış kenarlık (border simülasyonu)
        outer = tk.Frame(self.root, bg=Theme.BORDER, padx=1, pady=1)
        outer.pack(fill="x", padx=28, pady=(0, 10))

        inner = tk.Frame(outer, bg=Theme.BG_FRAME)
        inner.pack(fill="both")

        # Frame başlığı
        tk.Label(
            inner,
            text=" Sistem Erişim Seviyesi (OpenClaw) ",
            font=Theme.F_LABEL,
            bg=Theme.BG_FRAME,
            fg=Theme.TEXT_GRAY,
            anchor="w",
        ).pack(fill="x", padx=10, pady=(7, 3))

        # İnce ayırıcı çizgi
        tk.Frame(inner, bg=Theme.BORDER_DIM, height=1).pack(fill="x", padx=10, pady=(0, 4))

        # Radio butonlar
        for value, label in AccessLevel.OPTIONS:
            rb = tk.Radiobutton(
                inner,
                text=f"  {label}",
                variable=self._access_var,
                value=value,
                font=Theme.F_RADIO,
                bg=Theme.BG_FRAME,
                fg=Theme.TEXT_WHITE,
                selectcolor=Theme.BG_PANEL,
                activebackground=Theme.BG_FRAME,
                activeforeground=Theme.TEXT_GREEN,
                indicatoron=True,
                bd=0,
                anchor="w",
                cursor="hand2",
            )
            rb.pack(fill="x", padx=14, pady=3)
            # Hover efekti
            rb.bind("<Enter>", lambda e, r=rb: r.config(fg=Theme.TEXT_GREEN))
            rb.bind("<Leave>", lambda e, r=rb: r.config(fg=Theme.TEXT_WHITE))

        tk.Frame(inner, bg=Theme.BG_FRAME, height=6).pack()

    # ── 3. Donanım Paneli ───────────────────────────────────────
    def _create_hardware_panel(self) -> None:
        panel = tk.Frame(self.root, bg=Theme.BG_PANEL)
        panel.pack(fill="x", padx=28, pady=(0, 12))

        gpu_ok    = Config.USE_GPU
        hw_color  = Theme.TEXT_GREEN if gpu_ok else Theme.TEXT_WARN
        hw_status = "AKTİF" if gpu_ok else "PASİF"
        hw_text   = f"Donanım Hızlandırma ({hw_status}) : {'AKTİF' if gpu_ok else 'PASİF'}"

        tk.Label(
            panel,
            text=hw_text,
            font=Theme.F_GPU,
            bg=Theme.BG_PANEL,
            fg=hw_color,
        ).pack(pady=(8, 2))

        if gpu_ok and Config.GPU_INFO:
            gpu_short = Config.GPU_INFO[:50]
            tk.Label(
                panel,
                text=f"[ {gpu_short} ]",
                font=Theme.F_SMALL,
                bg=Theme.BG_PANEL,
                fg=Theme.TEXT_GRAY,
            ).pack(pady=(0, 4))

        # Heartbeat göstergesi
        tk.Label(
            panel,
            text="[+] Heartbeat & Skill Motoru : AKTİF",
            font=Theme.F_SMALL,
            bg=Theme.BG_PANEL,
            fg=Theme.GREEN_DIM,
        ).pack(pady=(0, 8))

    # ── 4. Mod Seçimi ───────────────────────────────────────────
    def _create_mode_section(self) -> None:
        tk.Label(
            self.root,
            text="Çalışma Modu Seçiniz",
            font=Theme.F_LABEL,
            bg=Theme.BG_MAIN,
            fg=Theme.TEXT_GRAY,
        ).pack(pady=(2, 8))

        modes = [
            ("►  ONLINE MOD  (Gemini 1.5)", "online"),
            ("►  LOCAL MOD   (Ollama Llama3)", "ollama"),
        ]
        for text, mode in modes:
            self._create_mode_button(text, mode)

    def _create_mode_button(self, text: str, mode: str) -> None:
        btn = tk.Button(
            self.root,
            text=text,
            font=Theme.F_BUTTON,
            bg=Theme.BG_BUTTON,
            fg=Theme.TEXT_WHITE,
            activebackground=Theme.RED,
            activeforeground=Theme.TEXT_WHITE,
            bd=0,
            highlightthickness=1,
            highlightbackground=Theme.BORDER,
            highlightcolor=Theme.TEXT_GREEN,
            cursor="hand2",
            width=30,
            height=2,
            command=lambda m=mode: self._pre_launch_check(m),
        )
        btn.pack(pady=5)

        btn.bind("<Enter>", lambda e, b=btn: b.config(bg=Theme.BG_HOVER, fg=Theme.TEXT_GREEN))
        btn.bind("<Leave>", lambda e, b=btn: b.config(bg=Theme.BG_BUTTON, fg=Theme.TEXT_WHITE))

    # ── 5. Durum Çubuğu ─────────────────────────────────────────
    def _create_status_bar(self) -> None:
        self.status_var = tk.StringVar(
            value="Sistem hazır.  Lütfen erişim yetkisi seçin."
        )
        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=Theme.F_STATUS,
            bg=Theme.BG_STATUS,
            fg=Theme.TEXT_DIM,
            anchor="w",
            padx=10,
            height=2,
        ).pack(side="bottom", fill="x")

    # ───────────────────────────────────────────────────────────
    # İŞ MANTIĞI
    # ───────────────────────────────────────────────────────────
    def _check_ollama_service(self) -> bool:
        for url in self.OLLAMA_URLS:
            try:
                r = requests.get(url, timeout=self.OLLAMA_TIMEOUT)
                if r.status_code == 200:
                    logger.info(f"Ollama servisi aktif: {url}")
                    return True
            except requests.exceptions.RequestException:
                continue
        logger.warning("Ollama servisine ulaşılamadı")
        return False

    def _pre_launch_check(self, mode: str) -> None:
        """Başlatma öncesi kontroller (erişim seviyesi + ollama + api key)"""
        if start_lotus_system is None:
            messagebox.showerror(
                "Sistem Başlatılamadı",
                f"lotus_system.py modülü yüklenemedi!\n\n"
                f"Hata: {import_error_message}\n\n"
                "Çözüm:\n"
                "  conda activate lts\n"
                "  pip install -r requirements.txt",
            )
            logger.error(f"Başlatma başarısız: {import_error_message}")
            return

        # Seçilen erişim seviyesini Config üzerinden sisteme aktar
        access = self._access_var.get()
        Config.ACCESS_LEVEL = access
        logger.info(f"Erişim seviyesi: {access}")

        # Provider modunu ayarla
        Config.set_provider_mode(mode)

        # API key kontrolü
        if not Config.validate_critical_settings():
            if mode == "online":
                messagebox.showerror(
                    "Yapılandırma Hatası",
                    "Online mod için Google API Anahtarı bulunamadı!\n\n"
                    "Çözüm:\n"
                    "  1. .env dosyasını açın\n"
                    "  2. GOOGLE_API_KEY değerini girin\n"
                    "  3. Launcher'ı yeniden başlatın",
                )
                logger.error("API anahtarı eksik")
                return

        # Ollama kontrolü
        if mode == "ollama":
            self.status_var.set("Ollama servisi kontrol ediliyor...")
            self.root.update()

            if not self._check_ollama_service():
                devam = messagebox.askyesno(
                    "Servis Uyarısı",
                    "Ollama servisi çalışmıyor!\n\n"
                    "Local mod için Ollama aktif olmalıdır.\n"
                    "Terminal'de: ollama serve\n\n"
                    "Yine de devam etmek istiyor musunuz?",
                    icon="warning",
                )
                if not devam:
                    self.status_var.set("Başlatma iptal edildi.")
                    logger.info("Kullanıcı başlatmayı iptal etti")
                    return

        self._launch_system(mode, access)

    def _launch_system(self, mode: str, access: str) -> None:
        """Sistemi başlat"""
        self.status_var.set(
            f"{mode.upper()} modu yükleniyor...  Yetki: {access.upper()}"
        )
        self.root.update()

        self._print_banner(mode, access)
        self.root.destroy()

        try:
            logger.info(f"Sistem başlatılıyor — mod: {mode}  erişim: {access}")
            start_lotus_system(mode)
        except Exception as e:
            error_msg = f"Sistem çalışma hatası: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            print(f"\n{Colors.FAIL}╔{'═' * 39}╗{Colors.ENDC}")
            print(f"{Colors.FAIL}║   KRiTiK SiSTEM HATASI{' ' * 17}║{Colors.ENDC}")
            print(f"{Colors.FAIL}╚{'═' * 39}╝{Colors.ENDC}")
            print(f"\n{Colors.WARNING}Hata : {e}{Colors.ENDC}")
            print(f"{Colors.CYAN}Log  : {LOG_FILE}{Colors.ENDC}\n")
            input("Cikmak icin Enter tusuna basin...")

    # ───────────────────────────────────────────────────────────
    # TERMINAL BANNER
    # ───────────────────────────────────────────────────────────
    def _print_banner(self, mode: str, access: str) -> None:
        os.system('cls' if os.name == 'nt' else 'clear')

        gpu_info = Config.GPU_INFO if Config.USE_GPU else "CPU (Standart)"
        access_desc = AccessLevel.DESCRIPTIONS.get(access, access)

        print(f"\n{Colors.OKGREEN}{'═' * 62}{Colors.ENDC}")
        print(f"{Colors.BOLD}  LotusAI SİSTEMİ BAŞLATILIYOR{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{'═' * 62}{Colors.ENDC}")
        print(f"{Colors.CYAN}  Sürüm     : {Colors.ENDC}{Config.VERSION}")
        print(f"{Colors.CYAN}  Mod       : {Colors.ENDC}{mode.upper()}")
        print(f"{Colors.CYAN}  Donanım   : {Colors.ENDC}{gpu_info}")
        print(f"{Colors.CYAN}  Erişim    : {Colors.ENDC}{access.upper()}  ({access_desc})")
        print(f"{Colors.CYAN}  Heartbeat : {Colors.ENDC}Proaktif Skill Motoru AKTİF")
        print(f"{Colors.OKGREEN}{'═' * 62}{Colors.ENDC}\n")

    # ───────────────────────────────────────────────────────────
    # KAPATMA
    # ───────────────────────────────────────────────────────────
    def _on_closing(self) -> None:
        logger.info("Launcher kapatıldı")
        self.root.destroy()
        sys.exit(0)


# ═══════════════════════════════════════════════════════════════
# ANA PROGRAM
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    try:
        root = tk.Tk()
        LauncherApp(root)
        root.mainloop()
    except Exception as e:
        error_msg = f"Launcher GUI başlatılamadı: {e}"
        logger.critical(error_msg)
        print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")

        if start_lotus_system:
            print(f"\n{Colors.WARNING}GUI hatası! Terminal modunda başlatılıyor...{Colors.ENDC}")
            start_lotus_system("online")
        else:
            print(f"\n{Colors.FAIL}Sistem başlatılamadı. {LOG_FILE} dosyasını kontrol edin.{Colors.ENDC}")
            sys.exit(1)


if __name__ == "__main__":
    main()
