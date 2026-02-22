"""
LotusAI Launcher - Ana Başlatıcı
Versiyon: 2.5.3
Python: 3.11+
Açıklama: LotusAI sistemini Online veya Local modda başlatır
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
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'

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
    # Config bulunamazsa varsayılan ayarlar
    @dataclass
    class Config:
        PROJECT_NAME: str = "LotusAI"
        VERSION: str = "2.5.3"
        LOG_DIR: Path = Path("logs")
        USE_GPU: bool = False
        GPU_INFO: str = "N/A"
        
        @staticmethod
        def set_provider_mode(mode: str) -> None:
            """Sağlayıcı modunu ayarla"""
            pass
        
        @staticmethod
        def validate_critical_settings() -> bool:
            """Kritik ayarları doğrula"""
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
    import_error_message = f"Bağımlılık eksik: {str(e)}\n\nLütfen şu komutu çalıştırın:\nconda activate lts\npip install -r requirements.txt"
    logger.error(import_error_message)
except Exception as e:
    import_error_message = f"Sistem dosyası hatası: {str(e)}"
    logger.error(f"{import_error_message}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════
# UI TEMA AYARLARI
# ═══════════════════════════════════════════════════════════════
class Theme:
    """Launcher UI renk paleti"""
    BG_DARK = "#1a1a2e"
    BG_MEDIUM = "#16213e"
    BG_LIGHT = "#0f3460"
    ACCENT = "#e94560"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#95a5a6"
    TEXT_MUTED = "#bdc3c7"
    SUCCESS = "#27ae60"
    WARNING = "#f39c12"


# ═══════════════════════════════════════════════════════════════
# LAUNCHER UYGULAMASI
# ═══════════════════════════════════════════════════════════════
class LauncherApp:
    """
    LotusAI Görsel Başlatıcı
    
    Özellikler:
    - 4K/HiDPI desteği
    - Online (Gemini) ve Local (Ollama) mod
    - Servis sağlık kontrolü
    - Kullanıcı dostu hata mesajları
    """
    
    # UI Boyutları
    BASE_WIDTH = 450
    BASE_HEIGHT = 500
    SCALE_FACTOR = 1.5
    
    # Ollama Servis Ayarları
    OLLAMA_URLS = [
        "http://127.0.0.1:11434/api/tags",
        "http://localhost:11434/api/tags"
    ]
    OLLAMA_TIMEOUT = 2.0  # saniye
    
    def __init__(self, root: tk.Tk) -> None:
        """Launcher başlatıcı"""
        self.root = root
        self._setup_window()
        self._setup_ui()
        logger.info("Launcher başlatıldı")
    
    def _setup_window(self) -> None:
        """Pencere ayarlarını yapılandır"""
        # HiDPI ölçekleme
        try:
            self.root.tk.call('tk', 'scaling', self.SCALE_FACTOR)
        except Exception as e:
            logger.warning(f"Ölçekleme ayarlanamadı: {e}")
        
        # Pencere özellikleri
        self.root.title(f"{Config.PROJECT_NAME} v{Config.VERSION} - Launcher")
        
        # Boyutlandırma
        width = int(self.BASE_WIDTH * self.SCALE_FACTOR)
        height = int(self.BASE_HEIGHT * self.SCALE_FACTOR)
        
        # Ekran merkezleme
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.configure(bg=Theme.BG_DARK)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_ui(self) -> None:
        """UI bileşenlerini oluştur"""
        # Başlık
        self._create_header()
        
        # Donanım bilgi paneli
        self._create_info_panel()
        
        # Mod seçim başlığı
        tk.Label(
            self.root,
            text="Çalışma Modunu Seçiniz",
            font=("Segoe UI", 11, "bold"),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_PRIMARY
        ).pack(pady=(30, 15))
        
        # Mod butonları
        self._create_mode_buttons()
        
        # Durum çubuğu
        self._create_status_bar()
    
    def _create_header(self) -> None:
        """Başlık bölümünü oluştur"""
        tk.Label(
            self.root,
            text=Config.PROJECT_NAME.upper(),
            font=("Segoe UI", 32, "bold"),
            bg=Theme.BG_DARK,
            fg=Theme.ACCENT
        ).pack(pady=(40, 5))
        
        tk.Label(
            self.root,
            text=f"AI Operating System v{Config.VERSION}",
            font=("Segoe UI", 10),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY
        ).pack(pady=(0, 20))
    
    def _create_info_panel(self) -> None:
        """Donanım bilgi panelini oluştur"""
        frame = tk.Frame(self.root, bg=Theme.BG_MEDIUM, bd=1, relief="flat")
        frame.pack(fill="x", padx=40, pady=10)

        # GPU durumu
        gpu_status = "AKTİF" if Config.USE_GPU else "PASİF"
        gpu_color = Theme.SUCCESS if Config.USE_GPU else Theme.WARNING

        tk.Label(
            frame,
            text=f"Donanım Hızlandırma: {gpu_status}",
            font=("Segoe UI", 10, "bold"),
            bg=Theme.BG_MEDIUM,
            fg=gpu_color
        ).pack(pady=(10, 2))

        # GPU detayı
        if Config.USE_GPU:
            gpu_text = Config.GPU_INFO
            if len(gpu_text) > 45:
                gpu_text = gpu_text[:42] + "..."

            tk.Label(
                frame,
                text=f"GPU: {gpu_text}",
                font=("Segoe UI", 8, "italic"),
                bg=Theme.BG_MEDIUM,
                fg=Theme.TEXT_MUTED
            ).pack(pady=(0, 4))

        # Heartbeat & Skill sistemi göstergesi
        tk.Label(
            frame,
            text="💓 Proaktif Heartbeat + Skill Sistemi: AKTİF",
            font=("Segoe UI", 9, "bold"),
            bg=Theme.BG_MEDIUM,
            fg=Theme.SUCCESS
        ).pack(pady=(2, 6))
    
    def _create_mode_buttons(self) -> None:
        """Mod seçim butonlarını oluştur"""
        buttons = [
            ("🌐 ONLINE (Gemini Pro)", Theme.BG_LIGHT, "online"),
            ("💻 LOCAL (Ollama/Llama 3.1)", Theme.BG_MEDIUM, "ollama")
        ]
        
        for text, color, mode in buttons:
            btn = self._create_styled_button(text, color, mode)
            btn.pack(pady=10)
    
    def _create_styled_button(self, text: str, color: str, mode: str) -> tk.Button:
        """Hover efektli stilize buton oluştur"""
        btn = tk.Button(
            self.root,
            text=text,
            bg=color,
            fg=Theme.TEXT_PRIMARY,
            font=("Segoe UI", 11, "bold"),
            width=30,
            height=2,
            bd=0,
            cursor="hand2",
            activebackground=Theme.ACCENT,
            activeforeground=Theme.TEXT_PRIMARY,
            command=lambda: self._pre_launch_check(mode)
        )
        
        # Hover efektleri
        btn.bind("<Enter>", lambda e: btn.config(bg=Theme.ACCENT))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        
        return btn
    
    def _create_status_bar(self) -> None:
        """Alt durum çubuğunu oluştur"""
        self.status_var = tk.StringVar(value="Sistem Başlatılmaya Hazır")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg=Theme.BG_LIGHT,
            fg=Theme.TEXT_MUTED,
            height=2
        )
        self.status_label.pack(side="bottom", fill="x")
    
    def _check_ollama_service(self) -> bool:
        """Ollama servisinin çalışıp çalışmadığını kontrol et"""
        for url in self.OLLAMA_URLS:
            try:
                response = requests.get(url, timeout=self.OLLAMA_TIMEOUT)
                if response.status_code == 200:
                    logger.info(f"Ollama servisi aktif: {url}")
                    return True
            except requests.exceptions.RequestException:
                continue
        
        logger.warning("Ollama servisine ulaşılamadı")
        return False
    
    def _pre_launch_check(self, mode: str) -> None:
        """Başlatma öncesi kontroller"""
        if start_lotus_system is None:
            error_title = "Sistem Başlatılamadı"
            error_msg = (
                "lotus_system.py modülü yüklenemedi!\n\n"
                f"Hata: {import_error_message}\n\n"
                "Çözüm:\n"
                "1. Terminal'de: conda activate lts\n"
                "2. Bağımlılıkları yükleyin: pip install -r requirements.txt"
            )
            messagebox.showerror(error_title, error_msg)
            logger.error(f"Başlatma başarısız: {import_error_message}")
            return
        
        # Config ayarla
        Config.set_provider_mode(mode)
        
        # Kritik ayar kontrolü
        if not Config.validate_critical_settings():
            if mode == "online":
                messagebox.showerror(
                    "Yapılandırma Hatası",
                    "Online mod için Google API Anahtarı bulunamadı!\n\n"
                    "Çözüm:\n"
                    "1. .env dosyasını açın\n"
                    "2. GOOGLE_API_KEY değişkenini ekleyin\n"
                    "3. Launcher'ı yeniden başlatın"
                )
                logger.error("API anahtarı eksik")
                return
        
        # Ollama kontrolü (Local mod için)
        if mode == "ollama":
            self.status_var.set("Ollama servisi kontrol ediliyor...")
            self.root.update()
            
            if not self._check_ollama_service():
                response = messagebox.askyesno(
                    "Servis Uyarısı",
                    "⚠️ Ollama servisi çalışmıyor!\n\n"
                    "Local mod için Ollama'nın aktif olması gerekir.\n"
                    "Terminal'de şu komutu çalıştırın:\n"
                    "  ollama serve\n\n"
                    "Yine de devam etmek istiyor musunuz?",
                    icon='warning'
                )
                if not response:
                    self.status_var.set("Başlatma iptal edildi")
                    logger.info("Kullanıcı başlatmayı iptal etti")
                    return
        
        # Sistemi başlat
        self._launch_system(mode)
    
    def _launch_system(self, mode: str) -> None:
        """LotusAI sistemini başlat"""
        self.status_var.set(f"{mode.upper()} modu yükleniyor...")
        self.root.update()
        
        # Banner yazdır
        self._print_banner(mode)
        
        # GUI'yi kapat
        self.root.destroy()
        
        # Motoru başlat
        try:
            logger.info(f"Sistem {mode} modunda başlatılıyor")
            start_lotus_system(mode)
        except Exception as e:
            error_msg = f"Sistem çalışma hatası: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            print(f"\n{Colors.FAIL}╔═══════════════════════════════════════╗{Colors.ENDC}")
            print(f"{Colors.FAIL}║   KRİTİK SİSTEM HATASI                ║{Colors.ENDC}")
            print(f"{Colors.FAIL}╚═══════════════════════════════════════╝{Colors.ENDC}")
            print(f"\n{Colors.WARNING}Hata: {e}{Colors.ENDC}")
            print(f"\n{Colors.CYAN}Detaylar için şu dosyayı kontrol edin:{Colors.ENDC}")
            print(f"{Colors.OKBLUE}  {LOG_FILE}{Colors.ENDC}\n")
            input("Çıkmak için Enter tuşuna basın...")
    
    def _print_banner(self, mode: str) -> None:
        """Terminal başlangıç banner'ı"""
        os.system('cls' if os.name == 'nt' else 'clear')

        gpu_info = Config.GPU_INFO if Config.USE_GPU else "CPU (Standart)"

        print(f"\n{Colors.OKGREEN}{'═' * 60}{Colors.ENDC}")
        print(f"{Colors.BOLD} 🚀 {Config.PROJECT_NAME} SİSTEMİ BAŞLATILIYOR{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{'═' * 60}{Colors.ENDC}")
        print(f"{Colors.CYAN} 🛠  Sürüm    :{Colors.ENDC} {Config.VERSION}")
        print(f"{Colors.CYAN} 🧠 Mod      :{Colors.ENDC} {mode.upper()}")
        print(f"{Colors.CYAN} 💻 Donanım  :{Colors.ENDC} {gpu_info}")
        print(f"{Colors.CYAN} 💓 Heartbeat:{Colors.ENDC} Proaktif Skill Motoru AKTİF")
        print(f"{Colors.OKGREEN}{'═' * 60}{Colors.ENDC}\n")
    
    def _on_closing(self) -> None:
        """Launcher kapatma işlemi"""
        logger.info("Launcher kapatıldı")
        self.root.destroy()
        sys.exit(0)


# ═══════════════════════════════════════════════════════════════
# ANA PROGRAM
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    """Ana başlatıcı fonksiyon"""
    try:
        root = tk.Tk()
        app = LauncherApp(root)
        root.mainloop()
    except Exception as e:
        error_msg = f"Launcher GUI başlatılamadı: {e}"
        logger.critical(error_msg)
        print(f"{Colors.FAIL}{error_msg}{Colors.ENDC}")
        
        # Fallback: Direkt terminal başlatma
        if start_lotus_system:
            print(f"\n{Colors.WARNING}GUI hatası! Sistem terminal modunda başlatılıyor...{Colors.ENDC}")
            start_lotus_system("online")
        else:
            print(f"\n{Colors.FAIL}Sistem başlatılamadı. logs/launcher.log dosyasını kontrol edin.{Colors.ENDC}")
            sys.exit(1)


if __name__ == "__main__":
    main()