import tkinter as tk
from tkinter import messagebox
import sys
import os
import traceback
import logging
import requests
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from config import Config
except ImportError:
    # Fallback Config if not found
    class Config:
        PROJECT_NAME = "LotusAI"
        VERSION = "2.5.3"
        LOG_DIR = Path("logs")
        USE_GPU = False
        GPU_INFO = "N/A"
        @staticmethod
        def set_provider_mode(mode): pass
        @staticmethod
        def validate_critical_settings(): return True

# --- LOGLAMA YAPILANDIRMASI ---
# Klasör yoksa oluştur
LOG_DIR = getattr(Config, "LOG_DIR", Path("logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "launcher.log"

logging.basicConfig(
    filename=LOG_FILE, 
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("LotusLauncher")

# --- SİSTEM BAŞLATICI FONKSİYONU ---
start_lotus_system = None
import_error_message = ""

try:
    # lotus_system.py dosyasındaki start_lotus_system fonksiyonunu içeri aktar
    from lotus_system import start_lotus_system
except ImportError as e:
    import_error_message = f"Bağımlılık Eksik: {str(e)}"
    logger.error(import_error_message)
except Exception as e:
    import_error_message = f"Sistem Dosyası Hatası: {str(e)}"
    logger.error(f"{import_error_message}\n{traceback.format_exc()}")

class LauncherApp:
    """
    LotusAI Görsel Başlatıcı (Launcher).
    v2.5.3 - 4K/HIDPI ve Çoklu Mod Desteği.
    """
    def __init__(self, root):
        self.root = root
        
        # --- 4K / HIDPI ÖLÇEKLEME AYARLARI ---
        # Windows ve Linux üzerinde daha keskin görünüm için
        self.SCALE_FACTOR = 1.5 
        
        try:
            # Tkinter ölçeklendirme komutu
            self.root.tk.call('tk', 'scaling', self.SCALE_FACTOR)
        except:
            pass
            
        self.root.title(f"{Config.PROJECT_NAME} v{Config.VERSION} - Launcher")
        
        # Temel Boyutlar (Ölçeklendirilmiş)
        base_width = 450
        base_height = 500
        
        self.window_width = int(base_width * self.SCALE_FACTOR)
        self.window_height = int(base_height * self.SCALE_FACTOR)
        
        # Ekranın ortasına yerleştir
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        center_x = int(screen_width/2 - self.window_width/2)
        center_y = int(screen_height/2 - self.window_height/2)
        
        self.root.geometry(f'{self.window_width}x{self.window_height}+{center_x}+{center_y}')
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)
        
        # Pencere kapatma protokolü
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.setup_ui()

    def setup_ui(self):
        """Arayüz bileşenlerini oluşturur."""
        
        # Başlık Bölümü
        tk.Label(self.root, text=Config.PROJECT_NAME.upper(), font=("Segoe UI", 32, "bold"), 
                 bg="#1a1a2e", fg="#e94560").pack(pady=(40, 5))
        
        tk.Label(self.root, text=f"AI Operating System v{Config.VERSION}", 
                 font=("Segoe UI", 10), bg="#1a1a2e", fg="#95a5a6").pack(pady=(0, 20))

        # Donanım Bilgi Paneli
        info_frame = tk.Frame(self.root, bg="#16213e", bd=1, relief="flat")
        info_frame.pack(fill="x", padx=40, pady=10)

        gpu_status = "AKTİF" if Config.USE_GPU else "PASİF"
        gpu_color = "#27ae60" if Config.USE_GPU else "#f39c12"
        
        tk.Label(info_frame, text=f"Donanım Hızlandırma: {gpu_status}", font=("Segoe UI", 10, "bold"), 
                 bg="#16213e", fg=gpu_color).pack(pady=10)
        
        if Config.USE_GPU:
            gpu_desc = f"GPU: {Config.GPU_INFO}"
            if len(gpu_desc) > 40: gpu_desc = gpu_desc[:37] + "..."
            tk.Label(info_frame, text=gpu_desc, font=("Segoe UI", 8, "italic"), 
                     bg="#16213e", fg="#bdc3c7").pack(pady=(0, 10))

        # Mod Seçimi Etiketi
        tk.Label(self.root, text="Çalışma Modunu Seçiniz", font=("Segoe UI", 11, "bold"), 
                 bg="#1a1a2e", fg="#ffffff").pack(pady=(30, 15))

        # Butonlar
        # ONLINE MOD
        self.btn_online = self.create_styled_button("🌐 ONLINE (Gemini Pro)", "#0f3460", "online")
        self.btn_online.pack(pady=10)

        # LOCAL MOD (Ollama)
        self.btn_local = self.create_styled_button("💻 LOCAL (Ollama/Llama 3.1)", "#16213e", "ollama")
        self.btn_local.pack(pady=10)

        # Alt Durum Çubuğu
        self.status_var = tk.StringVar(value="Sistem Başlatılmaya Hazır")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9), 
                                     bg="#0f3460", fg="#bdc3c7", height=2)
        self.status_label.pack(side="bottom", fill="x")

    def create_styled_button(self, text, color, mode):
        """Hover efektli stilize buton oluşturur."""
        btn = tk.Button(
            self.root, text=text, bg=color, fg="white", 
            font=("Segoe UI", 11, "bold"), width=30, height=2, 
            bd=0, cursor="hand2", activebackground="#e94560", activeforeground="white",
            command=lambda: self.pre_launch_check(mode)
        )
        # Mouse üzerine gelince renk değiştir
        btn.bind("<Enter>", lambda e: btn.config(bg="#e94560"))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    def check_local_engine(self):
        """Ollama yerel servisinin aktif olup olmadığını kontrol eder."""
        urls = [
            "http://127.0.0.1:11434/api/tags",
            "http://localhost:11434/api/tags"
        ]
        for url in urls:
            try:
                response = requests.get(url, timeout=1.5)
                if response.status_code == 200:
                    return True
            except:
                continue
        return False

    def pre_launch_check(self, mode):
        """Sistemi başlatmadan önceki son kontroller."""
        if start_lotus_system is None:
            messagebox.showerror("Kritik Hata", f"lotus_system.py yüklenemedi!\nDetay: {import_error_message}")
            return

        # 1. Config Yapılandırması
        Config.set_provider_mode(mode)
        if not Config.validate_critical_settings():
            if mode == "online":
                messagebox.showerror("Eksik Ayar", "Online mod için API Anahtarı bulunamadı!\nLütfen .env dosyasını kontrol edin.")
                return

        # 2. Yerel Mod Kontrolü
        if mode == "ollama":
            self.status_var.set("Ollama servisi kontrol ediliyor...")
            self.root.update()
            
            if not self.check_local_engine():
                msg = "Ollama servisine (127.0.0.1:11434) ulaşılamadı!\n\nServisin arka planda çalıştığından emin olun.\nYine de devam etmek istiyor musunuz?"
                confirm = messagebox.askyesno("Servis Uyarısı", msg, icon='warning')
                if not confirm:
                    self.status_var.set("Başlatma iptal edildi.")
                    return

        # 3. Başlatma
        self.launch_system(mode)

    def launch_system(self, mode):
        """Launcher'ı kapatır ve ana LotusAI motorunu terminale bırakır."""
        self.status_var.set(f"LotusAI {mode.upper()} modu yükleniyor...")
        self.root.update()
        
        # Terminale şık bir yükleme banner'ı yazdır
        self.print_banner(mode)
        
        # GUI'yi kapat
        self.root.destroy()
        
        # Motoru başlat
        try:
            start_lotus_system(mode)
        except Exception as e:
            logger.error(f"Sistem Çalışma Hatası: {str(e)}\n{traceback.format_exc()}")
            print(f"\n{Colors.FAIL}[!] KRİTİK SİSTEM HATASI: {e}{Colors.ENDC}")
            input("\nDetaylar için logs/launcher.log dosyasını inceleyin. Çıkmak için Enter...")

    def print_banner(self, mode):
        """Terminal çıktısını temiz ve profesyonel hale getirir."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "═"*60)
        print(f" 🚀 {Config.PROJECT_NAME} SİSTEMİ BAŞLATILIYOR")
        print(f" 🛠  Sürüm     : {Config.VERSION}")
        print(f" 🧠 Mod       : {mode.upper()}")
        print(f" 💻 Donanım   : {Config.GPU_INFO if Config.USE_GPU else 'CPU (Standart Mod)'}")
        print("═"*60 + "\n")

    def on_closing(self):
        """Launcher'dan güvenli çıkış."""
        self.root.destroy()
        sys.exit()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = LauncherApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Launcher GUI başlatılamadı: {e}")
        # Hata durumunda terminalden direkt başlatmayı dene
        if start_lotus_system:
            print("Launcher hatası nedeniyle sistem direkt başlatılıyor...")
            start_lotus_system("online")