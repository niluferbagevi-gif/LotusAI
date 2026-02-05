import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import traceback
import logging
import threading
import requests # Ollama kontrolü için
from config import Config

# --- LOGLAMA YAPILANDIRMASI (Config ile Uyumlu) ---
LOG_FILE = Config.LOG_DIR / "launcher.log"
logging.basicConfig(
    filename=LOG_FILE, 
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LotusLauncher")

# --- SİSTEM BAŞLATICI FONKSİYONU ---
start_lotus_system = None
import_error_message = ""

try:
    # Proje kök dizinini yola ekle
    sys.path.append(os.getcwd())
    from lotus_system import start_lotus_system
except ImportError as e:
    import_error_message = f"Bağımlılık Eksik: {str(e)}"
    logger.error(import_error_message)
except Exception as e:
    import_error_message = f"Sistem Dosyası Hatası: {str(e)}"
    logger.error(f"{import_error_message}\n{traceback.format_exc()}")

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{Config.PROJECT_NAME} v{Config.VERSION} - Launcher")
        self.ui_scale = self.detect_ui_scale()
        self.root.tk.call("tk", "scaling", self.ui_scale)
        
        # Pencere Boyutları ve Konumu
        self.window_width = self.scaled(500)
        self.window_height = self.scaled(550)
        
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - self.window_width/2)
        center_y = int(screen_height/2 - self.window_height/2)
        
        self.root.geometry(f'{self.window_width}x{self.window_height}+{center_x}+{center_y}')
        self.root.configure(bg="#1a1a2e") # Koyu Lacivert/Modern tema
        self.root.resizable(False, False)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.setup_ui()

    def scaled(self, value):
        """Piksel bazlı değerleri ekran ölçeğine göre hesaplar."""
        return max(1, int(value * self.ui_scale))

    def detect_ui_scale(self):
        """Ekran çözünürlüğü ve DPI bilgisine göre UI ölçeğini belirler."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        try:
            dpi = float(self.root.winfo_fpixels("1i"))
        except Exception:
            dpi = 96.0

        dpi_scale = dpi / 96.0
        resolution_scale = min(screen_width / 1920, screen_height / 1080)
        ui_scale = max(1.0, min(2.5, max(dpi_scale, resolution_scale)))

        # 4K + 250% ölçekli cihazlarda (örn. Asus Zenbook Pro Duo) daha okunaklı görünüm
        if screen_width >= 3800 and screen_height >= 2100:
            ui_scale = max(ui_scale, 2.5)

        return ui_scale

    def setup_ui(self):
        """Arayüz elemanlarını profesyonel bir görünümle oluşturur."""
        # Başlık ve Versiyon
        tk.Label(self.root, text=Config.PROJECT_NAME.upper(), font=("Segoe UI", self.scaled(18), "bold"), 
                 bg="#1a1a2e", fg="#e94560").pack(pady=(self.scaled(30), 0))
        
        tk.Label(self.root, text=f"AI Operating System v{Config.VERSION}", 
                 font=("Segoe UI", self.scaled(10)), bg="#1a1a2e", fg="#95a5a6").pack(pady=(0, self.scaled(20)))

        # Bilgi Paneli (Frame)
        info_frame = tk.Frame(self.root, bg="#16213e", bd=1, relief="flat")
        info_frame.pack(fill="x", padx=self.scaled(40), pady=self.scaled(10))

        gpu_status = "AKTİF" if Config.USE_GPU else "PASİF"
        gpu_color = "#27ae60" if Config.USE_GPU else "#f39c12"
        
        tk.Label(info_frame, text=f"Donanım Hızlandırma: {gpu_status}", font=("Segoe UI", self.scaled(9)), 
                 bg="#16213e", fg=gpu_color).pack(pady=self.scaled(5))
        
        if Config.USE_GPU:
            tk.Label(info_frame, text=f"GPU: {Config.GPU_INFO}", font=("Segoe UI", self.scaled(8), "italic"), 
                     bg="#16213e", fg="#bdc3c7").pack(pady=(0, self.scaled(5)))

        # Mod Seçimi Alanı
        tk.Label(self.root, text="Çalışma Modunu Seçiniz", font=("Segoe UI", self.scaled(11), "bold"), 
                 bg="#1a1a2e", fg="#ffffff").pack(pady=(self.scaled(20), self.scaled(10)))

        # Butonlar
        self.btn_online = self.create_styled_button("🌐 ONLINE (Gemini Pro)", "#0f3460", "online")
        self.btn_online.pack(pady=self.scaled(10))

        self.btn_local = self.create_styled_button("💻 LOCAL (Ollama/Llama 3.1)", "#16213e", "local")
        self.btn_local.pack(pady=self.scaled(10))

        # Durum Göstergesi
        self.status_var = tk.StringVar(value="Sistem Başlatılmaya Hazır")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", self.scaled(9)), 
                                     bg="#0f3460", fg="#bdc3c7", height=max(1, int(2 * self.ui_scale / 1.5)))
        self.status_label.pack(side="bottom", fill="x")

    def create_styled_button(self, text, color, mode):
        """Özel tasarım ve hover efektli buton."""
        btn = tk.Button(
            self.root, text=text, bg=color, fg="white", 
            font=("Segoe UI", self.scaled(11), "bold"), width=max(20, int(30 * self.ui_scale / 1.6)), height=max(2, int(2 * self.ui_scale / 1.5)), 
            bd=0, cursor="hand2", activebackground="#e94560", activeforeground="white",
            command=lambda: self.pre_launch_check(mode)
        )
        btn.bind("<Enter>", lambda e: btn.config(bg="#e94560"))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    def check_local_engine(self):
        """Ollama servisinin çalışıp çalışmadığını kontrol eder."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def pre_launch_check(self, mode):
        """Sistemi başlatmadan önce son kontrolleri yapar."""
        if start_lotus_system is None:
            messagebox.showerror("Kritik Hata", f"lotus_system.py yüklenemedi!\nDetay: {import_error_message}")
            return

        # 1. Config Doğrulaması
        Config.set_provider_mode(mode)
        if not Config.validate_critical_settings():
            messagebox.showerror("Eksik Ayar", f"'{mode.upper()}' modu için kritik ayarlar (API Key vb.) eksik!\nLütfen .env dosyanızı kontrol edin.")
            return

        # 2. Yerel Mod Kontrolü
        if mode == "local":
            self.status_var.set("Ollama servisi kontrol ediliyor...")
            self.root.update()
            if not self.check_local_engine():
                messagebox.showwarning("Yerel Servis Hatası", "Ollama servisi bulunamadı! Lütfen yerel yapay zeka sunucusunun çalıştığından emin olun.")
                self.status_var.set("Hata: Ollama çalışmıyor.")
                return

        # 3. Başlatma İşlemi
        self.launch_system(mode)

    def launch_system(self, mode):
        """GUI'yi kapatır ve ana motoru başlatır."""
        self.status_var.set(f"LotusAI {mode.upper()} modu yükleniyor...")
        self.root.update()
        
        # Görsel bir veda çıktı terminale
        self.print_banner(mode)
        
        # Arayüzü kapat
        self.root.destroy()
        
        try:
            # Lotus Ana Sistemini Başlat
            start_lotus_system(mode)
        except Exception as e:
            logger.error(f"Sistem Çalışma Hatası: {str(e)}\n{traceback.format_exc()}")
            print(f"\n[!] SİSTEM DURDURULDU: {e}")
            input("\nDetaylar için logları inceleyin. Çıkmak için Enter...")

    def print_banner(self, mode):
        """Terminal çıktısını profesyonelleştirir."""
        print("\n" + "═"*60)
        print(f" 🚀 {Config.PROJECT_NAME} YÜKLENİYOR")
        print(f" 🛠  Sürüm     : {Config.VERSION}")
        print(f" 🧠 Mod       : {mode.upper()}")
        print(f" 💻 Donanım   : {Config.GPU_INFO if Config.USE_GPU else 'CPU Only'}")
        print("═"*60 + "\n")

    def on_closing(self):
        """Güvenli çıkış kontrolü."""
        if messagebox.askokcancel("Çıkış", "LotusAI Launcher'dan çıkmak istiyor musunuz?"):
            self.root.destroy()
            sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    # Windows'ta ikon desteği (eğer varsa)
    # if os.path.exists("static/favicon.ico"): root.iconbitmap("static/favicon.ico")
    
    app = LauncherApp(root)
    root.mainloop()
