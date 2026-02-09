import tkinter as tk
from tkinter import messagebox, ttk, font as tkfont
import codecs
import locale
import sys
import os
import traceback
import logging
import threading
import requests
from config import Config

# --- LOGLAMA YAPILANDIRMASI (Config ile Uyumlu) ---
LOG_FILE = Config.LOG_DIR / "launcher.log"
logging.basicConfig(
    filename=LOG_FILE, 
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LotusLauncher")


def ensure_turkish_locale():
    """Türkçe karakter desteği için uygun locale ayarla."""
    candidates = ["tr_TR.UTF-8", "tr_TR.utf8", "tr_TR"]
    for candidate in candidates:
        try:
            locale.setlocale(locale.LC_ALL, candidate)
            os.environ["LANG"] = candidate
            os.environ["LC_ALL"] = candidate
            return candidate
        except locale.Error:
            continue
    return None


def normalize_text(text):
    """Kaçış dizilerini (\\uXXXX) gerçek Unicode karakterlerine çevir."""
    if isinstance(text, str) and ("\\u" in text or "\\U" in text):
        try:
            return codecs.decode(text, "unicode_escape")
        except Exception:
            return text
    return text

# --- SİSTEM BAŞLATICI FONKSİYONU ---
start_lotus_system = None
import_error_message = ""

try:
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

        ensure_turkish_locale()
        self._configure_tk_encoding()
        
        # --- 4K / HIDPI ÖLÇEKLEME AYARLARI ---
        # WSL üzerinde otomatik DPI algılama bazen başarısız olabilir.
        # Asus Zenbook 4K %250 ölçek için manuel çarpan: 2.5
        self.SCALE_FACTOR = 2.5
        
        # Tkinter iç ölçeklendirmesini ayarla (Yazı tipleri ve widget'lar için)
        # Bu, Linux/WSL ortamında widget'ların büyümesini sağlar.
        try:
            self.root.tk.call('tk', 'scaling', self.SCALE_FACTOR)
        except:
            pass
            
        self.root.title(f"{Config.PROJECT_NAME} v{Config.VERSION} - Launcher")
        self.ui_font_family = self._select_font_family()
        self._set_default_font(self.ui_font_family)
        self.t = normalize_text
        
        # Temel Boyutlar (Ölçeklenmemiş)
        base_width = 500
        base_height = 550
        
        # Ölçeklenmiş Boyutlar (4K Ekranda düzgün görünmesi için çarpıyoruz)
        self.window_width = int(base_width * self.SCALE_FACTOR)
        self.window_height = int(base_height * self.SCALE_FACTOR)
        
        # Ekran boyutlarını al
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Ortala
        center_x = int(screen_width/2 - self.window_width/2)
        center_y = int(screen_height/2 - self.window_height/2)
        
        self.root.geometry(f'{self.window_width}x{self.window_height}+{center_x}+{center_y}')
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.setup_ui()

    def _configure_tk_encoding(self):
        """Tk/Tcl tarafında UTF-8 kodlamasını kullan."""
        try:
            self.root.tk.call("encoding", "system", "utf-8")
        except tk.TclError:
            pass

    def _select_font_family(self):
        """Türkçe karakterleri sorunsuz gösterebilecek bir font aile adı seç."""
        preferred_fonts = ["Noto Sans", "DejaVu Sans", "Arial", "Liberation Sans", "Segoe UI"]
        available_fonts = {name.lower(): name for name in tkfont.families(self.root)}
        for font_name in preferred_fonts:
            actual = available_fonts.get(font_name.lower())
            if actual:
                return actual
        return tkfont.nametofont("TkDefaultFont").actual("family")

    def _set_default_font(self, font_family):
        """Türkçe karakterleri sorunsuz gösterebilecek bir varsayılan font seç."""
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family=font_family)

    def setup_ui(self):
        """Arayüz elemanlarını profesyonel bir görünümle oluşturur."""
        
        # Font boyutlarını DPI ölçeğine göre çok abartmamak için 
        # tk scaling komutu zaten fontları büyütür, bu yüzden 
        # font puanlarını (size) orijinal tutuyoruz veya hafif revize ediyoruz.
        
        # Başlık ve Versiyon
        tk.Label(self.root, text=self.t(Config.PROJECT_NAME.upper()), font=(self.ui_font_family, 36, "bold"), 
                 bg="#1a1a2e", fg="#e94560").pack(pady=(int(30*self.SCALE_FACTOR/2), 0))
        
        tk.Label(self.root, text=self.t(f"AI Operating System v{Config.VERSION}"), 
                 font=(self.ui_font_family, 10), bg="#1a1a2e", fg="#95a5a6").pack(pady=(0, int(20*self.SCALE_FACTOR/2)))

        # Bilgi Paneli (Frame)
        info_frame = tk.Frame(self.root, bg="#16213e", bd=1, relief="flat")
        # Paddingleri de ölçeğe göre biraz rahatlatıyoruz
        info_frame.pack(fill="x", padx=int(40*self.SCALE_FACTOR/2), pady=int(10*self.SCALE_FACTOR/2))

        gpu_status = "AKTİF" if Config.USE_GPU else "PASİF"
        gpu_color = "#27ae60" if Config.USE_GPU else "#f39c12"
        
        tk.Label(info_frame, text=self.t(f"Donanım Hızlandırma: {gpu_status}"), font=(self.ui_font_family, 9), 
                 bg="#16213e", fg=gpu_color).pack(pady=5)
        
        if Config.USE_GPU:
            tk.Label(info_frame, text=self.t(f"GPU: {Config.GPU_INFO}"), font=(self.ui_font_family, 8, "italic"), 
                     bg="#16213e", fg="#bdc3c7").pack(pady=(0, 5))

        # Mod Seçimi Alanı
        tk.Label(self.root, text=self.t("Çalışma Modunu Seçiniz"), font=(self.ui_font_family, 11, "bold"), 
                 bg="#1a1a2e", fg="#ffffff").pack(pady=(int(20*self.SCALE_FACTOR/2), int(10*self.SCALE_FACTOR/2)))

        # Butonlar
        self.btn_online = self.create_styled_button(self.t("🌐 ONLINE (Gemini Pro)"), "#0f3460", "online")
        # Buton arası boşlukları ayarla
        self.btn_online.pack(pady=int(10*self.SCALE_FACTOR/3))

        self.btn_local = self.create_styled_button(self.t("💻 LOCAL (Ollama/Llama 3.1)"), "#16213e", "local")
        self.btn_local.pack(pady=int(10*self.SCALE_FACTOR/3))

        # Durum Göstergesi
        self.status_var = tk.StringVar(value=self.t("Sistem Başlatılmaya Hazır"))
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=(self.ui_font_family, 9), 
                                     bg="#0f3460", fg="#bdc3c7", height=2)
        self.status_label.pack(side="bottom", fill="x")

    def create_styled_button(self, text, color, mode):
        """Özel tasarım ve hover efektli buton."""
        # Buton genişliği ve yüksekliği karakter bazlıdır, pixel bazlı DEĞİLDİR.
        # Bu yüzden width/height değerlerini scale factor ile çarpmıyoruz, 
        # çünkü tk scaling zaten fontu büyüttüğü için buton otomatik büyüyecek.
        btn = tk.Button(
            self.root, text=text, bg=color, fg="white", 
            font=(self.ui_font_family, 11, "bold"), width=30, height=2, 
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
        
        self.print_banner(mode)
        
        self.root.destroy()
        
        try:
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
    
    # WSLg veya X Server ikon desteği
    # if os.path.exists("static/favicon.ico"): root.iconbitmap("static/favicon.ico")
    
    app = LauncherApp(root)
    root.mainloop()
