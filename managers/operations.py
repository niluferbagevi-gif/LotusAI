import json
import logging
import threading
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# --- YAPILANDIRMA VE FALLBACK ---
try:
    from config import Config
except ImportError:
    class Config:
        WORK_DIR = os.getcwd()
        STATIC_DIR = Path("static")
        USE_GPU = False

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Operations")

# --- GPU KONTROLÜ (Config Entegreli) ---
HAS_GPU = False
DEVICE = "cpu"
USE_GPU_CONFIG = getattr(Config, "USE_GPU", False)

if USE_GPU_CONFIG:
    try:
        import torch
        if torch.cuda.is_available():
            HAS_GPU = True
            DEVICE = "cuda"
            try:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 OperationsManager GPU Aktif: {gpu_name}")
            except:
                logger.info("🚀 OperationsManager GPU Aktif")
        else:
            logger.info("ℹ️ Operations: Config GPU açık ancak donanım bulunamadı. CPU kullanılacak.")
    except ImportError:
        logger.info("ℹ️ PyTorch yüklü değil, işlemler CPU modunda.")
else:
    logger.info("ℹ️ Operasyon işlemleri CPU modunda (Config ayarı).")

# Paket servis modülünü güvenli şekilde içe aktar
try:
    from managers.delivery import DeliveryManager
except ImportError:
    DeliveryManager = None
    logger.warning("⚠️ DeliveryManager modülü bulunamadı. Paket servis botu devre dışı.")


class OperationsManager:
    """
    LotusAI Saha ve Operasyon Yöneticisi.
    
    Yetenekler:
    - Stok Yönetimi: Ürün girişi, çıkışı ve kritik seviye takibi.
    - Rezervasyon: Kayıt, onaylama, iptal ve WhatsApp entegrasyonu.
    - Akıllı Menü: GPU/AI destekli dinamik öneri sistemi.
    - Paket Servis: DeliveryManager üzerinden bot kontrolü ve durum takibi.
    - Veri Güvenliği: RLock ile eşzamanlılık ve otomatik yedekli çalışma.
    """
    
    def __init__(self):
        # Yollar
        default_work_dir = getattr(Config, "WORK_DIR", os.getcwd())
        self.work_dir = Path(default_work_dir)
        self.db_file = self.work_dir / "lotus_operasyon.json"
        self.menu_file = self.work_dir / "lotus_menu.json"
        self.backup_dir = self.work_dir / "backups" / "operations"
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Dizin oluşturma hatası: {e}")
        
        # Çoklu ajan erişimi için Reentrant Lock
        self.lock = threading.RLock()
        
        # Donanım Bilgisi
        self.device = DEVICE
        self.has_gpu = HAS_GPU

        # Servisler
        self.delivery_manager = None
        self._init_delivery()

        # Başlatma
        self._init_databases()
        self.menu_data = self._load_menu()
        
        gpu_status = f"GPU Aktif ({self.device})" if self.has_gpu else "CPU Modu"
        logger.info(f"✅ Operasyon Yöneticisi aktif. Donanım: {gpu_status}")

    def _init_delivery(self):
        """Paket servis modülünü başlatır."""
        if DeliveryManager:
            try:
                self.delivery_manager = DeliveryManager()
            except Exception as e:
                logger.error(f"DeliveryManager başlatılamadı: {e}")

    @property
    def is_selenium_active(self) -> bool:
        """Paket servis botunun aktiflik durumunu döner."""
        return bool(self.delivery_manager and getattr(self.delivery_manager, 'is_selenium_active', False))

    # --- VERİTABANI YÖNETİMİ ---

    def _init_databases(self):
        """Veritabanı dosyalarını kontrol eder, onarır veya oluşturur."""
        with self.lock:
            if not self.db_file.exists():
                self._internal_save_db({"stok": {}, "rezervasyonlar": [], "last_id": 100})
            else:
                try:
                    data = json.loads(self.db_file.read_text(encoding="utf-8"))
                    if "stok" not in data or "rezervasyonlar" not in data:
                        raise ValueError("Eksik veri yapısı")
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"⚠️ Operasyon DB bozuk: {e}. Kurtarma başlatılıyor...")
                    self._recover_db()

            if not self.menu_file.exists():
                self._create_default_menu()

    def _recover_db(self):
        """Bozuk DB'yi yedekler ve en son sağlam yedekten döner."""
        try:
            corrupt_path = self.db_file.with_suffix(".json.corrupt")
            shutil.move(str(self.db_file), str(corrupt_path))
            
            backups = sorted(list(self.backup_dir.glob("ops_backup_*.json")))
            if backups:
                shutil.copy2(str(backups[-1]), str(self.db_file))
                logger.info("✅ Operasyon verileri yedekten kurtarıldı.")
            else:
                self._internal_save_db({"stok": {}, "rezervasyonlar": [], "last_id": 100})
        except Exception as e:
            logger.error(f"Kritik kurtarma hatası: {e}")

    def _internal_save_db(self, data: Dict):
        """Dahili kullanım için veriyi kaydeder ve yedek alır."""
        try:
            # Önce mevcut olanı yedeğe al
            if self.db_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.backup_dir / f"ops_backup_{timestamp}.json"
                shutil.copy2(self.db_file, backup_path)
                
                # Son 10 yedeği tut
                backups = sorted(list(self.backup_dir.glob("ops_backup_*.json")))
                if len(backups) > 10:
                    for old in backups[:-10]: old.unlink()

            self.db_file.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"DB Kayıt Hatası: {e}")

    def _load_db(self) -> Dict:
        """Veriyi thread-safe şekilde yükler."""
        with self.lock:
            try:
                return json.loads(self.db_file.read_text(encoding="utf-8"))
            except:
                return {"stok": {}, "rezervasyonlar": [], "last_id": 100}

    # --- MENÜ VE ÜRÜN YÖNETİMİ ---

    def _load_menu(self) -> Dict:
        if self.menu_file.exists():
            try:
                return json.loads(self.menu_file.read_text(encoding="utf-8"))
            except: pass
        return {}

    def _create_default_menu(self):
        default_menu = {
            "Kahvaltılar": [
                {"name": "Serpme Kahvaltı", "price": "450 TL", "desc": "Sınırsız çay ile (En az 2 kişilik)"},
                {"name": "Hızlı Kahvaltı Tabağı", "price": "280 TL", "desc": "Tek kişilik pratik seçenek"}
            ],
            "Ana Yemekler": [
                {"name": "Çökertme Kebabı", "price": "380 TL", "desc": "İmza yemeğimiz; bonfile ve çıtır patates"},
                {"name": "Köri Soslu Tavuk", "price": "260 TL", "desc": "Özel baharat harmanıyla"}
            ],
            "Sıcak İçecekler": [
                {"name": "Sahlep", "price": "85 TL", "desc": "Tarçınlı geleneksel kış lezzeti"},
                {"name": "Türk Kahvesi", "price": "70 TL", "desc": "Geleneksel köz tadında"}
            ]
        }
        self.menu_file.write_text(json.dumps(default_menu, indent=4, ensure_ascii=False), encoding="utf-8")

    def get_menu_list(self) -> str:
        """Formatlanmış menü listesi döner."""
        if not self.menu_data: return "Menü şu an güncelleniyor."
        lines = ["--- 🌿 LOTUS BAĞEVİ GÜNCEL MENÜ ---"]
        for cat, items in self.menu_data.items():
            lines.append(f"\n📂 {cat.upper()}")
            for item in items:
                lines.append(f" • {item['name']} ({item['price']}) - {item.get('desc', '')}")
        return "\n".join(lines)

    def get_recommendation(self, weather_context: str = "") -> str:
        """Hava durumu ve saate göre GPU/AI destekli akıllı öneri sunar."""
        hour = datetime.now().hour
        weather = weather_context.lower()
        
        prefix = "🤖 [AI Önerisi]: " if self.has_gpu else ""
        
        if any(k in weather for k in ["soğuk", "kar", "yağmur"]):
            return f"{prefix}Hava dışarıda biraz sert. İçinizi ısıtacak bir 'Sıcak Sahlep' veya 'Cortado' öneririm."
        
        if 8 <= hour < 13:
            return f"{prefix}Şu an tam kahvaltı saati! 'Serpme Kahvaltı'mız güne harika bir başlangıç olur."
        
        if hour >= 18:
            return f"{prefix}Akşam yemeği için imza yemeğimiz 'Çökertme Kebabı' kesinlikle önerimdir."
            
        return f"{prefix}Ortaya bir 'Mix Atıştırmalık Tabağı' söyleyip keyfinize bakabilirsiniz."

    # --- REZERVASYON SİSTEMİ ---

    def add_reservation(self, name: str, time_slot: str, count: Union[int, str], phone: str = None, messenger: Any = None) -> str:
        """Yeni bir rezervasyon kaydeder ve onay gönderir."""
        with self.lock:
            try:
                qty = int(count)
                if qty <= 0: return "❌ Hata: Kişi sayısı geçersiz."
                
                db = self._load_db()
                db["last_id"] += 1
                res_id = db["last_id"]
                
                new_res = {
                    "id": res_id,
                    "name": name.title(),
                    "time": time_slot,
                    "pax": qty,
                    "phone": phone or "Yok",
                    "status": "Onaylandı",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                db["rezervasyonlar"].append(new_res)
                self._internal_save_db(db)
                
                msg = f"✅ Rezervasyon #{res_id} kaydedildi: {name} ({time_slot}, {qty} kişi)."
                
                # WhatsApp Onayı
                if phone and phone != "Yok" and messenger:
                    try:
                        confirm_text = (f"Merhaba {name.title()}, Lotus Bağevi rezervasyonunuz onaylanmıştır.\n"
                                        f"🗓 Zaman: {time_slot}\n👥 Kişi: {qty}\nBekliyoruz!")
                        messenger.send_whatsapp_text(phone, confirm_text)
                        msg += "\n📲 WhatsApp onay mesajı gönderildi."
                    except: pass
                
                return msg
            except Exception as e:
                logger.error(f"Rezervasyon hatası: {e}")
                return "❌ Rezervasyon eklenemedi."

    def cancel_reservation(self, res_id: int) -> bool:
        """Rezervasyonu ID üzerinden iptal eder."""
        with self.lock:
            db = self._load_db()
            original_len = len(db["rezervasyonlar"])
            db["rezervasyonlar"] = [r for r in db["rezervasyonlar"] if r["id"] != int(res_id)]
            
            if len(db["rezervasyonlar"]) < original_len:
                self._internal_save_db(db)
                return True
            return False

    # --- STOK YÖNETİMİ ---

    def update_stock(self, item_name: str, amount: float, operation: str = "add") -> bool:
        """Stok miktarını günceller (add/remove)."""
        with self.lock:
            db = self._load_db()
            name = item_name.strip().title()
            
            current = db["stok"].get(name, {"miktar": 0.0})
            if operation == "add":
                new_qty = current["miktar"] + amount
            else:
                new_qty = max(0, current["miktar"] - amount)
                
            db["stok"][name] = {
                "miktar": new_qty,
                "son_guncelleme": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self._internal_save_db(db)
            return True

    def check_stock_critical(self, threshold: float = 5.0) -> List[str]:
        """Kritik seviyenin altına düşen ürünleri listeler."""
        db = self._load_db()
        return [f"{name} ({data['miktar']})" for name, data in db["stok"].items() if data['miktar'] < threshold]

    def process_invoice_items(self, items_list: List[Dict]) -> str:
        """Gaya'nın faturadan okuduğu listeyi stoklara işler."""
        processed = []
        for item in items_list:
            name = item.get("isim", "Bilinmeyen Ürün")
            qty = item.get("adet", item.get("miktar", 1.0))
            try:
                amount = float(qty) if not isinstance(qty, str) else float(''.join(filter(lambda x: x.isdigit() or x == '.', qty)))
            except: amount = 1.0
            
            if self.update_stock(name, amount, "add"):
                processed.append(f"{name.title()} (+{amount})")
        
        return "✅ Stoklar Güncellendi: " + ", ".join(processed) if processed else "İşlenecek ürün bulunamadı."

    # --- DURUM RAPORLAMA ---

    def get_status_report(self) -> str:
        """Sistemin genel sağlık ve operasyon özetini döner."""
        db = self._load_db()
        res_list = db.get("rezervasyonlar", [])
        
        # Bugünün rezervasyonları
        today = datetime.now().strftime("%Y-%m-%d")
        today_res = [r for r in res_list if today in str(r.get("time", ""))]
        
        report = [
            "--- 📊 OPERASYONEL DURUM RAPORU ---",
            f"⚙️ Donanım: {'🚀 GPU Aktif' if self.has_gpu else '💻 CPU Modu'}",
            f"📅 Bugünün Rezervasyonları: {len(today_res)} / Toplam: {len(res_list)}",
            f"🤖 Paket Servis Botu: {'✅ AKTİF' if self.is_selenium_active else '⚪ KAPALI'}"
        ]
        
        critical = self.check_stock_critical()
        if critical:
            report.append(f"⚠️ KRİTİK STOK UYARISI: {', '.join(critical)}")
            
        return "\n".join(report)

    def get_ops_summary(self) -> str:
        """Ajanlar için kısa bağlam özeti."""
        db = self._load_db()
        hw = "GPU" if self.has_gpu else "CPU"
        return f"Ops ({hw}): {len(db.get('rezervasyonlar', []))} Kayıt | Bot: {'Açık' if self.is_selenium_active else 'Kapalı'}"

# import json
# import logging
# import threading
# from pathlib import Path
# from datetime import datetime

# # LotusAI merkezi yapılandırmasını içe aktar
# try:
#     from config import Config
# except ImportError:
#     # Eğer config dosyası bulunamazsa varsayılan yolları belirle
#     class Config:
#         WORK_DIR = Path(".")
#         LOG_DIR = Path("logs")

# # Paket servis modülünü (DeliveryManager) güvenli şekilde içe aktar
# try:
#     from managers.delivery import DeliveryManager
# except ImportError:
#     try:
#         from delivery import DeliveryManager
#     except ImportError:
#         DeliveryManager = None

# # --- LOGLAMA ---
# if not (Config.WORK_DIR / "logs").exists():
#     (Config.WORK_DIR / "logs").mkdir(parents=True, exist_ok=True)

# log_path = Config.WORK_DIR / "lotus_operations.log"
# logger = logging.getLogger("LotusAI.Operations")
# if not logger.handlers:
#     handler = logging.FileHandler(log_path, encoding='utf-8')
#     formatter = logging.Formatter('%(asctime)s - OPS - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# class OperationsManager:
#     """
#     LotusAI Operasyon Yöneticisi.
#     Stok, Rezervasyon, Menü ve Paket Servis (DeliveryManager) işlemlerini merkezi olarak yönetir.
#     Multi-agent sistemlerde güvenli çalışması için Thread-Safe (Kilitleme) yapısına sahiptir.
#     """
#     def __init__(self):
#         # Dosya yolları merkezi Config üzerinden yönetilir
#         self.db_file = Config.WORK_DIR / "lotus_operasyon.json"
#         self.menu_file = Config.WORK_DIR / "lotus_menu.json"
        
#         # Çoklu ajan erişimi (Thread-safety) için kilit mekanizması
#         self.lock = threading.Lock()
        
#         # Paket Servis Yöneticisi Başlatma
#         self.delivery_manager = None
#         if DeliveryManager:
#             try:
#                 self.delivery_manager = DeliveryManager()
#             except Exception as e:
#                 logger.error(f"DeliveryManager başlatılamadı: {e}")
#         else:
#             logger.warning("DeliveryManager modülü bulunamadı. Paket servis özellikleri kısıtlı.")

#         # Veritabanlarını kontrol et, yoksa oluştur ve yükle
#         self._init_databases()
#         self.menu_data = self._load_menu()

#     # --- ÖZELLİKLER (Properties) ---
#     @property
#     def is_selenium_active(self):
#         """DeliveryManager botunun (Selenium) aktif olup olmadığını kontrol eder."""
#         if self.delivery_manager and hasattr(self.delivery_manager, 'is_selenium_active'):
#             return self.delivery_manager.is_selenium_active
#         return False

#     # --- VERİTABANI YARDIMCILARI ---
#     def _init_databases(self):
#         """Veritabanı dosyalarını güvenli şekilde oluşturur veya bozuksa onarır."""
#         with self.lock:
#             if not self.db_file.exists():
#                 self._internal_save_db({"stok": {}, "rezervasyonlar": [], "last_id": 100})
#                 logger.info("Yeni operasyon veritabanı oluşturuldu.")
#             else:
#                 try:
#                     data = json.loads(self.db_file.read_text(encoding="utf-8"))
#                     if "last_id" not in data: # Eski versiyon desteği
#                         data["last_id"] = 100 + len(data.get("rezervasyonlar", []))
#                         self._internal_save_db(data)
#                 except (json.JSONDecodeError, Exception):
#                     logger.error("Operasyon DB bozuk! Yedeklenip sıfırlanıyor.")
#                     corrupt_path = self.db_file.with_suffix(".json.corrupt")
#                     self.db_file.replace(corrupt_path)
#                     self._internal_save_db({"stok": {}, "rezervasyonlar": [], "last_id": 100})

#             if not self.menu_file.exists():
#                 self._create_default_menu()

#     def _load_db(self):
#         """Veritabanını thread-safe şekilde diskten okur."""
#         with self.lock:
#             try:
#                 if self.db_file.exists():
#                     return json.loads(self.db_file.read_text(encoding="utf-8"))
#             except Exception as e:
#                 logger.error(f"DB Okuma Hatası: {e}")
#             return {"stok": {}, "rezervasyonlar": [], "last_id": 100}

#     def _save_db(self, data):
#         """Dışarıdan çağrılabilen thread-safe kayıt metodu."""
#         with self.lock:
#             self._internal_save_db(data)

#     def _internal_save_db(self, data):
#         """Sınıf içi kullanım için kilitsiz kayıt metodu (Deadlock önlemek için)."""
#         try:
#             self.db_file.write_text(
#                 json.dumps(data, indent=4, ensure_ascii=False), 
#                 encoding="utf-8"
#             )
#         except Exception as e:
#             logger.error(f"DB Kayıt Hatası: {e}")

#     def _load_menu(self):
#         """Menü verisini yükler."""
#         if self.menu_file.exists():
#             try:
#                 return json.loads(self.menu_file.read_text(encoding="utf-8"))
#             except Exception as e:
#                 logger.error(f"Menü okuma hatası: {e}")
#         return {}
    
#     def _create_default_menu(self):
#         """Sistem ilk kurulumu için örnek bir menü dosyası oluşturur."""
#         default_menu = {
#             "Kahvaltılar": [
#                 {"name": "Serpme Kahvaltı", "price": "450 TL", "desc": "Sınırsız çay ile, en az 2 kişilik"},
#                 {"name": "Hızlı Kahvaltı Tabağı", "price": "280 TL", "desc": "Tek kişilik pratik kahvaltı"}
#             ],
#             "Ana Yemekler": [
#                 {"name": "Çökertme Kebabı", "price": "380 TL", "desc": "İmza yemeğimiz; bonfile dilimleri ve çıtır patates"},
#                 {"name": "Köri Soslu Tavuk", "price": "260 TL", "desc": "Özel baharat harmanıyla"}
#             ],
#             "Atıştırmalıklar": [
#                 {"name": "Mix Tabağı", "price": "220 TL", "desc": "Sosis, patates ve börek çeşitleri"},
#                 {"name": "Patates Kızartması", "price": "120 TL", "desc": "Cajun baharatlı"}
#             ],
#             "Kahveler": [
#                 {"name": "Türk Kahvesi", "price": "70 TL", "desc": "Geleneksel lezzet"},
#                 {"name": "Cortado", "price": "90 TL", "desc": "Süt ve espressonun uyumu"}
#             ]
#         }
#         try:
#             self.menu_file.write_text(json.dumps(default_menu, indent=4, ensure_ascii=False), encoding="utf-8")
#             logger.info("Varsayılan menü dosyası oluşturuldu.")
#         except Exception as e:
#             logger.error(f"Menü oluşturma hatası: {e}")

#     # --- MENÜ VE BAĞLAM YÖNETİMİ ---
#     def get_context_summary(self):
#         """Ajanların 'Biz ne satıyoruz?' sorusuna yanıt verebilmesi için özet döner."""
#         if not self.menu_data:
#             return "Menü bilgisi şu an erişilemiyor."
        
#         categories = list(self.menu_data.keys())
#         summary = f"HİZMETLERİMİZ: Kategoriler: {', '.join(categories)}. "
        
#         examples = []
#         for items in self.menu_data.values():
#             if items: examples.append(items[0]['name'])
        
#         summary += f"Öne Çıkan Ürünler: {', '.join(examples[:5])}."
#         return summary

#     def _get_item_price(self, item_name_search):
#         """Menüden ürün ismine göre fiyatı dinamik olarak bulur."""
#         if not self.menu_data: return "(Fiyat Sorunuz)"
        
#         search_lower = item_name_search.lower()
#         for items in self.menu_data.values():
#             for item in items:
#                 if search_lower in item['name'].lower():
#                     return f"({item['price']})"
#         return "(Güncel Fiyat)"

#     def get_menu_list(self):
#         """Kullanıcıya sunulacak formatlanmış tam menü listesi."""
#         if not self.menu_data: return "Menü verisi bulunamadı."
            
#         menu_text = "--- 🌿 LOTUS BAĞEVİ GÜNCEL MENÜSÜ ---\n"
#         for category, items in self.menu_data.items():
#             menu_text += f"\n📂 {category.upper()}\n"
#             for item in items:
#                 menu_text += f" • {item['name']} ({item['price']}): {item.get('desc', '')}\n"
#         return menu_text

#     def get_recommendation(self, weather_context=""):
#         """Hava durumu ve saate göre akıllı menü önerisi yapar."""
#         hour = datetime.now().hour
#         w_lower = weather_context.lower() if weather_context else ""
        
#         prices = {
#             "Sahlep": self._get_item_price("Sahlep"),
#             "Cortado": self._get_item_price("Cortado"),
#             "Serpme": self._get_item_price("Serpme Kahvaltı"),
#             "Cokertme": self._get_item_price("Çökertme"),
#             "Sezar": self._get_item_price("Sezar"),
#             "Mix": self._get_item_price("Mix")
#         }

#         if any(k in w_lower for k in ["soğuk", "kar", "yağmur"]):
#              return f"Hava dışarıda biraz sert. İçinizi ısıtacak bir 'Sıcak Sahlep' {prices['Sahlep']} veya 'Cortado' {prices['Cortado']} öneririm."

#         if 8 <= hour < 13:
#             return f"Şu an tam kahvaltı saati! 'Serpme Kahvaltı'mız {prices['Serpme']} güne harika bir başlangıç olur."
        
#         if hour >= 13:
#             if "özel" in w_lower or hour > 18:
#                 return f"Akşam yemeği için imza yemeğimiz 'Çökertme Kebabı' {prices['Cokertme']} kesinlikle önerimdir."
#             if "hafif" in w_lower or "diyet" in w_lower:
#                 return f"'Tavuklu Sezar Salata' {prices['Sezar']} hem doyurucu hem hafif bir seçenektir."

#         return f"Ortaya bir 'Mix Atıştırmalık Tabağı' {prices['Mix']} söyleyip keyfinize bakabilirsiniz."

#     # --- REZERVASYON YÖNETİMİ ---
#     def add_reservation(self, name, time_slot, count, phone=None, messaging_manager=None):
#         """Yeni rezervasyon kaydeder ve WhatsApp onayı gönderir."""
#         try:
#             # Temel doğrulama
#             try:
#                 count_val = int(count)
#                 if count_val <= 0: return "❌ Hata: Kişi sayısı 0'dan büyük olmalıdır."
#             except ValueError:
#                 return "❌ Hata: Geçersiz kişi sayısı."

#             data = self._load_db()
#             data["last_id"] += 1
#             res_id = data["last_id"]
            
#             new_res = {
#                 "id": res_id,
#                 "isim": name.title(),
#                 "zaman": time_slot,
#                 "kisi": count_val,
#                 "telefon": phone if phone else "Yok",
#                 "durum": "Onaylandı",
#                 "kayit_tarihi": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }
            
#             if "rezervasyonlar" not in data: data["rezervasyonlar"] = []
#             data["rezervasyonlar"].append(new_res)
#             self._save_db(data)
            
#             logger.info(f"Rezervasyon eklendi: #{res_id} - {name}")
#             result_msg = f"✅ Rezervasyon Oluşturuldu (Kod: #{res_id}):\n👤 İsim: {name}\n🕒 Zaman: {time_slot}\n👥 Kişi: {count_val}"

#             # WhatsApp Bildirimi
#             if phone and messaging_manager and phone != "Yok":
#                 try:
#                     msg_text = (f"Merhaba {name.title()}, Lotus Bağevi rezervasyonunuz alınmıştır.\n"
#                                 f"🗓 Tarih/Saat: {time_slot}\n👥 Kişi: {count_val}\nBizi tercih ettiğiniz için teşekkürler!")
                    
#                     response = messaging_manager.send_whatsapp_text(phone, msg_text)
#                     if isinstance(response, dict) and response.get("status") == "success":
#                         result_msg += "\n📲 WhatsApp onay mesajı başarıyla gönderildi."
#                     else:
#                         result_msg += "\n📲 WhatsApp bildirimi sıraya alındı."
#                 except Exception as e:
#                     logger.error(f"Bildirim gönderim hatası: {e}")
            
#             return result_msg
#         except Exception as e:
#             logger.error(f"Rezervasyon ekleme hatası: {e}")
#             return "❌ Rezervasyon eklenirken teknik bir hata oluştu."

#     def get_status_report(self):
#         """Sistemin genel operasyonel durumunu detaylı raporlar."""
#         data = self._load_db()
#         res_list = data.get("rezervasyonlar", [])
        
#         status_msg = f"--- 📊 OPERASYON DURUMU ---\n"
#         status_msg += f"Toplam Rezervasyon: {len(res_list)}\n"
        
#         if res_list:
#             last_res = res_list[-3:]
#             status_msg += "Son Kayıtlar:\n" + "\n".join([f"- {r['isim']} ({r['zaman']})" for r in last_res])
        
#         # Paket Servis Durumu
#         bot_status = "✅ AKTİF" if self.is_selenium_active else "⚪ KAPALI"
#         status_msg += f"\n\n🤖 Paket Servis Botu: {bot_status}"
             
#         return status_msg

#     def get_ops_summary(self):
#         """Atlas/Gaya gibi ajanların hızlı bağlam okuması için kısa özet."""
#         data = self._load_db()
#         res_count = len(data.get("rezervasyonlar", []))
#         bot = "Aktif" if self.is_selenium_active else "Kapalı"
#         return f"Operasyon Özeti: {res_count} Rezervasyon | Paket Servis: {bot}"

#     # --- STOK YÖNETİMİ ---
#     def process_invoice_items(self, items_list):
#         """Gaya'nın faturadan okuduğu ürünleri stok veritabanına işler."""
#         data = self._load_db()
#         if "stok" not in data: data["stok"] = {}
        
#         processed = []
#         for item in items_list:
#             name = item.get("isim", "Bilinmeyen Ürün").strip().title()
            
#             # Sayısal miktar tespiti
#             raw_qty = item.get("adet", item.get("miktar", 1))
#             try:
#                 # Eğer string gelirse (örn: "5 adet") sadece sayı kısmını al
#                 if isinstance(raw_qty, str):
#                     qty = float(''.join(filter(lambda x: x.isdigit() or x == '.', raw_qty)))
#                 else:
#                     qty = float(raw_qty)
#             except:
#                 qty = 1.0
            
#             if name in data["stok"]:
#                 # Mevcut miktarı sayısal olarak güncelle
#                 try:
#                     current_qty = float(data["stok"][name].get("miktar", 0))
#                 except:
#                     current_qty = 0.0
                
#                 data["stok"][name] = {
#                     "miktar": current_qty + qty,
#                     "son_guncelleme": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 }
#             else:
#                 data["stok"][name] = {
#                     "miktar": qty, 
#                     "son_guncelleme": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                 }
#             processed.append(f"{name} ({qty})")
            
#         self._save_db(data)
#         logger.info(f"Stok güncellendi: {', '.join(processed)}")
#         return f"✅ Stok Güncellendi: {', '.join(processed)}"

#     # --- PAKET SERVİS ENTEGRASYONU (Wrapper Metodlar) ---
#     def start_service(self):
#         """Paket servis botunu başlatır."""
#         if self.delivery_manager:
#             return self.delivery_manager.start_service()
#         return False

#     def stop_service(self):
#         """Paket servis botunu durdurur."""
#         if self.delivery_manager:
#             self.delivery_manager.stop_service()

#     def check_orders(self):
#         """Yeni siparişleri kontrol eder."""
#         if self.delivery_manager:
#             return self.delivery_manager.check_new_orders()
#         return []

#     def check_delivery_platforms(self):
#         """Platformların genel durumunu kontrol eder ve kullanıcıya bilgi verir."""
#         orders = self.check_orders()
#         if orders:
#             return "🚨 DİKKAT: Yeni siparişler var: " + ", ".join(orders)
        
#         if self.is_selenium_active:
#             return "✅ Paket servis panelleri açık, şu an yeni sipariş yok."
#         return "⚠️ Paket servis modülü şu an aktif değil."