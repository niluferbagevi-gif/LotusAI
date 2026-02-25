"""
LotusAI managers/operations.py - Operations Manager
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu & Mimari Optimizasyon)
Açıklama: Saha, Stok, Rezervasyon ve Menü Yönetimi

Özellikler:
- Stok Yönetimi: Ürün girişi, çıkışı ve kritik seviye takibi.
- Rezervasyon: Kayıt, onaylama, iptal ve WhatsApp entegrasyonu.
- Akıllı Menü: Dinamik öneri sistemi.
- Paket Servis: DeliveryManager üzerinden bot kontrolü ve durum takibi.
- Veri Güvenliği: RLock ile eşzamanlılık, otomatik yedekli çalışma ve Config.DATA_DIR entegrasyonu.
- Erişim seviyesi kontrolleri (restricted/sandbox/full)
"""

import json
import logging
import threading
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Operations")

# --- GPU KONTROLÜ (Hafifletilmiş Config Entegrasyonu) ---
# Torch kütüphanesine gerek kalmadan sadece donanım bayrağını kullanıyoruz
HAS_GPU = getattr(Config, "USE_GPU", False)
DEVICE = "cuda" if HAS_GPU else "cpu"

if HAS_GPU:
    logger.info("🚀 OperationsManager GPU/Donanım Hızlandırma farkındalığıyla aktif")
else:
    logger.info("ℹ️ Operasyon işlemleri CPU modunda.")

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
    - Akıllı Menü: Durumsal dinamik öneri sistemi.
    - Paket Servis: DeliveryManager üzerinden bot kontrolü ve durum takibi.
    - Veri Güvenliği: RLock ile eşzamanlılık ve otomatik yedekli çalışma.
    - Erişim seviyesi kontrolleri (restricted/sandbox/full)
    """
    
    def __init__(self, access_level: Optional[str] = None):
        """
        OperationsManager başlatıcı
        
        Args:
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.access_level = access_level or getattr(Config, "ACCESS_LEVEL", "sandbox")
        
        # Yollar (Merkezi Config yapısına bağlandı)
        self.work_dir = Config.WORK_DIR
        self.db_file = Config.DATA_DIR / "lotus_operasyon.json"
        self.menu_file = Config.DATA_DIR / "lotus_menu.json"
        self.backup_dir = Config.DATA_DIR / "operations_backups"
        
        try:
            Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Dizin oluşturma hatası: {e}")
        
        # Eski veri taşıma işlemi (Geriye dönük uyumluluk)
        self._migrate_old_db()

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
        logger.info(f"✅ Operasyon Yöneticisi aktif. Donanım: {gpu_status}, Erişim: {self.access_level.upper()}")

    def _migrate_old_db(self) -> None:
        """Eski dizin yapısındaki JSON veritabanlarını yeni DATA_DIR'e taşır"""
        old_db = self.work_dir / "lotus/lotus_operasyon.json"
        if old_db.exists() and not self.db_file.exists():
            try:
                shutil.move(str(old_db), str(self.db_file))
                logger.info(f"📦 Eski operasyon DB yeni merkeze taşındı: {self.db_file}")
            except Exception as e:
                logger.warning(f"⚠️ Eski DB taşıma hatası: {e}")

        old_menu = self.work_dir / "lotus/lotus_menu.json"
        if old_menu.exists() and not self.menu_file.exists():
            try:
                shutil.move(str(old_menu), str(self.menu_file))
                logger.info(f"📦 Eski menü dosyası yeni merkeze taşındı: {self.menu_file}")
            except Exception as e:
                logger.warning(f"⚠️ Eski menü taşıma hatası: {e}")

    def _init_delivery(self):
        """Paket servis modülünü başlatır."""
        if DeliveryManager:
            try:
                self.delivery_manager = DeliveryManager(access_level=self.access_level)
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
        """Formatlanmış menü listesi döner. (Her erişim seviyesinde kullanılabilir)"""
        if not self.menu_data: return "Menü şu an güncelleniyor."
        lines = ["--- 🌿 LOTUS BAĞEVİ GÜNCEL MENÜ ---"]
        for cat, items in self.menu_data.items():
            lines.append(f"\n📂 {cat.upper()}")
            for item in items:
                lines.append(f" • {item['name']} ({item['price']}) - {item.get('desc', '')}")
        return "\n".join(lines)

    def get_recommendation(self, weather_context: str = "") -> str:
        """Hava durumu ve saate göre dinamik akıllı öneri sunar. (Her erişim seviyesinde kullanılabilir)"""
        hour = datetime.now().hour
        weather = weather_context.lower()
        
        prefix = "🤖 [AI Önerisi]: " if self.has_gpu else "💡 [Sistem Önerisi]: "
        
        if any(k in weather for k in ["soğuk", "kar", "yağmur"]):
            return f"{prefix}Hava dışarıda biraz sert. İçinizi ısıtacak bir 'Sıcak Sahlep' veya 'Cortado' öneririm."
        
        if 8 <= hour < 13:
            return f"{prefix}Şu an tam kahvaltı saati! 'Serpme Kahvaltı'mız güne harika bir başlangıç olur."
        
        if hour >= 18:
            return f"{prefix}Akşam yemeği için imza yemeğimiz 'Çökertme Kebabı' kesinlikle önerimdir."
            
        return f"{prefix}Ortaya bir 'Mix Atıştırmalık Tabağı' söyleyip keyfinize bakabilirsiniz."

    # --- REZERVASYON SİSTEMİ ---

    def add_reservation(self, name: str, time_slot: str, count: Union[int, str], phone: str = None, messenger: Any = None) -> str:
        """
        Yeni bir rezervasyon kaydeder ve onay gönderir.
        Sadece sandbox ve full modda çalışır.
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return "🔒 Kısıtlı modda rezervasyon eklenemez. Sadece bilgi alabilirsiniz."
        
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
        """
        Rezervasyonu ID üzerinden iptal eder.
        Sadece sandbox ve full modda çalışır.
        """
        if self.access_level == AccessLevel.RESTRICTED:
            logger.warning("🚫 Kısıtlı modda iptal engellendi")
            return False
        
        with self.lock:
            db = self._load_db()
            original_len = len(db["rezervasyonlar"])
            db["rezervasyonlar"] = [r for r in db["rezervasyonlar"] if r["id"] != int(res_id)]
            
            if len(db["rezervasyonlar"]) < original_len:
                self._internal_save_db(db)
                return True
            return False

    def get_reservations_summary(self) -> str:
        """Bugünkü rezervasyonları listeler (Her erişim seviyesinde kullanılabilir)."""
        db = self._load_db()
        res_list = db.get("rezervasyonlar", [])
        
        today = datetime.now().strftime("%Y-%m-%d")
        today_res = [r for r in res_list if today in str(r.get("time", ""))]
        
        if not today_res:
            return "Bugün için kayıtlı rezervasyon bulunmamaktadır."
            
        lines = []
        for r in today_res:
            lines.append(f"#{r['id']} - {r['name']} ({r['pax']} Kişi) - Saat: {r['time']}")
            
        return "\n".join(lines)

    # --- STOK YÖNETİMİ ---

    def update_stock(self, item_name: str, amount: float, operation: str = "add") -> bool:
        """
        Stok miktarını günceller (add/remove).
        Sadece sandbox ve full modda çalışır.
        """
        if self.access_level == AccessLevel.RESTRICTED:
            logger.warning("🚫 Kısıtlı modda stok güncelleme engellendi")
            return False
        
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
        """Kritik seviyenin altına düşen ürünleri listeler. (Her erişim seviyesinde kullanılabilir)"""
        db = self._load_db()
        return [f"{name} ({data['miktar']})" for name, data in db["stok"].items() if data['miktar'] < threshold]

    def get_stock_summary(self) -> str:
        """Mevcut stok listesini metin olarak döner. (Her erişim seviyesinde kullanılabilir)"""
        db = self._load_db()
        stocks = db.get("stok", {})
        
        if not stocks:
            return "Stokta kayıtlı ürün bulunmuyor."
            
        lines = []
        for name, data in stocks.items():
            lines.append(f"📦 {name}: {data['miktar']}")
            
        return "\n".join(lines)

    def process_invoice_items(self, items_list: List[Dict]) -> str:
        """
        Gaya'nın faturadan okuduğu listeyi stoklara işler.
        Sadece sandbox ve full modda çalışır.
        """
        if self.access_level == AccessLevel.RESTRICTED:
            return "🔒 Kısıtlı modda fatura işlenemez."
        
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
        """Sistemin genel sağlık ve operasyon özetini döner. (Her erişim seviyesinde kullanılabilir)"""
        db = self._load_db()
        res_list = db.get("rezervasyonlar", [])
        
        # Bugünün rezervasyonları
        today = datetime.now().strftime("%Y-%m-%d")
        today_res = [r for r in res_list if today in str(r.get("time", ""))]
        
        # Erişim seviyesi simgesi
        access_icon = {
            AccessLevel.RESTRICTED: "🔒",
            AccessLevel.SANDBOX: "📦",
            AccessLevel.FULL: "⚡"
        }.get(self.access_level, "🔐")
        
        report = [
            f"--- 📊 OPERASYONEL DURUM RAPORU ---",
            f"🔐 Erişim: {self.access_level.upper()} {access_icon}",
            f"⚙️ Donanım: {'🚀 GPU Aktif' if self.has_gpu else '💻 CPU Modu'}",
            f"📅 Bugünün Rezervasyonları: {len(today_res)} / Toplam: {len(res_list)}",
            f"🤖 Paket Servis Botu: {'✅ AKTİF' if self.is_selenium_active else '⚪ KAPALI'}"
        ]
        
        critical = self.check_stock_critical()
        if critical:
            report.append(f"⚠️ KRİTİK STOK UYARISI: {', '.join(critical)}")
            
        return "\n".join(report)

    def get_ops_summary(self) -> str:
        """Ajanlar için kısa bağlam özeti. (Her erişim seviyesinde kullanılabilir)"""
        db = self._load_db()
        hw = "GPU" if self.has_gpu else "CPU"
        return f"Ops ({hw}, {self.access_level}): {len(db.get('rezervasyonlar', []))} Kayıt | Bot: {'Açık' if self.is_selenium_active else 'Kapalı'}"