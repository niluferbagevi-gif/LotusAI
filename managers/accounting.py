import pandas as pd
import logging
import shutil
import threading
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

# GPU Desteği Kontrolü (NVIDIA RAPIDS - cuDF)
try:
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Proje içi modüller
try:
    from config import Config
except ImportError:
    class Config:
        WORK_DIR = Path.cwd()

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Accounting")

class AccountingManager:
    """
    LotusAI Muhasebe ve Finans Yöneticisi.
    
    Yetenekler:
    - GPU Hızlandırma: cuDF desteği ile büyük veri setlerinde yüksek performans.
    - Merkezi Kasa Defteri: Tüm gelir ve giderlerin güvenli takibi.
    - Akıllı Analiz: Kategori ve zaman bazlı finansal performans ölçümü.
    - Veri Güvenliği: Otomatik yedekleme ve hata kurtarma (Auto-Recovery).
    - Çoklu Ajan Desteği: Thread-safe (RLock) yapı ile eşzamanlı kayıt.
    """
    
    def __init__(self):
        # Yollar ve Yapılandırma
        self.work_dir = Path(getattr(Config, "WORK_DIR", "./data"))
        self.filename = self.work_dir / "lotus_kasa_defteri.csv"
        self.backup_dir = self.work_dir / "backups" / "accounting"
        
        # Donanım Durumu
        self.use_gpu = GPU_AVAILABLE
        
        # Çoklu thread erişimi için Reentrant Lock
        self.lock = threading.RLock()
        
        # Gerekli klasörleri oluştur
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Sabit Sütun Yapısı
        self.columns = ["Tarih", "Tur", "Kategori", "Aciklama", "Tutar", "User"]
        
        self._init_db()
        
        status = "GPU (cuDF) aktif" if self.use_gpu else "CPU (Pandas) aktif"
        logger.info(f"✅ Muhasebe Yöneticisi aktif hale getirildi. Mod: {status}")

    def _get_df_engine(self):
        """Kullanılacak veri motorunu (cudf veya pandas) döndürür."""
        return cudf if self.use_gpu else pd

    def _init_db(self):
        """Veritabanı dosyasını kontrol eder, yoksa oluşturur veya onarır."""
        with self.lock:
            if not self.filename.exists():
                self._create_empty_db()
            else:
                try:
                    # Başlangıçta veriyi pandas ile oku (küçük dosya uyumluluğu için)
                    df = pd.read_csv(self.filename)
                    # Sütun doğrulaması ve eksik tamamlama
                    missing_cols = [col for col in self.columns if col not in df.columns]
                    if missing_cols:
                        for col in missing_cols:
                            df[col] = "Bilinmiyor"
                        df.to_csv(self.filename, index=False, encoding="utf-8")
                        logger.info(f"🔧 Eksik sütunlar tamamlandı: {missing_cols}")
                except Exception as e:
                    logger.warning(f"⚠️ Kasa defteri bozuk (Hata: {e}), kurtarma başlatılıyor...")
                    self._recover_db()

    def _create_empty_db(self):
        """Yeni bir boş kasa defteri oluşturur."""
        try:
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False, encoding="utf-8")
            logger.info("🆕 Yeni kasa defteri oluşturuldu.")
        except Exception as e:
            logger.error(f"❌ DB oluşturma hatası: {e}")

    def _recover_db(self):
        """Bozuk dosyayı yedekleyip en son sağlam yedeği veya boş dosyayı devreye alır."""
        try:
            if self.filename.exists():
                corrupt_path = self.filename.with_suffix(".csv.corrupt")
                shutil.move(str(self.filename), str(corrupt_path))
            
            # En son sağlam yedeği bul
            backups = sorted(list(self.backup_dir.glob("kasa_yedek_*.csv")))
            if backups:
                shutil.copy2(str(backups[-1]), str(self.filename))
                logger.info("✅ Sistem son sağlam yedekten kurtarıldı.")
            else:
                self._create_empty_db()
        except Exception as e:
            logger.error(f"❌ Kurtarma hatası: {e}")

    def _backup_db(self):
        """Veritabanının zaman damgalı bir yedeğini oluşturur."""
        if not self.filename.exists():
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"kasa_yedek_{timestamp}.csv"
            shutil.copy2(self.filename, backup_path)
            
            # Rotasyon: Son 15 yedeği tut, eskileri sil
            backups = sorted(list(self.backup_dir.glob("kasa_yedek_*.csv")))
            if len(backups) > 15:
                for old_backup in backups[:-15]:
                    old_backup.unlink()
        except Exception as e:
            logger.error(f"⚠️ Yedekleme başarısız: {e}")

    def _parse_amount(self, tutar: Union[str, float, int]) -> float:
        """Karışık formatlı tutar girişlerini standart float değerine dönüştürür."""
        if isinstance(tutar, (int, float)):
            return float(tutar)
        
        try:
            temp = str(tutar).upper().replace("TL", "").replace("TRY", "").replace("₺", "").strip()
            if "," in temp and "." in temp:
                temp = temp.replace(".", "").replace(",", ".")
            elif "," in temp:
                temp = temp.replace(",", ".")
            return float(temp)
        except (ValueError, TypeError):
            logger.warning(f"⚠️ Tutar ayrıştırılamadı: {tutar}. 0.0 atandı.")
            return 0.0

    # --- KAYIT İŞLEMLERİ ---

    def add_entry(self, tur: str, aciklama: str, tutar: Any, kategori: str = "Genel", user_id: str = "Sistem") -> bool:
        """Kasa defterine yeni bir kayıt ekler."""
        tur = str(tur).upper()
        if tur not in ["GELIR", "GIDER"]:
            logger.error(f"❌ Geçersiz işlem türü: {tur}")
            return False

        with self.lock:
            try:
                self._backup_db()
                clean_tutar = self._parse_amount(tutar)
                engine = self._get_df_engine()

                # Mevcut veriyi yükle
                if self.filename.exists():
                    # Yazma işlemi sırasında küçük verilerde pandas daha güvenlidir (disk I/O)
                    df_main = pd.read_csv(self.filename)
                else:
                    df_main = pd.DataFrame(columns=self.columns)

                new_data = {
                    "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Tur": tur,
                    "Kategori": str(kategori).title(),
                    "Aciklama": str(aciklama),
                    "Tutar": clean_tutar,
                    "User": str(user_id)
                }
                
                new_row = pd.DataFrame([new_data])
                df_main = pd.concat([df_main, new_row], ignore_index=True)
                
                # Diske kaydet
                df_main.to_csv(self.filename, index=False, encoding="utf-8")
                
                logger.info(f"💰 {tur} Kaydedildi: {aciklama} | {clean_tutar} TL")
                return True
            except Exception as e:
                logger.error(f"❌ Kayıt hatası: {e}")
                return False

    def delete_entry(self, index: int) -> bool:
        """Belirli bir satırdaki kaydı siler."""
        with self.lock:
            try:
                df = pd.read_csv(self.filename)
                if 0 <= index < len(df):
                    self._backup_db()
                    df = df.drop(df.index[index])
                    df.to_csv(self.filename, index=False, encoding="utf-8")
                    logger.warning(f"🗑️ İndeks {index} üzerindeki kayıt silindi.")
                    return True
                return False
            except Exception as e:
                logger.error(f"❌ Silme hatası: {e}")
                return False

    # --- ANALİZ VE RAPORLAMA (GPU Hızlandırmalı Alanlar) ---

    def _load_data_to_engine(self):
        """Veriyi mevcut motor (GPU/CPU) ile yükler."""
        if not self.filename.exists():
            engine = self._get_df_engine()
            return engine.DataFrame(columns=self.columns)
        
        if self.use_gpu:
            return cudf.read_csv(self.filename)
        else:
            return pd.read_csv(self.filename)

    def get_balance(self) -> float:
        """Sistemin toplam net bakiyesini hesaplar."""
        with self.lock:
            try:
                df = self._load_data_to_engine()
                if df.empty: return 0.0
                
                total_gelir = df[df['Tur'] == 'GELIR']['Tutar'].sum()
                total_gider = df[df['Tur'] == 'GIDER']['Tutar'].sum()
                
                # GPU nesnesinden Python float değerine dönüştür
                result = float(total_gelir - total_gider)
                return result
            except Exception as e:
                logger.error(f"❌ Bakiye hesaplanamadı: {e}")
                return 0.0

    def get_filtered_data(self, start_date=None, end_date=None, category=None, user=None) -> pd.DataFrame:
        """Kriterlere göre filtrelenmiş verileri döner."""
        with self.lock:
            try:
                df = self._load_data_to_engine()
                if df.empty: 
                    return pd.DataFrame(columns=self.columns)
                
                df['Tarih'] = df['Tarih'].astype('datetime64[ns]')

                if start_date:
                    df = df[df['Tarih'] >= datetime.strptime(str(start_date), "%Y-%m-%d")]
                if end_date:
                    df = df[df['Tarih'] <= datetime.strptime(str(end_date), "%Y-%m-%d")]
                if category:
                    df = df[df['Kategori'] == str(category).title()]
                if user:
                    df = df[df['User'] == user]
                
                # Eğer GPU kullanılıyorsa, sonucu pandas'a çevirip dön (Dış dünya ile uyum için)
                return df.to_pandas() if self.use_gpu else df
            except Exception as e:
                logger.error(f"❌ Filtreleme hatası: {e}")
                return pd.DataFrame()

    def get_category_summary(self) -> Dict[str, Dict[str, float]]:
        """Kategori bazında harcama ve gelir özetini döndürür."""
        with self.lock:
            try:
                df = self._load_data_to_engine()
                if df.empty: return {}
                
                summary = df.groupby(['Kategori', 'Tur'])['Tutar'].sum()
                
                # GPU nesnesini standart sözlüğe dönüştürme
                if self.use_gpu:
                    res_pd = summary.to_pandas().unstack(fill_value=0)
                else:
                    res_pd = summary.unstack(fill_value=0)
                    
                return res_pd.to_dict(orient='index')
            except Exception:
                return {}

    def get_recent_transactions(self, limit: int = 5) -> str:
        """Son işlemleri listeler."""
        with self.lock:
            try:
                if not self.filename.exists(): return "Kayıt bulunamadı."
                df = pd.read_csv(self.filename) # UI işlemleri için CPU yeterli
                if df.empty: return "Henüz işlem kaydı yok."
                
                last_rows = df.tail(limit).iloc[::-1]
                result = []
                for _, row in last_rows.iterrows():
                    icon = "🟢" if row['Tur'] == "GELIR" else "🔴"
                    time_str = str(row['Tarih'])[:16]
                    result.append(f"{icon} [{time_str}] {row['Aciklama']} ({row['Tutar']:,.2f} TL) | {row['User']}")
                
                return "\n".join(result)
            except Exception:
                return "İşlem geçmişi okunamadı."

    def get_report(self) -> str:
        """Detaylı finansal rapor üretir."""
        with self.lock:
            try:
                df = self._load_data_to_engine()
                if df.empty: return "ℹ️ Kasa defterinde henüz kayıt bulunmuyor."
                
                balance = self.get_balance()
                total_gelir = float(df[df['Tur'] == 'GELIR']['Tutar'].sum())
                total_gider = float(df[df['Tur'] == 'GIDER']['Tutar'].sum())
                
                # Trend Analizi
                df['Tarih'] = df['Tarih'].astype('datetime64[ns]')
                thirty_days_ago = datetime.now() - timedelta(days=30)
                m_df = df[df['Tarih'] > thirty_days_ago]
                
                m_gelir = float(m_df[m_df['Tur'] == 'GELIR']['Tutar'].sum())
                m_gider = float(m_df[m_df['Tur'] == 'GIDER']['Tutar'].sum())
                
                report = [
                    "📊 LOTUSAI FİNANSAL DURUM RAPORU (GPU Destekli)",
                    f"{'='*40}",
                    f"💰 Mevcut Kasa: {balance:,.2f} TL",
                    f"📈 Toplam Gelir: {total_gelir:,.2f} TL",
                    f"📉 Toplam Gider: {total_gider:,.2f} TL",
                    f"{'-'*40}",
                    f"📅 Son 30 Günlük Performans:",
                    f"   Giriş: +{m_gelir:,.2f} TL",
                    f"   Çıkış: -{m_gider:,.2f} TL",
                    f"   Net:   {(m_gelir - m_gider):,.2f} TL",
                    f"{'-'*40}",
                    f"📝 Son İşlemler:\n{self.get_recent_transactions(3)}",
                    f"{'='*40}"
                ]
                return "\n".join(report)
            except Exception as e:
                return f"❌ Rapor oluşturulamadı: {e}"

    def export_to_excel(self, target_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Kayıtları Excel formatına dönüştürür."""
        try:
            if not target_path:
                target_path = self.work_dir / f"Lotus_Finans_{datetime.now().strftime('%Y%m%d')}.xlsx"
            
            df = pd.read_csv(self.filename)
            df.to_excel(target_path, index=False)
            logger.info(f"📁 Finans raporu dışa aktarıldı: {target_path}")
            return str(target_path)
        except Exception as e:
            logger.error(f"❌ Excel export hatası: {e}")
            return None