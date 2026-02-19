"""
LotusAI Accounting Manager
Sürüm: 2.5.3
Açıklama: Muhasebe ve finans yönetimi

Özellikler:
- GPU hızlandırma (cuDF)
- Merkezi kasa defteri
- Kategori bazlı analiz
- Otomatik yedekleme
- Veri kurtarma
- Thread-safe operations
"""

import pandas as pd
import logging
import shutil
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config

logger = logging.getLogger("LotusAI.Accounting")


# ═══════════════════════════════════════════════════════════════
# GPU (cuDF)
# ═══════════════════════════════════════════════════════════════
HAS_GPU = False
cudf = None

if Config.USE_GPU:
    try:
        import cudf
        HAS_GPU = True
        logger.info("🚀 AccountingManager: GPU (cuDF) aktif")
    except ImportError:
        logger.info("ℹ️ AccountingManager: cuDF yok, Pandas kullanılacak")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class TransactionType(Enum):
    """İşlem tipleri"""
    INCOME = "GELIR"
    EXPENSE = "GIDER"
    
    @property
    def emoji(self) -> str:
        """İşlem emoji'si"""
        return "🟢" if self == TransactionType.INCOME else "🔴"


class Category(Enum):
    """İşlem kategorileri"""
    GENERAL = "Genel"
    KITCHEN = "Mutfak/Operasyon"
    STAFF = "Personel"
    MARKETING = "Pazarlama"
    UTILITIES = "Faturalar"
    INVENTORY = "Stok"
    MAINTENANCE = "Bakım"
    OTHER = "Diğer"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class Transaction:
    """İşlem verisi"""
    date: datetime
    transaction_type: TransactionType
    category: str
    description: str
    amount: float
    user: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e çevir"""
        return {
            "Tarih": self.date.strftime("%Y-%m-%d %H:%M:%S"),
            "Tur": self.transaction_type.value,
            "Kategori": self.category,
            "Aciklama": self.description,
            "Tutar": self.amount,
            "User": self.user
        }


@dataclass
class FinancialSummary:
    """Finansal özet"""
    balance: float
    total_income: float
    total_expense: float
    monthly_income: float
    monthly_expense: float
    monthly_net: float
    
    def __str__(self) -> str:
        return (
            f"Bakiye: {self.balance:,.2f} TL | "
            f"Gelir: {self.total_income:,.2f} TL | "
            f"Gider: {self.total_expense:,.2f} TL"
        )


@dataclass
class AccountingMetrics:
    """Muhasebe metrikleri"""
    total_transactions: int = 0
    income_transactions: int = 0
    expense_transactions: int = 0
    backups_created: int = 0
    recoveries_performed: int = 0
    errors_encountered: int = 0


# ═══════════════════════════════════════════════════════════════
# ACCOUNTING MANAGER
# ═══════════════════════════════════════════════════════════════
class AccountingManager:
    """
    LotusAI Muhasebe ve Finans Yöneticisi
    
    Yetenekler:
    - GPU hızlandırma: cuDF ile büyük veri setlerinde yüksek performans
    - Merkezi kasa defteri: Tüm gelir ve giderlerin güvenli takibi
    - Akıllı analiz: Kategori ve zaman bazlı finansal ölçümler
    - Veri güvenliği: Otomatik yedekleme ve hata kurtarma
    - Thread-safe: Çoklu agent desteği
    
    Database schema:
    - Tarih: İşlem zamanı
    - Tur: GELIR veya GIDER
    - Kategori: İşlem kategorisi
    - Aciklama: İşlem açıklaması
    - Tutar: İşlem tutarı (TL)
    - User: İşlemi yapan kullanıcı/agent
    """
    
    # Column schema
    COLUMNS = ["Tarih", "Tur", "Kategori", "Aciklama", "Tutar", "User"]
    
    # Backup settings
    MAX_BACKUPS = 15
    
    def __init__(self):
        """Accounting manager başlatıcı"""
        # Paths
        self.work_dir = Config.WORK_DIR
        self.filename = self.work_dir / "lotus_kasa_defteri.csv"
        self.backup_dir = self.work_dir / "backups" / "accounting"
        
        # Hardware
        self.use_gpu = HAS_GPU
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metrics
        self.metrics = AccountingMetrics()
        
        # Create directories
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Dizin oluşturma hatası: {e}")
        
        # Initialize database
        self._init_db()
        
        status = "GPU (cuDF)" if self.use_gpu else "CPU (Pandas)"
        logger.info(f"✅ Muhasebe Yöneticisi aktif ({status})")
    
    # ───────────────────────────────────────────────────────────
    # DATABASE INITIALIZATION
    # ───────────────────────────────────────────────────────────
    
    def _get_df_engine(self):
        """Veri motorunu döndür (cudf veya pandas)"""
        return cudf if (self.use_gpu and cudf) else pd
    
    def _init_db(self) -> None:
        """Veritabanını başlat"""
        with self.lock:
            if not self.filename.exists():
                self._create_empty_db()
            else:
                self._validate_and_repair_db()
    
    def _create_empty_db(self) -> None:
        """Boş veritabanı oluştur"""
        try:
            df = pd.DataFrame(columns=self.COLUMNS)
            df.to_csv(self.filename, index=False, encoding="utf-8")
            logger.info("🆕 Yeni kasa defteri oluşturuldu")
        except Exception as e:
            logger.error(f"DB oluşturma hatası: {e}")
            self.metrics.errors_encountered += 1
    
    def _validate_and_repair_db(self) -> None:
        """Veritabanını doğrula ve onar"""
        try:
            df = pd.read_csv(self.filename)
            
            # Column validation
            missing_cols = [
                col for col in self.COLUMNS
                if col not in df.columns
            ]
            
            if missing_cols:
                for col in missing_cols:
                    df[col] = "Bilinmiyor"
                
                df.to_csv(self.filename, index=False, encoding="utf-8")
                logger.info(f"🔧 Eksik sütunlar eklendi: {missing_cols}")
        
        except Exception as e:
            logger.warning(f"⚠️ Kasa defteri bozuk: {e}")
            self._recover_db()
    
    def _recover_db(self) -> None:
        """Veritabanını kurtar"""
        try:
            # Move corrupted file
            if self.filename.exists():
                corrupt_path = self.filename.with_suffix(".csv.corrupt")
                shutil.move(str(self.filename), str(corrupt_path))
            
            # Find latest backup
            backups = sorted(
                list(self.backup_dir.glob("kasa_yedek_*.csv"))
            )
            
            if backups:
                shutil.copy2(str(backups[-1]), str(self.filename))
                logger.info("✅ Son yedekten kurtarıldı")
            else:
                self._create_empty_db()
            
            self.metrics.recoveries_performed += 1
        
        except Exception as e:
            logger.error(f"Kurtarma hatası: {e}")
            self.metrics.errors_encountered += 1
    
    # ───────────────────────────────────────────────────────────
    # BACKUP SYSTEM
    # ───────────────────────────────────────────────────────────
    
    def _backup_db(self) -> None:
        """Veritabanını yedekle"""
        if not self.filename.exists():
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"kasa_yedek_{timestamp}.csv"
            shutil.copy2(self.filename, backup_path)
            
            # Rotation: Keep only last MAX_BACKUPS
            backups = sorted(
                list(self.backup_dir.glob("kasa_yedek_*.csv"))
            )
            
            if len(backups) > self.MAX_BACKUPS:
                for old_backup in backups[:-self.MAX_BACKUPS]:
                    old_backup.unlink()
            
            self.metrics.backups_created += 1
        
        except Exception as e:
            logger.error(f"Yedekleme hatası: {e}")
            self.metrics.errors_encountered += 1
    
    # ───────────────────────────────────────────────────────────
    # AMOUNT PARSING
    # ───────────────────────────────────────────────────────────
    
    def _parse_amount(self, tutar: Union[str, float, int]) -> float:
        """
        Tutar parse
        
        Args:
            tutar: Ham tutar değeri
        
        Returns:
            Temiz float değer
        """
        if isinstance(tutar, (int, float)):
            return float(tutar)
        
        try:
            # String temizleme
            temp = str(tutar).upper()
            temp = temp.replace("TL", "").replace("TRY", "")
            temp = temp.replace("₺", "").strip()
            
            # Binlik ve ondalık ayırıcı
            if "," in temp and "." in temp:
                temp = temp.replace(".", "").replace(",", ".")
            elif "," in temp:
                temp = temp.replace(",", ".")
            
            return float(temp)
        
        except (ValueError, TypeError) as e:
            logger.warning(f"Tutar parse hatası ({tutar}): {e}")
            return 0.0
    
    # ───────────────────────────────────────────────────────────
    # TRANSACTION OPERATIONS
    # ───────────────────────────────────────────────────────────
    
    def add_entry(
        self,
        tur: str,
        aciklama: str,
        tutar: Any,
        kategori: str = "Genel",
        user_id: str = "Sistem"
    ) -> bool:
        """
        Yeni işlem ekle
        
        Args:
            tur: İşlem tipi (GELIR/GIDER)
            aciklama: Açıklama
            tutar: Tutar
            kategori: Kategori
            user_id: Kullanıcı/Agent
        
        Returns:
            Başarılı ise True
        """
        # Validate transaction type
        tur = str(tur).upper()
        if tur not in ["GELIR", "GIDER"]:
            logger.error(f"Geçersiz işlem türü: {tur}")
            return False
        
        with self.lock:
            try:
                # Backup
                self._backup_db()
                
                # Parse amount
                clean_tutar = self._parse_amount(tutar)
                
                # Validate amount
                if clean_tutar < 0:
                    logger.error(f"Negatif tutar: {clean_tutar}")
                    return False
                
                # Load existing data
                if self.filename.exists():
                    df_main = pd.read_csv(self.filename)
                else:
                    df_main = pd.DataFrame(columns=self.COLUMNS)
                
                # Create new transaction
                transaction = Transaction(
                    date=datetime.now(),
                    transaction_type=TransactionType.INCOME if tur == "GELIR" else TransactionType.EXPENSE,
                    category=str(kategori).title(),
                    description=str(aciklama),
                    amount=clean_tutar,
                    user=str(user_id)
                )
                
                # Add to dataframe
                new_row = pd.DataFrame([transaction.to_dict()])
                df_main = pd.concat([df_main, new_row], ignore_index=True)
                
                # Save
                df_main.to_csv(self.filename, index=False, encoding="utf-8")
                
                # Update metrics
                self.metrics.total_transactions += 1
                if tur == "GELIR":
                    self.metrics.income_transactions += 1
                else:
                    self.metrics.expense_transactions += 1
                
                logger.info(
                    f"💰 {tur} kaydedildi: {aciklama} | "
                    f"{clean_tutar:,.2f} TL"
                )
                
                return True
            
            except Exception as e:
                logger.error(f"Kayıt hatası: {e}")
                self.metrics.errors_encountered += 1
                return False
    
    def delete_entry(self, index: int) -> bool:
        """
        İşlem sil
        
        Args:
            index: Satır index'i
        
        Returns:
            Başarılı ise True
        """
        with self.lock:
            try:
                df = pd.read_csv(self.filename)
                
                if 0 <= index < len(df):
                    self._backup_db()
                    df = df.drop(df.index[index])
                    df.to_csv(self.filename, index=False, encoding="utf-8")
                    
                    logger.warning(f"🗑️ Index {index} silindi")
                    return True
                
                logger.error(f"Geçersiz index: {index}")
                return False
            
            except Exception as e:
                logger.error(f"Silme hatası: {e}")
                self.metrics.errors_encountered += 1
                return False
    
    # ───────────────────────────────────────────────────────────
    # ANALYSIS & REPORTING
    # ───────────────────────────────────────────────────────────
    
    def _load_data_to_engine(self):
        """Veriyi yükle (GPU/CPU)"""
        if not self.filename.exists():
            engine = self._get_df_engine()
            return engine.DataFrame(columns=self.COLUMNS)
        
        if self.use_gpu and cudf:
            return cudf.read_csv(self.filename)
        else:
            return pd.read_csv(self.filename)
    
    def get_balance(self) -> float:
        """
        Net bakiye hesapla
        
        Returns:
            Bakiye (TL)
        """
        with self.lock:
            try:
                df = self._load_data_to_engine()
                
                if df.empty:
                    return 0.0
                
                total_income = df[df['Tur'] == 'GELIR']['Tutar'].sum()
                total_expense = df[df['Tur'] == 'GIDER']['Tutar'].sum()
                
                return float(total_income - total_expense)
            
            except Exception as e:
                logger.error(f"Bakiye hesaplama hatası: {e}")
                return 0.0
    
    def get_summary(self) -> FinancialSummary:
        """
        Finansal özet
        
        Returns:
            FinancialSummary objesi
        """
        with self.lock:
            try:
                df = self._load_data_to_engine()
                
                if df.empty:
                    return FinancialSummary(
                        balance=0.0,
                        total_income=0.0,
                        total_expense=0.0,
                        monthly_income=0.0,
                        monthly_expense=0.0,
                        monthly_net=0.0
                    )
                
                # Total calculations
                total_income = float(df[df['Tur'] == 'GELIR']['Tutar'].sum())
                total_expense = float(df[df['Tur'] == 'GIDER']['Tutar'].sum())
                balance = total_income - total_expense
                
                # Monthly calculations
                if self.use_gpu:
                    df['Tarih'] = cudf.to_datetime(df['Tarih'])
                else:
                    df['Tarih'] = pd.to_datetime(df['Tarih'])
                
                thirty_days_ago = datetime.now() - timedelta(days=30)
                monthly_df = df[df['Tarih'] > thirty_days_ago]
                
                monthly_income = float(
                    monthly_df[monthly_df['Tur'] == 'GELIR']['Tutar'].sum()
                )
                monthly_expense = float(
                    monthly_df[monthly_df['Tur'] == 'GIDER']['Tutar'].sum()
                )
                monthly_net = monthly_income - monthly_expense
                
                return FinancialSummary(
                    balance=balance,
                    total_income=total_income,
                    total_expense=total_expense,
                    monthly_income=monthly_income,
                    monthly_expense=monthly_expense,
                    monthly_net=monthly_net
                )
            
            except Exception as e:
                logger.error(f"Özet oluşturma hatası: {e}")
                return FinancialSummary(0, 0, 0, 0, 0, 0)
    
    def get_filtered_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        user: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filtrelenmiş veri
        
        Args:
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD)
            category: Kategori
            user: Kullanıcı
        
        Returns:
            Pandas DataFrame
        """
        with self.lock:
            try:
                df = self._load_data_to_engine()
                
                if df.empty:
                    return pd.DataFrame(columns=self.COLUMNS)
                
                # Date conversion
                if self.use_gpu:
                    df['Tarih'] = cudf.to_datetime(df['Tarih'])
                else:
                    df['Tarih'] = pd.to_datetime(df['Tarih'])
                
                # Filters
                if start_date:
                    start_dt = datetime.strptime(str(start_date), "%Y-%m-%d")
                    df = df[df['Tarih'] >= start_dt]
                
                if end_date:
                    end_dt = datetime.strptime(str(end_date), "%Y-%m-%d")
                    df = df[df['Tarih'] <= end_dt]
                
                if category:
                    df = df[df['Kategori'] == str(category).title()]
                
                if user:
                    df = df[df['User'] == user]
                
                # Convert to pandas if GPU
                return df.to_pandas() if self.use_gpu else df
            
            except Exception as e:
                logger.error(f"Filtreleme hatası: {e}")
                return pd.DataFrame()
    
    def get_category_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Kategori özeti
        
        Returns:
            Kategori bazlı gelir/gider dict
        """
        with self.lock:
            try:
                df = self._load_data_to_engine()
                
                if df.empty:
                    return {}
                
                summary = df.groupby(['Kategori', 'Tur'])['Tutar'].sum()
                
                # Convert to dict
                if self.use_gpu:
                    res_pd = summary.to_pandas().unstack(fill_value=0)
                else:
                    res_pd = summary.unstack(fill_value=0)
                
                return res_pd.to_dict(orient='index')
            
            except Exception:
                return {}
    
    def get_recent_transactions(self, limit: int = 5) -> str:
        """
        Son işlemler
        
        Args:
            limit: Maksimum kayıt
        
        Returns:
            Formatlanmış işlem listesi
        """
        with self.lock:
            try:
                if not self.filename.exists():
                    return "Kayıt bulunamadı"
                
                df = pd.read_csv(self.filename)
                
                if df.empty:
                    return "Henüz işlem kaydı yok"
                
                last_rows = df.tail(limit).iloc[::-1]
                result = []
                
                for _, row in last_rows.iterrows():
                    icon = "🟢" if row['Tur'] == "GELIR" else "🔴"
                    time_str = str(row['Tarih'])[:16]
                    result.append(
                        f"{icon} [{time_str}] {row['Aciklama']} "
                        f"({row['Tutar']:,.2f} TL) | {row['User']}"
                    )
                
                return "\n".join(result)
            
            except Exception:
                return "İşlem geçmişi okunamadı"
    
    def get_report(self) -> str:
        """
        Detaylı finansal rapor
        
        Returns:
            Formatlanmış rapor
        """
        with self.lock:
            try:
                df = self._load_data_to_engine()
                
                if df.empty:
                    return "ℹ️ Kasa defterinde kayıt yok"
                
                summary = self.get_summary()
                
                report = [
                    "📊 LOTUSAI FİNANSAL DURUM RAPORU",
                    "═" * 40,
                    f"💰 Mevcut Kasa: {summary.balance:,.2f} TL",
                    f"📈 Toplam Gelir: {summary.total_income:,.2f} TL",
                    f"📉 Toplam Gider: {summary.total_expense:,.2f} TL",
                    "─" * 40,
                    f"📅 Son 30 Günlük Performans:",
                    f"   Giriş: +{summary.monthly_income:,.2f} TL",
                    f"   Çıkış: -{summary.monthly_expense:,.2f} TL",
                    f"   Net:   {summary.monthly_net:,.2f} TL",
                    "─" * 40,
                    f"📝 Son İşlemler:\n{self.get_recent_transactions(3)}",
                    "═" * 40
                ]
                
                return "\n".join(report)
            
            except Exception as e:
                return f"❌ Rapor oluşturulamadı: {e}"
    
    # ───────────────────────────────────────────────────────────
    # EXPORT
    # ───────────────────────────────────────────────────────────
    
    def export_to_excel(
        self,
        target_path: Optional[Union[str, Path]] = None
    ) -> Optional[str]:
        """
        Excel'e aktar
        
        Args:
            target_path: Hedef dosya yolu
        
        Returns:
            Dosya yolu veya None
        """
        try:
            if not target_path:
                timestamp = datetime.now().strftime("%Y%m%d")
                target_path = self.work_dir / f"Lotus_Finans_{timestamp}.xlsx"
            
            df = pd.read_csv(self.filename)
            df.to_excel(target_path, index=False)
            
            logger.info(f"📁 Excel'e aktarıldı: {target_path}")
            return str(target_path)
        
        except Exception as e:
            logger.error(f"Excel export hatası: {e}")
            return None
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Muhasebe metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "total_transactions": self.metrics.total_transactions,
            "income_transactions": self.metrics.income_transactions,
            "expense_transactions": self.metrics.expense_transactions,
            "backups_created": self.metrics.backups_created,
            "recoveries_performed": self.metrics.recoveries_performed,
            "errors_encountered": self.metrics.errors_encountered,
            "gpu_enabled": self.use_gpu,
            "current_balance": self.get_balance()
        }