import re
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import torch  # GPU desteği ve donanım kontrolü için
from config import Config

# --- LOGGING ---
logger = logging.getLogger("LotusAI.Kerberos")

class KerberosAgent:
    """
    Kerberos (Security and Financial Audit Chief) - LotusAI System Guardian.
    
    Capabilities:
    - Field Audit: Instant identity and threat analysis via camera.
    - Financial Audit: Monitors cash movements, audits budget discipline.
    - Anomaly Detection: Reports movements at suspicious hours and high-risk expenditures.
    - Authority: Ensures security by manipulating SystemState in critical situations.
    - Hardware Monitoring: Monitors GPU/CPU health and manages hardware-accelerated tasks.
    """
    
    def __init__(self, tools_dict: Dict[str, Any]):
        """
        Initializes the Kerberos module with GPU awareness.
        :param tools_dict: Tool pool provided by the Engine (camera, accounting, state, etc.).
        """
        self.tools = tools_dict
        self.agent_name = "KERBEROS"
        self.lock = threading.RLock()
        
        # --- GPU / HARDWARE CONFIGURATION ---
        # Detect if CUDA is available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = torch.cuda.device_count() if self.device.type == "cuda" else 0
        
        # Audit Thresholds
        self.high_expense_threshold = getattr(Config, 'HIGH_EXPENSE_THRESHOLD', 2000.0)
        self.working_hours = (8, 22) # Normal working hours between 08:00 - 22:00
        
        logger.info(f"🛡️ {self.agent_name} Security and Audit module active on {self.device.type.upper()}.")
        if self.gpu_count > 0:
            logger.info(f"🚀 GPU Acceleration active: {torch.cuda.get_device_name(0)} detected.")

    def get_system_prompt(self) -> str:
        """
        System instruction defining Kerberos's personality and philosophy.
        """
        return (
            f"Sen {Config.PROJECT_NAME} sisteminin sert, şüpheci ve korumacı Güvenlik Şefi KERBEROS'sun. "
            "Görevin: Halil Bey'in (Patron) kaynaklarını ve dijital güvenliğini her şeyin üstünde tutmak. "
            "Karakterin: Disiplinli, iğneleyici, taviz vermeyen ve son derece dikkatli. "
            "Harcamaları kuruşu kuruşuna sorgula, yüksek harcamalarda eleştirel bir ton kullan. "
            "Güvenlik açıklarını asla küçümseme, her zaman en kötü senaryoyu düşünerek tedbir al. "
            "Halil Bey'e sadıksın ama sistemin selameti için gerekirse onu da uyarabilirsin. "
            f"Sistem şu an {self.device.type.upper()} üzerinde çalışıyor, teknik performans takibi senin sorumluluğunda."
        )

    def get_context_data(self) -> str:
        """
        Prepares a security and financial status report of the system through the eyes of Kerberos.
        """
        context_parts = ["\n[🛡️ KERBEROS DENETİM RAPORU]"]
        
        with self.lock:
            # 1. Hardware Status (GPU/CPU Check)
            hw_info = f"⚙️ DONANIM: {self.device.type.upper()}"
            if self.device.type == "cuda":
                memory_usage = torch.cuda.memory_allocated(0) / 1024**2
                hw_info += f" | VRAM Kullanımı: {memory_usage:.2f} MB"
            context_parts.append(hw_info)

            # 2. Live Security Analysis (SecurityManager Integration)
            if 'security' in self.tools:
                try:
                    # Security manager might use the device set here
                    status, user, info = self.tools['security'].analyze_situation()
                    user_name = user.get('name', 'Bilinmiyor') if user else "Görüş Alanı Boş"
                    
                    if status == "SORGULAMA":
                        context_parts.append(f"🚨 UYARI: Sahada tanınmayan bir yabancı var! Kimlik tespiti yapılamadı.")
                    elif status == "ONAYLI":
                        context_parts.append(f"👤 TAKİP: {user_name} şu an görüş alanında. Hareketlerini izliyorum.")
                    else:
                        context_parts.append("✅ DURUM: Çevrede tehdit yok, bölge temiz.")
                except Exception as e:
                    logger.debug(f"Kerberos security context error: {e}")

            # 3. Financial Audit (AccountingManager Integration)
            acc_tool = self.tools.get('accounting') or self.tools.get('finance')
            if acc_tool:
                try:
                    if hasattr(acc_tool, 'get_balance'):
                        balance = acc_tool.get_balance()
                        context_parts.append(f"💰 KASA: {balance:,.2f} TL mevcut. Gereksiz harcamalardan kaçınılmalı.")
                    
                    if hasattr(acc_tool, 'get_recent_transactions'):
                        recent = acc_tool.get_recent_transactions(limit=2)
                        if "Kayıt yok" not in str(recent):
                            context_parts.append(f"📝 SON HAREKETLER:\n{recent}")
                except Exception as e:
                    logger.debug(f"Kerberos financial context error: {e}")

        return "\n".join(context_parts)

    def _clean_amount(self, raw_val: Any) -> float:
        """Converts text-based amount information to numbers (Minimizes margin of error)."""
        if isinstance(raw_val, (int, float)): return float(raw_val)
        try:
            clean = str(raw_val).lower().replace("tl", "").replace(",", ".").strip()
            clean = "".join(c for c in clean if c.isdigit() or c == '.')
            return float(clean) if clean else 0.0
        except: return 0.0

    def audit_invoice(self, invoice_data: Dict[str, Any]) -> str:
        """
        Audits the invoice from Gaya, performs risk analysis, and processes it into the system.
        """
        if not invoice_data:
            return "🛡️ REDDEDİLDİ: Boş veri denetlenemez!"

        firma = invoice_data.get("firma", "Bilinmeyen Firma")
        tutar = self._clean_amount(invoice_data.get("toplam_tutar", 0))
        
        with self.lock:
            acc_tool = self.tools.get('accounting') or self.tools.get('finance')
            
            # 1. Risk Assessment
            audit_comment = ""
            risk_level = "Düşük"
            
            if tutar >= self.high_expense_threshold:
                risk_level = "YÜKSEK"
                audit_comment = f"⚠️ Halil Bey, bu miktar ({tutar} TL) bütçeyi sarsabilir! Onay veriyor musunuz?"
                if 'state' in self.tools:
                    self.tools['state'].set_state(4, reason=f"Yüksek Gider Denetimi: {firma}")
            elif tutar <= 0:
                risk_level = "KRİTİK"
                return "❌ DENETİM BAŞARISIZ: Tutar sıfır veya negatif. Bu fatura şüpheli!"

            # 2. Accounting Processing
            acc_status = "Muhasebe modülü kapalı."
            if acc_tool and hasattr(acc_tool, 'add_entry'):
                try:
                    success = acc_tool.add_entry(
                        tur="GIDER",
                        aciklama=f"Kerberos Denetimli: {firma}",
                        tutar=tutar,
                        kategori=invoice_data.get("kategori", "Genel"),
                        user_id="KERBEROS"
                    )
                    acc_status = "✅ Kayıt doğrulandı ve deftere işlendi." if success else "❌ Kayıt başarısız!"
                except Exception as e:
                    acc_status = f"❌ Sistem Hatası: {e}"

            # 3. Final Report
            report = [
                f"🛡️ KERBEROS DENETİM RAPORU (Risk: {risk_level})",
                f"{'='*35}",
                f"🏢 KURUM: {firma}",
                f"💸 TUTAR: {tutar:,.2f} TL",
                f"📅 TARİH: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                f"⚙️ İŞLEMCİ: {self.device.type.upper()}",
                f"{'-'*35}",
                f"SİSTEM: {acc_status}",
                f"NOT: {audit_comment if audit_comment else 'İşlem makul, onaylandı.'}"
            ]
            return "\n".join(report)

    def check_security_anomaly(self) -> Optional[str]:
        """
        Checks for anomalies in the system (Night activity, intruder detection, hardware health, etc.).
        """
        with self.lock:
            hour = datetime.now().hour
            
            # 1. Midnight Activity
            if hour < self.working_hours[0] or hour > self.working_hours[1]:
                if 'security' in self.tools:
                    status, user, _ = self.tools['security'].analyze_situation()
                    if status in ["ONAYLI", "SORGULAMA"]:
                        return f"🚨 ANOMALİ: Saat {hour}:00 civarında sahada hareketlilik tespit ettim!"

            # 2. Hardware Resource Control (CPU & GPU)
            if 'system' in self.tools:
                health = self.tools['system'].get_resource_stats()
                
                # CPU Check
                if health.get('cpu_percent', 0) > 90:
                    return "⚠️ SİSTEM YORGUN: CPU kullanımı %90'ı aştı. Bazı süreçleri durdurmamı ister misiniz?"
                
                # GPU Check (If CUDA is active)
                if self.device.type == "cuda":
                    try:
                        # Note: Simple health check via torch
                        # In a more advanced version, we could use NVML for temperature
                        if torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory > 0.95:
                            return "🔥 KRİTİK: GPU VRAM neredeyse tamamen dolu! Sistem yavaşlayabilir."
                    except:
                        pass
            
        return None

# import random

# class KerberosAgent:
#     """
#     Kerberos (Güvenlik ve Mali Denetim Şefi) için özel yetenekleri yöneten sınıf.
#     Görevi: Kameradaki kişiyi tanımak, kasadaki son hareketleri denetlemek ve faturaları sorgulayarak işlemek.
#     """
#     def __init__(self, tools_dict):
#         self.tools = tools_dict

#     def get_context_data(self):
#         """
#         Kerberos için güvenlik ve muhasebe özetini hazırlar.
#         Bu veriler LLM'e (Yapay Zekaya) gönderilerek Kerberos'un güncel durumdan haberdar olmasını sağlar.
#         """
#         context_parts = []
        
#         # 1. Kamera / Güvenlik Kontrolü
#         if 'camera' in self.tools:
#             try:
#                 cam_tool = self.tools['camera']
#                 # Eğer son görülen kişi bilgisi varsa
#                 if hasattr(cam_tool, 'last_seen_person') and cam_tool.last_seen_person:
#                     person = cam_tool.last_seen_person
#                     context_parts.append(f"\n### GÜVENLİK KAMERASI RAPORU ###\nAnlık Durum: Kamerada '{person}' tespit edildi. Gözünü üzerinden ayırma.")
#                 else:
#                     context_parts.append(f"\n### GÜVENLİK KAMERASI RAPORU ###\nKamera aktif, şu an tanınan bir tehdit veya kişi yok.")
#             except Exception as e:
#                 # Hata olursa Yapay Zeka bunu bilmeli
#                 context_parts.append(f"\n### GÜVENLİK UYARISI ###\nKamera sistemine erişilemiyor! Hata: {e}")
#         else:
#             # Eğer kamera aracı hiç yüklenmemişse
#             context_parts.append("\n### GÜVENLİK UYARISI ###\nKamera modülü devre dışı! Kör noktasın.")

#         # 2. Muhasebe (Son Harcamalar)
#         # Accounting veya Finance yöneticisini bul
#         acc_tool = self.tools.get('accounting') or self.tools.get('finance')
        
#         if acc_tool:
#             try:
#                 # Son 3 işlemi getir (Harcamaları kontrol etmek için)
#                 if hasattr(acc_tool, 'get_recent_transactions'):
#                     recent = acc_tool.get_recent_transactions(limit=3)
                    
#                     # Eğer veri varsa ve "Kayıt yok" yazmıyorsa raporla
#                     if recent and "Kayıt yok" not in recent:
#                         context_parts.append(f"\n### SON KASA HAREKETLERİ (DENETLE) ###\n{recent}\n(Bu harcamaları gereksizse sert bir dille eleştir.)")
#                     else:
#                         context_parts.append(f"\n### SON KASA HAREKETLERİ ###\nHenüz işlem yok. Gözüm üzerinde.")
#             except Exception as e:
#                 print(f"Kerberos Muhasebe Hatası: {e}")

#         return "".join(context_parts)

#     def audit_invoice(self, invoice_data):
#         """
#         Fatura/Fiş verilerini denetler ve muhasebeye işler.
#         Kerberos karakterine uygun olarak harcamayı yargılar.
#         """
#         if not invoice_data:
#             return "Fiş okunamadı! Bulanık mı çektiniz? Düzgün gönderin."

#         firma = invoice_data.get("firma", "Bilinmeyen Firma")
#         tutar_str = invoice_data.get("toplam_tutar", "0")
        
#         # Tutar temizleme (TL ve boşlukları at)
#         try:
#             clean_tutar = float(str(tutar_str).replace("TL", "").replace(".", "").replace(",", ".").strip())
#         except:
#             clean_tutar = 0.0

#         acc_msg = ""
#         audit_comment = ""

#         # 1. Muhasebeye İşle
#         if 'accounting' in self.tools:
#             try:
#                 self.tools['accounting'].add_transaction(
#                     description=f"Fatura: {firma}",
#                     amount=clean_tutar,
#                     type="GIDER",
#                     category="Operasyon"
#                 )
#                 acc_msg = "✅ Tutar kasadan düşüldü."
#             except Exception as e:
#                 acc_msg = f"❌ Kayıt Hatası: {e}"

#         # 2. Kerberos Yorumu (Denetim)
#         if clean_tutar > 1000:
#             audit_comment = f"⚠️ {clean_tutar} TL mi? Bu harcama gerçekten gerekli miydi Halil Bey? Para kolay kazanılmıyor!"
#         elif clean_tutar > 0:
#             audit_comment = "Onaylandı. Ama gereksiz harcamalardan kaçınalım."
#         else:
#             audit_comment = "Tutar okunamadı, manuel kontrol gerekli."

#         return (
#             f"🛡️ MALİ DENETİM RAPORU:\n"
#             f"🏢 Firma: {firma}\n"
#             f"💸 Tutar: {tutar_str} TL\n"
#             f"--------------------------\n"
#             f"{acc_msg}\n"
#             f"🗣️ Kerberos Görüşü: {audit_comment}"
#         )


# class KerberosAgent:
#     """
#     Kerberos (Güvenlik ve Muhasebe Şefi) için özel yetenekleri yöneten sınıf.
#     Görevi: Kameradaki kişiyi tanımak ve kasadaki son hareketleri denetlemek.
#     """
#     def __init__(self, tools_dict):
#         self.tools = tools_dict

#     def get_context_data(self):
#         """
#         Kerberos için güvenlik ve muhasebe özetini hazırlar.
#         Bu veriler LLM'e (Yapay Zekaya) gönderilerek Kerberos'un güncel durumdan haberdar olmasını sağlar.
#         """
#         context_parts = []
        
#         # 1. Kamera / Güvenlik Kontrolü
#         if 'camera' in self.tools:
#             try:
#                 cam_tool = self.tools['camera']
#                 # Eğer son görülen kişi bilgisi varsa
#                 if hasattr(cam_tool, 'last_seen_person') and cam_tool.last_seen_person:
#                     person = cam_tool.last_seen_person
#                     context_parts.append(f"\n### GÜVENLİK KAMERASI RAPORU ###\nAnlık Durum: Kamerada '{person}' tespit edildi. Gözünü üzerinden ayırma.")
#                 else:
#                     context_parts.append(f"\n### GÜVENLİK KAMERASI RAPORU ###\nKamera aktif, şu an tanınan bir tehdit veya kişi yok.")
#             except Exception as e:
#                 # Hata olursa Yapay Zeka bunu bilmeli
#                 context_parts.append(f"\n### GÜVENLİK UYARISI ###\nKamera sistemine erişilemiyor! Hata: {e}")
#         else:
#             # Eğer kamera aracı hiç yüklenmemişse
#             context_parts.append("\n### GÜVENLİK UYARISI ###\nKamera modülü devre dışı! Kör noktasın.")

#         # 2. Muhasebe (Son Harcamalar)
#         # Accounting veya Finance yöneticisini bul
#         acc_tool = self.tools.get('accounting') or self.tools.get('finance')
        
#         if acc_tool:
#             try:
#                 # Son 3 işlemi getir (Harcamaları kontrol etmek için)
#                 if hasattr(acc_tool, 'get_recent_transactions'):
#                     recent = acc_tool.get_recent_transactions(limit=3)
                    
#                     # Eğer veri varsa ve "Kayıt yok" yazmıyorsa raporla
#                     if recent and "Kayıt yok" not in recent:
#                         context_parts.append(f"\n### SON KASA HAREKETLERİ ###\n{recent}\n(Bu harcamaları gereksizse eleştir.)")
#                     else:
#                         context_parts.append(f"\n### SON KASA HAREKETLERİ ###\nHenüz yeni işlem yok.")
#             except Exception as e:
#                 print(f"Kerberos Muhasebe Hatası: {e}")
            
#         return "\n".join(context_parts)