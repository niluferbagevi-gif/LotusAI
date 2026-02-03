import os
import sys
import subprocess
import shlex
import shutil
import time
import logging
import threading
import re
import fnmatch
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Dict
from config import Config

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.CodeManager")

class CodeManager:
    """
    LotusAI Dosya, Terminal ve Geliştirme Yöneticisi.
    Sürüm 2.6 - GPU İzleme ve Gelişmiş Sistem Raporlama Destekli
    """
    
    def __init__(self, work_dir: Optional[Union[str, Path]] = None):
        # Sandbox (Güvenli Alan) sınırlarını belirle
        self.root_dir = Path(work_dir).resolve() if work_dir else Path(Config.WORK_DIR).resolve()
        self.backup_dir = self.root_dir / "backups" / "code"
        
        # Çoklu ajan erişimi için Reentrant Lock (Yarış durumlarını önler)
        self.lock = threading.RLock()
        
        # Gerekli klasörleri oluştur
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Dizin oluşturma hatası: {e}")
        
        # İzin verilen güvenli terminal komutları
        self.allowed_commands = {
            "ls", "dir", "git", "python", "pip", "echo", "date", "whoami", 
            "type", "cat", "mkdir", "cd", "touch", "where", "which", 
            "pytest", "npm", "node", "tree", "find", "grep", "nvidia-smi"
        }
        
        # Filtrelenecek (görünmemesi gereken) dizin ve dosyalar
        self.exclude_dirs = {'.git', '__pycache__', 'backups', 'lotus_vector_db', 'venv', 'env', 'node_modules', 'faces', 'voices', '.pytest_cache'}
        self.exclude_files = {'lotus_system.db', '.env', '.DS_Store', 'users_db.json.backup', 'out.wav', 'launcher_error.log'}

        logger.info(f"✅ CodeManager aktif. Güvenli Bölge: {self.root_dir}")

    def _is_safe_path(self, path: Path) -> bool:
        """Dosya yolunun sandbox içinde olup olmadığını kontrol eder."""
        try:
            # Hem mutlak yolu al hem de sembolik linkleri çöz
            resolved_path = path.resolve()
            # Kök dizine göre göreceli mi kontrol et
            return resolved_path.is_relative_to(self.root_dir)
        except (ValueError, Exception):
            return False

    # --- DOSYA SİSTEMİ YÖNETİMİ ---

    def list_files(self, pattern: str = "*", recursive: bool = True) -> str:
        """Proje dizinindeki dosyaları listeler (Güvenlik filtreli)."""
        with self.lock:
            try:
                files = []
                search_func = self.root_dir.rglob if recursive else self.root_dir.glob
                
                for path in search_func(pattern):
                    # Gizli veya sistem dizinlerini atla
                    if any(part in self.exclude_dirs for part in path.parts):
                        continue
                    
                    if path.is_file() and path.name not in self.exclude_files:
                        # Sadece işlenebilir metin tabanlı dosyaları listele
                        if path.suffix in ('.py', '.txt', '.md', '.json', '.html', '.css', '.js', '.yaml', '.yml', '.sql', '.sh'):
                            rel_path = path.relative_to(self.root_dir)
                            files.append(str(rel_path))
                
                return "\n".join(sorted(files)) if files else "🔍 Eşleşen dosya bulunamadı."
            except Exception as e:
                logger.error(f"Listeleme hatası: {e}")
                return f"❌ Listeleme hatası: {str(e)}"

    def read_file(self, filename: str) -> str:
        """Dosya içeriğini güvenli bir şekilde okur."""
        with self.lock:
            try:
                # "Bu dosya" veya "kendini oku" gibi talepleri yönet
                if any(k in filename.lower() for k in ["bu dosya", "kendini", "self"]):
                    target_path = Path(sys.argv[0]).resolve()
                else:
                    target_path = (self.root_dir / filename.strip()).resolve()
                
                # GÜVENLİK: Sandbox kontrolü
                if not self._is_safe_path(target_path) and "bu dosya" not in filename.lower():
                    logger.warning(f"🚫 Yasaklı bölge erişim denemesi: {target_path}")
                    return "[GÜVENLİK]: Proje dizini dışındaki dosyalara erişim yetkiniz yok."
                
                if not target_path.exists():
                    return f"❌ HATA: '{filename}' dosyası bulunamadı."
                    
                if target_path.is_dir():
                    return "❌ HATA: Bu bir klasördür, lütfen list_files kullanın."

                return target_path.read_text(encoding="utf-8", errors="replace")
                
            except Exception as e:
                logger.error(f"Okuma hatası: {e}")
                return f"❌ Okuma hatası: {str(e)}"

    def save_file(self, filename: str, content: str) -> str:
        """Dosyayı yedek alarak kaydeder veya günceller."""
        with self.lock:
            try:
                target_path = (self.root_dir / filename.strip()).resolve()

                # Güvenlik Kontrolü
                if not self._is_safe_path(target_path):
                    return "[GÜVENLİK]: Sandbox dışına dosya yazma yetkiniz yok."

                # Alt klasörleri otomatik oluştur
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Yedekleme (Dosya varsa)
                if target_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"{target_path.stem}_{timestamp}{target_path.suffix}.bak"
                    backup_path = self.backup_dir / backup_name
                    shutil.copy2(target_path, backup_path)

                # Yazma işlemi
                target_path.write_text(content, encoding="utf-8")
                
                logger.info(f"💾 Dosya Güncellendi: {filename}")
                return f"✅ Başarıyla kaydedildi: {filename} (Yedek alındı)"
                
            except Exception as e:
                logger.error(f"Yazma hatası: {e}")
                return f"❌ Yazma hatası: {str(e)}"

    def delete_file(self, filename: str) -> str:
        """Dosyayı kalıcı olarak silmeden önce yedeğini alır."""
        with self.lock:
            try:
                target_path = (self.root_dir / filename.strip()).resolve()

                if not self._is_safe_path(target_path):
                    return "[GÜVENLİK]: Sandbox dışındaki dosyaları silemezsiniz."

                if not target_path.exists():
                    return f"❌ HATA: Silinecek dosya bulunamadı: {filename}"

                # Silmeden önce son bir yedek al
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.backup_dir / f"DELETED_{target_path.name}_{timestamp}.bak"
                shutil.copy2(target_path, backup_path)

                target_path.unlink() # Dosyayı sil
                logger.warning(f"🗑️ Dosya Silindi: {filename}")
                return f"✅ Dosya silindi ve yedeği alındı: {filename}"
            except Exception as e:
                return f"❌ Silme hatası: {str(e)}"

    def search_code(self, query: str, is_regex: bool = False, file_ext: str = "*.py") -> str:
        """Proje içinde metin veya Regex ile arama yapar."""
        with self.lock:
            results = []
            try:
                for path in self.root_dir.rglob(file_ext):
                    if any(part in self.exclude_dirs for part in path.parts):
                        continue
                    
                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        match = False
                        if is_regex:
                            if re.search(query, content, re.IGNORECASE):
                                match = True
                        elif query.lower() in content.lower():
                            match = True
                            
                        if match:
                            rel_path = path.relative_to(self.root_dir)
                            results.append(str(rel_path))
                    except:
                        continue
                
                if results:
                    return f"🔍 '{query}' ifadesi şu dosyalarda bulundu:\n" + "\n".join(results)
                return "🔍 Eşleşen sonuç bulunamadı."
            except Exception as e:
                return f"❌ Arama hatası: {str(e)}"

    # --- TERMİNAL VE GPU YÖNETİMİ ---

    def run_terminal(self, command: str, timeout: int = 45) -> str:
        """Güvenli komut listesi üzerinden terminal komutu çalıştırır."""
        with self.lock:
            try:
                # Komutu güvenli parçala
                if os.name == 'nt':
                    cmd_parts = command.split()
                else:
                    cmd_parts = shlex.split(command)
                    
                if not cmd_parts: return "⚠️ Komut girmediniz."
                
                base_cmd = cmd_parts[0].lower()
                
                # Gelişmiş Güvenlik: Yasaklı karakter kontrolü
                illegal_chars = [";", "&&", "||", ">", ">>", "|"]
                if any(char in command for char in illegal_chars):
                    return "🚫 [GÜVENLİK]: Zincirleme komutlar veya yönlendirmeler yasaktır."

                if base_cmd not in self.allowed_commands:
                    return f"🚫 [GÜVENLİK]: '{base_cmd}' komutuna izniniz yok."

                # Windows yerleşik komutları için shell kontrolü
                use_shell = os.name == 'nt' and base_cmd in ['dir', 'echo', 'type', 'mkdir', 'date', 'tree']

                # Komutu çalıştır
                result = subprocess.run(
                    command if use_shell else cmd_parts, 
                    capture_output=True, 
                    text=True, 
                    cwd=str(self.root_dir), 
                    timeout=timeout,
                    shell=use_shell
                )
                
                output = result.stdout
                if result.stderr:
                    output += f"\n[HATA ÇIKTISI]: {result.stderr}"
                
                logger.info(f"💻 Terminal İşlemi: {command}")
                return f"--- TERMİNAL ÇIKTISI ---\n{output.strip()}" if output.strip() else "✅ İşlem tamamlandı."
                
            except subprocess.TimeoutExpired:
                return f"⏱️ ZAMAN AŞIMI: İşlem {timeout} saniyeyi geçtiği için durduruldu."
            except Exception as e:
                logger.error(f"Terminal hatası: {e}")
                return f"❌ Sistem hatası: {str(e)}"

    def get_gpu_info(self) -> str:
        """Sistemdeki NVIDIA GPU durumunu sorgular."""
        try:
            # nvidia-smi komutunu dene
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = result.stdout.strip().split(", ")
                return (f"🚀 GPU: {data[0]}\n"
                        f"📊 Bellek: {data[2]}MB / {data[1]}MB Boş\n"
                        f"🔥 Yük: %{data[3]}")
            return "ℹ️ GPU: NVIDIA sürücüsü bulunamadı veya GPU yok."
        except:
            return "ℹ️ GPU: Sistemde aktif GPU tespit edilemedi."

    def get_file_info(self, filename: str) -> str:
        """Dosya hakkında detaylı bilgi döner."""
        try:
            target_path = (self.root_dir / filename.strip()).resolve()
            if not self._is_safe_path(target_path) or not target_path.exists():
                return "❌ Dosya bulunamadı veya erişim yasak."
            
            stats = target_path.stat()
            size_kb = round(stats.st_size / 1024, 2)
            mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            return (f"Dosya: {filename}\n"
                    f"Boyut: {size_kb} KB\n"
                    f"Son Değişiklik: {mod_time}\n"
                    f"Tür: {target_path.suffix}")
        except Exception as e:
            return f"❌ Bilgi alma hatası: {e}"

    def get_system_info(self) -> str:
        """Ajanlar için çalışma ortamı özeti (GPU Destekli)."""
        gpu_status = self.get_gpu_info()
        return (f"🖥️ Sistem: {sys.platform}\n"
                f"🐍 Python: {sys.version.split()[0]}\n"
                f"{gpu_status}\n"
                f"📁 Çalışma Dizini: {self.root_dir}\n"
                f"🛡️ Sandbox Durumu: AKTİF")