"""
LotusAI Code Manager
Sürüm: 2.5.4 (Eklendi: Erişim Seviyesi Desteği)
Açıklama: Dosya, terminal ve geliştirme yönetimi

Özellikler:
- Sandbox güvenliği
- Dosya operasyonları (erişim seviyesine göre kısıtlı)
- Terminal komutu çalıştırma (erişim seviyesine göre whitelist)
- GPU bilgisi
- Otomatik yedekleme
- Kod arama
- Thread-safe operasyonlar
"""

import os
import sys
import subprocess
import shlex
import shutil
import logging
import threading
import re
import ast
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from config import Config, AccessLevel

logger = logging.getLogger("LotusAI.CodeManager")


# ═══════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════
class FileType(Enum):
    """Dosya tipleri"""
    PYTHON = ".py"
    TEXT = ".txt"
    MARKDOWN = ".md"
    JSON = ".json"
    YAML = ".yaml"
    HTML = ".html"
    CSS = ".css"
    JAVASCRIPT = ".js"
    SQL = ".sql"
    SHELL = ".sh"
    OTHER = ""


class CommandStatus(Enum):
    """Komut durumları"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    FORBIDDEN = "forbidden"


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class FileInfo:
    """Dosya bilgisi"""
    path: Path
    size_kb: float
    modified: datetime
    file_type: str
    lines: int = 0


@dataclass
class SearchResult:
    """Arama sonucu"""
    file_path: str
    line_number: int
    line_content: str
    match_count: int


@dataclass
class CodeManagerMetrics:
    """Code manager metrikleri"""
    files_read: int = 0
    files_written: int = 0
    files_deleted: int = 0
    backups_created: int = 0
    commands_executed: int = 0
    searches_performed: int = 0
    errors_encountered: int = 0


# ═══════════════════════════════════════════════════════════════
# CODE MANAGER
# ═══════════════════════════════════════════════════════════════
class CodeManager:
    """
    LotusAI Dosya, Terminal ve Geliştirme Yöneticisi
    
    Yetenekler:
    - Dosya sistemi yönetimi (okuma, yazma, silme)
    - Sandbox güvenliği (proje dizini dışına çıkamaz)
    - Terminal komutu çalıştırma (whitelist + erişim seviyesi)
    - GPU bilgisi sorgulama
    - Otomatik yedekleme
    - Kod arama (regex destekli)
    - Thread-safe operasyonlar
    
    Güvenlik:
    - Sandbox: Sadece proje dizini içinde çalışır
    - Command whitelist: Sadece güvenli komutlar
    - Path validation: Sembolik link kontrolü
    - Erişim seviyesi: restricted (sadece okuma), sandbox (okuma+yazma), full (tüm yetkiler)
    """
    
    # Allowed extensions
    ALLOWED_EXTENSIONS = {
        '.py', '.txt', '.md', '.json', '.html', '.css', '.js',
        '.yaml', '.yml', '.sql', '.sh', '.jsx', '.tsx', '.vue'
    }
    
    # Allowed commands (full mod için geniş, sandbox için daha kısıtlı)
    ALLOWED_COMMANDS_FULL = {
        "ls", "dir", "git", "python", "pip", "echo", "date",
        "whoami", "type", "cat", "mkdir", "cd", "touch",
        "where", "which", "pytest", "npm", "node", "tree",
        "find", "grep", "nvidia-smi", "ps", "top", "htop"
    }
    
    # Sandbox modunda sadece okuma amaçlı komutlar
    ALLOWED_COMMANDS_SANDBOX = {
        "ls", "dir", "echo", "date", "whoami", "type", "cat",
        "where", "which", "tree", "find", "grep", "nvidia-smi"
    }
    
    # Restricted modda komut çalıştırma yasak
    ALLOWED_COMMANDS_RESTRICTED = set()
    
    # Exclude patterns
    EXCLUDE_DIRS = {
        '.git', '__pycache__', 'backups', 'lotus_vector_db',
        'venv', 'env', 'node_modules', 'faces', 'voices',
        '.pytest_cache', 'dist', 'build', '.vscode'
    }
    
    EXCLUDE_FILES = {
        'lotus_system.db', '.env', '.DS_Store',
        'users_db.json.backup', 'out.wav', 'launcher_error.log',
        '*.pyc', '*.pyo', '*.pyd'
    }
    
    # Command timeout
    DEFAULT_TIMEOUT = 45
    
    # Illegal command characters
    ILLEGAL_CHARS = [";", "&&", "||", ">", ">>", "|", "`", "$"]
    
    def __init__(self, work_dir: Optional[Union[str, Path]] = None, access_level: str = "sandbox"):
        """
        Code manager başlatıcı
        
        Args:
            work_dir: Çalışma dizini (None ise Config.WORK_DIR)
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.access_level = access_level
        
        # Sandbox root
        if work_dir:
            self.root_dir = Path(work_dir).resolve()
        else:
            self.root_dir = Config.WORK_DIR.resolve()
        
        self.backup_dir = self.root_dir / "backups" / "code"
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metrics
        self.metrics = CodeManagerMetrics()
        
        # Create directories
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Dizin oluşturma hatası: {e}")
        
        logger.info(f"✅ CodeManager aktif (Sandbox: {self.root_dir}, Erişim: {self.access_level})")
    
    # ───────────────────────────────────────────────────────────
    # SECURITY
    # ───────────────────────────────────────────────────────────
    
    def _is_safe_path(self, path: Path) -> bool:
        """
        Path güvenlik kontrolü
        
        Args:
            path: Kontrol edilecek path
        
        Returns:
            Güvenliyse True
        """
        try:
            # Resolve sembolik linkleri
            resolved_path = path.resolve()
            
            # Sandbox içinde mi?
            return resolved_path.is_relative_to(self.root_dir)
        
        except (ValueError, Exception):
            return False
    
    def _is_allowed_extension(self, path: Path) -> bool:
        """Dosya uzantısı izinli mi"""
        return path.suffix.lower() in self.ALLOWED_EXTENSIONS
    
    def _check_write_permission(self, operation: str) -> bool:
        """
        Yazma izni kontrolü (sandbox veya full gerekir)
        
        Args:
            operation: İşlem adı (log için)
        
        Returns:
            İzin varsa True
        """
        if self.access_level == AccessLevel.RESTRICTED:
            logger.warning(f"🚫 {operation}: Kısıtlı modda yazma izni yok")
            return False
        return True
    
    def _check_delete_permission(self, operation: str) -> bool:
        """
        Silme izni kontrolü (sadece full modda)
        
        Args:
            operation: İşlem adı (log için)
        
        Returns:
            İzin varsa True
        """
        if self.access_level != AccessLevel.FULL:
            logger.warning(f"🚫 {operation}: Sadece full modda silme izni var")
            return False
        return True
    
    def _get_allowed_commands(self) -> set:
        """Erişim seviyesine göre izinli komutları döndür"""
        if self.access_level == AccessLevel.FULL:
            return self.ALLOWED_COMMANDS_FULL
        elif self.access_level == AccessLevel.SANDBOX:
            return self.ALLOWED_COMMANDS_SANDBOX
        else:  # restricted
            return self.ALLOWED_COMMANDS_RESTRICTED
    
    # ───────────────────────────────────────────────────────────
    # FILE OPERATIONS
    # ───────────────────────────────────────────────────────────
    
    def list_files(
        self,
        pattern: str = "*",
        recursive: bool = True
    ) -> str:
        """
        Dosyaları listele (Tüm erişim seviyelerine açık)
        
        Args:
            pattern: Glob pattern
            recursive: Recursive arama
        
        Returns:
            Dosya listesi (satır satır)
        """
        with self.lock:
            try:
                files = []
                search_func = self.root_dir.rglob if recursive else self.root_dir.glob
                
                for path in search_func(pattern):
                    # Exclude check
                    if any(part in self.EXCLUDE_DIRS for part in path.parts):
                        continue
                    
                    if path.is_file():
                        # File check
                        if path.name in self.EXCLUDE_FILES:
                            continue
                        
                        # Extension check
                        if self._is_allowed_extension(path):
                            rel_path = path.relative_to(self.root_dir)
                            files.append(str(rel_path))
                
                return (
                    "\n".join(sorted(files))
                    if files else "🔍 Eşleşen dosya yok"
                )
            
            except Exception as e:
                logger.error(f"Listeleme hatası: {e}")
                self.metrics.errors_encountered += 1
                return f"❌ Listeleme hatası: {str(e)[:100]}"
    
    def read_file(self, filename: str) -> str:
        """
        Dosya oku (Tüm erişim seviyelerine açık)
        
        Args:
            filename: Dosya adı veya yolu
        
        Returns:
            Dosya içeriği
        """
        with self.lock:
            try:
                # Special handling for "this file" / "self"
                if any(k in filename.lower() for k in ["bu dosya", "kendini", "self"]):
                    target_path = Path(sys.argv[0]).resolve()
                else:
                    target_path = (self.root_dir / filename.strip()).resolve()
                
                # Security check
                if not self._is_safe_path(target_path):
                    if "bu dosya" not in filename.lower():
                        logger.warning(f"🚫 Sandbox dışı erişim: {target_path}")
                        return "[GÜVENLİK]: Proje dışı dosyalara erişim yasak"
                
                # Existence check
                if not target_path.exists():
                    return f"❌ Dosya bulunamadı: {filename}"
                
                # Directory check
                if target_path.is_dir():
                    return "❌ Bu bir dizin, list_files kullanın"
                
                # Read
                content = target_path.read_text(encoding="utf-8", errors="replace")
                
                self.metrics.files_read += 1
                return content
            
            except Exception as e:
                logger.error(f"Okuma hatası ({filename}): {e}")
                self.metrics.errors_encountered += 1
                return f"❌ Okuma hatası: {str(e)[:100]}"
    
    def save_file(self, filename: str, content: str) -> str:
        """
        Dosya kaydet (sandbox ve full modda)
        
        Args:
            filename: Dosya adı veya yolu
            content: İçerik
        
        Returns:
            Sonuç mesajı
        """
        # İzin kontrolü
        if not self._check_write_permission("Dosya kaydetme"):
            return "🚫 [GÜVENLİK]: Bu işlem için yetkiniz yok (sandbox veya full gerekli)"
        
        with self.lock:
            try:
                target_path = (self.root_dir / filename.strip()).resolve()
                
                # Security check
                if not self._is_safe_path(target_path):
                    return "[GÜVENLİK]: Sandbox dışına yazma yasak"
                
                # Create parent directories
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Backup if exists
                if target_path.exists():
                    self._backup_file(target_path)
                
                # Write
                target_path.write_text(content, encoding="utf-8")
                
                self.metrics.files_written += 1
                logger.info(f"💾 Dosya kaydedildi: {filename}")
                
                return f"✅ Kaydedildi: {filename}"
            
            except Exception as e:
                logger.error(f"Yazma hatası ({filename}): {e}")
                self.metrics.errors_encountered += 1
                return f"❌ Yazma hatası: {str(e)[:100]}"
    
    def delete_file(self, filename: str) -> str:
        """
        Dosya sil (sadece full modda)
        
        Args:
            filename: Dosya adı veya yolu
        
        Returns:
            Sonuç mesajı
        """
        # İzin kontrolü
        if not self._check_delete_permission("Dosya silme"):
            return "🚫 [GÜVENLİK]: Dosya silme için full erişim gerekli"
        
        with self.lock:
            try:
                target_path = (self.root_dir / filename.strip()).resolve()
                
                # Security check
                if not self._is_safe_path(target_path):
                    return "[GÜVENLİK]: Sandbox dışında silme yasak"
                
                # Existence check
                if not target_path.exists():
                    return f"❌ Dosya bulunamadı: {filename}"
                
                # Backup before delete
                self._backup_file(target_path, prefix="DELETED_")
                
                # Delete
                target_path.unlink()
                
                self.metrics.files_deleted += 1
                logger.warning(f"🗑️ Dosya silindi: {filename}")
                
                return f"✅ Silindi (yedek alındı): {filename}"
            
            except Exception as e:
                logger.error(f"Silme hatası ({filename}): {e}")
                self.metrics.errors_encountered += 1
                return f"❌ Silme hatası: {str(e)[:100]}"
    
    def _backup_file(self, path: Path, prefix: str = "") -> None:
        """Dosya yedekleme"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{prefix}{path.stem}_{timestamp}{path.suffix}.bak"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(path, backup_path)
            self.metrics.backups_created += 1
        
        except Exception as e:
            logger.error(f"Backup hatası: {e}")
    
    def get_file_info(self, filename: str) -> str:
        """
        Dosya bilgisi (Tüm erişim seviyelerine açık)
        
        Args:
            filename: Dosya adı
        
        Returns:
            Formatlanmış bilgi
        """
        try:
            target_path = (self.root_dir / filename.strip()).resolve()
            
            if not self._is_safe_path(target_path):
                return "❌ Erişim yasak"
            
            if not target_path.exists():
                return "❌ Dosya bulunamadı"
            
            # File stats
            stats = target_path.stat()
            size_kb = round(stats.st_size / 1024, 2)
            mod_time = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # Line count (for text files)
            lines = 0
            if target_path.suffix in {'.py', '.txt', '.md'}:
                try:
                    lines = len(target_path.read_text().splitlines())
                except Exception:
                    pass
            
            info_lines = [
                f"Dosya: {filename}",
                f"Boyut: {size_kb} KB",
                f"Son Değişiklik: {mod_time}",
                f"Tür: {target_path.suffix}"
            ]
            
            if lines > 0:
                info_lines.append(f"Satır Sayısı: {lines}")
            
            return "\n".join(info_lines)
        
        except Exception as e:
            return f"❌ Bilgi alma hatası: {str(e)[:100]}"
    
    # ───────────────────────────────────────────────────────────
    # CODE SEARCH
    # ───────────────────────────────────────────────────────────
    
    def search_code(
        self,
        query: str,
        is_regex: bool = False,
        file_ext: str = "*.py"
    ) -> str:
        """
        Kod arama (Tüm erişim seviyelerine açık)
        
        Args:
            query: Arama terimi
            is_regex: Regex kullan
            file_ext: Dosya uzantısı pattern
        
        Returns:
            Arama sonuçları
        """
        with self.lock:
            results = []
            
            try:
                for path in self.root_dir.rglob(file_ext):
                    # Exclude check
                    if any(part in self.EXCLUDE_DIRS for part in path.parts):
                        continue
                    
                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")
                        
                        # Search
                        match = False
                        if is_regex:
                            if re.search(query, content, re.IGNORECASE):
                                match = True
                        else:
                            if query.lower() in content.lower():
                                match = True
                        
                        if match:
                            rel_path = path.relative_to(self.root_dir)
                            results.append(str(rel_path))
                    
                    except Exception:
                        continue
                
                self.metrics.searches_performed += 1
                
                if results:
                    return (
                        f"🔍 '{query}' bulundu ({len(results)} dosya):\n" +
                        "\n".join(results)
                    )
                
                return "🔍 Eşleşen sonuç yok"
            
            except Exception as e:
                logger.error(f"Arama hatası: {e}")
                self.metrics.errors_encountered += 1
                return f"❌ Arama hatası: {str(e)[:100]}"
    
    # ───────────────────────────────────────────────────────────
    # TERMINAL OPERATIONS
    # ───────────────────────────────────────────────────────────
    
    def run_terminal(
        self,
        command: str,
        timeout: int = DEFAULT_TIMEOUT
    ) -> str:
        """
        Terminal komutu çalıştır (erişim seviyesine göre whitelist)
        
        Args:
            command: Komut
            timeout: Timeout (saniye)
        
        Returns:
            Komut çıktısı
        """
        with self.lock:
            try:
                # Parse command
                if os.name == 'nt':
                    cmd_parts = command.split()
                else:
                    cmd_parts = shlex.split(command)
                
                if not cmd_parts:
                    return "⚠️ Komut girmediniz"
                
                base_cmd = cmd_parts[0].lower()
                
                # Security: Illegal characters
                if any(char in command for char in self.ILLEGAL_CHARS):
                    return "🚫 [GÜVENLİK]: Zincirleme komutlar yasak"
                
                # Security: Erişim seviyesine göre whitelist
                allowed_commands = self._get_allowed_commands()
                if base_cmd not in allowed_commands:
                    return f"🚫 [GÜVENLİK]: '{base_cmd}' komutu {self.access_level} modunda yasak"
                
                # Restricted modda hiçbir komuta izin verilmez (whitelist zaten boş)
                if self.access_level == AccessLevel.RESTRICTED:
                    return "🚫 [GÜVENLİK]: Kısıtlı modda terminal komutu çalıştırılamaz"
                
                # Windows shell commands
                use_shell = (
                    os.name == 'nt' and
                    base_cmd in {'dir', 'echo', 'type', 'mkdir', 'date', 'tree'}
                )
                
                # Execute
                result = subprocess.run(
                    command if use_shell else cmd_parts,
                    capture_output=True,
                    text=True,
                    cwd=str(self.root_dir),
                    timeout=timeout,
                    shell=use_shell
                )
                
                # Output
                output = result.stdout
                if result.stderr:
                    output += f"\n[STDERR]: {result.stderr}"
                
                self.metrics.commands_executed += 1
                logger.info(f"💻 Terminal ({self.access_level}): {command[:50]}")
                
                return (
                    f"--- TERMİNAL ---\n{output.strip()}"
                    if output.strip() else "✅ İşlem tamamlandı"
                )
            
            except subprocess.TimeoutExpired:
                return f"⏱️ TIMEOUT: {timeout} saniye aşıldı"
            
            except Exception as e:
                logger.error(f"Terminal hatası: {e}")
                self.metrics.errors_encountered += 1
                return f"❌ Terminal hatası: {str(e)[:100]}"
    
    # ───────────────────────────────────────────────────────────
    # SYSTEM INFO
    # ───────────────────────────────────────────────────────────
    
    def get_gpu_info(self) -> str:
        """
        GPU bilgisi (Tüm erişim seviyelerine açık)
        
        Returns:
            GPU durumu
        """
        # Config check
        if not Config.USE_GPU:
            return "ℹ️ GPU: Config'de devre dışı (CPU modu)"
        
        try:
            # nvidia-smi
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                report = []
                
                for i, line in enumerate(lines):
                    data = [x.strip() for x in line.split(',')]
                    
                    if len(data) >= 4:
                        report.append(
                            f"🚀 GPU {i}: {data[0]}\n"
                            f"📊 Bellek: {data[2]}MB / {data[1]}MB\n"
                            f"🔥 Yük: %{data[3]}"
                        )
                
                return "\n".join(report) if report else "ℹ️ GPU verisi yok"
            
            return "ℹ️ GPU: nvidia-smi yanıt vermedi"
        
        except FileNotFoundError:
            return "ℹ️ GPU: nvidia-smi yüklü değil"
        
        except Exception as e:
            return f"ℹ️ GPU hatası: {str(e)[:50]}"
    
    def get_system_info(self) -> str:
        """
        Sistem bilgisi (Tüm erişim seviyelerine açık)
        
        Returns:
            Formatlanmış sistem bilgisi
        """
        gpu_status = self.get_gpu_info()
        
        return "\n".join([
            f"🖥️  Sistem: {sys.platform}",
            f"🐍 Python: {sys.version.split()[0]}",
            gpu_status,
            f"📁 Çalışma Dizini: {self.root_dir}",
            f"🛡️  Sandbox: AKTİF",
            f"🔐 Erişim Seviyesi: {self.access_level}"
        ])
    
    # ───────────────────────────────────────────────────────────
    # UTILITIES
    # ───────────────────────────────────────────────────────────
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Code manager metrikleri
        
        Returns:
            Metrik dictionary
        """
        return {
            "files_read": self.metrics.files_read,
            "files_written": self.metrics.files_written,
            "files_deleted": self.metrics.files_deleted,
            "backups_created": self.metrics.backups_created,
            "commands_executed": self.metrics.commands_executed,
            "searches_performed": self.metrics.searches_performed,
            "errors_encountered": self.metrics.errors_encountered,
            "sandbox_root": str(self.root_dir),
            "access_level": self.access_level
        }