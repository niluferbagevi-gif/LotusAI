"""
LotusAI GitHub Manager
Sürüm: 2.6.0 (Dinamik Erişim Seviyesi Senkronu)
Açıklama: Bulut tabanlı GitHub depo yönetimi ve kod analizi

Özellikler:
- PyGithub ile resmi API entegrasyonu
- Depo (Repository) dosya ağacını çıkarma
- Dosya içeriği okuma (Cloud tabanlı)
- Son commit'leri (değişiklik geçmişini) izleme
- Erişim seviyesi kontrolleri
"""

import logging
import base64
from typing import Optional, List, Dict, Any

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
try:
    from config import Config, AccessLevel
except ImportError:
    class Config:
        ACCESS_LEVEL = "sandbox"
        GITHUB_TOKEN = None
        GITHUB_REPO = "niluferbagevi-gif/LotusAI"
    class AccessLevel:
        RESTRICTED = "restricted"
        SANDBOX = "sandbox"
        FULL = "full"

logger = logging.getLogger("LotusAI.Github")

# ═══════════════════════════════════════════════════════════════
# PYGITHUB LIBRARIES
# ═══════════════════════════════════════════════════════════════
GITHUB_AVAILABLE = False
try:
    from github import Github
    from github.GithubException import GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    logger.warning(
        "⚠️ PyGithub kütüphanesi eksik.\n"
        "Çalıştırmak için: pip install PyGithub"
    )

# ═══════════════════════════════════════════════════════════════
# GITHUB MANAGER
# ═══════════════════════════════════════════════════════════════
class GithubManager:
    """
    LotusAI GitHub Bağlantı ve Analiz Yöneticisi
    
    Yetenekler:
    - Belirtilen reponun (örn: niluferbagevi-gif/LotusAI) dosya yapısını listeler.
    - Repodaki herhangi bir dosyanın güncel kaynak kodunu okur.
    - Repoya atılan son commit'leri analiz edip "kim ne değiştirdi" bilgisini sunar.
    """
    
    def __init__(self, access_level: Optional[str] = None):
        """
        Github manager başlatıcı
        
        Args:
            access_level: Erişim seviyesi (restricted, sandbox, full)
        """
        self.access_level = access_level or getattr(Config, "ACCESS_LEVEL", "sandbox")
        
        # Config'den bilgileri al
        self.token = getattr(Config, "GITHUB_TOKEN", None)
        self.repo_name = getattr(Config, "GITHUB_REPO", "niluferbagevi-gif/LotusAI")
        
        self.g = None
        self.repo = None
        
        if GITHUB_AVAILABLE:
            if self.token:
                try:
                    self.g = Github(self.token)
                    self.repo = self.g.get_repo(self.repo_name)
                    logger.info(f"✅ GithubManager aktif. Bağlanılan Repo: {self.repo_name}")
                except GithubException as e:
                    logger.error(f"❌ GitHub API bağlantı hatası (Token geçersiz olabilir): {e.data}")
                except Exception as e:
                    logger.error(f"❌ GitHub genel hatası: {e}")
            else:
                logger.warning("⚠️ GITHUB_TOKEN bulunamadı. GithubManager salt okunur/anonim modda çalışabilir (API limitlerine takılabilir).")
                try:
                    # Token olmadan anonim bağlanmayı dene (saatte 60 istek limiti vardır)
                    self.g = Github()
                    self.repo = self.g.get_repo(self.repo_name)
                    logger.info(f"ℹ️ GithubManager anonim modda bağlandı: {self.repo_name}")
                except Exception as e:
                    logger.error(f"❌ Anonim GitHub bağlantısı başarısız: {e}")

    @property
    def is_active(self) -> bool:
        """Manager'ın repoya başarılı bir şekilde bağlanıp bağlanmadığını kontrol eder."""
        return self.repo is not None

    # ───────────────────────────────────────────────────────────
    # REPO OPERATIONS
    # ───────────────────────────────────────────────────────────
    
    def list_repo_files(self, path: str = "") -> str:
        """
        Repodaki dosyaları ve klasörleri listeler.
        
        Args:
            path: İncelenecek alt klasör (boş bırakılırsa ana dizin)
        Returns:
            Dosya ve klasörlerin listesi (string formatında)
        """
        if not self.is_active:
            return "❌ GitHub reposuna bağlantı kurulamadı."
            
        try:
            contents = self.repo.get_contents(path)
            # Eğer tek bir dosya ise listeye çeviriyoruz
            if not isinstance(contents, list):
                contents = [contents]
                
            files = []
            folders = []
            
            for content_file in contents:
                if content_file.type == "dir":
                    folders.append(f"📁 {content_file.path}/")
                else:
                    files.append(f"📄 {content_file.path}")
                    
            result = "\n".join(folders + files)
            return result if result else "Bu dizin boş."
            
        except GithubException as e:
            return f"❌ Dosyalar listelenirken hata oluştu: {e.data.get('message', str(e))}"

    def read_file_from_repo(self, file_path: str) -> str:
        """
        Repodaki bir dosyanın içeriğini okur.
        
        Args:
            file_path: Okunacak dosyanın tam yolu (örn: 'managers/finance.py')
        Returns:
            Dosyanın metin içeriği
        """
        if not self.is_active:
            return "❌ GitHub reposuna bağlantı kurulamadı."
            
        try:
            file_content = self.repo.get_contents(file_path)
            
            if isinstance(file_content, list):
                return f"❌ '{file_path}' bir dosya değil, bir klasör. Lütfen list_repo_files metodunu kullanın."
                
            # Dosya içeriği base64 olarak gelir, decode ediyoruz
            decoded_content = base64.b64decode(file_content.content).decode("utf-8")
            return decoded_content
            
        except GithubException as e:
            if e.status == 404:
                return f"❌ Dosya bulunamadı: {file_path}"
            return f"❌ Dosya okunurken hata oluştu: {e.data.get('message', str(e))}"
            
    def get_recent_commits(self, limit: int = 5) -> str:
        """
        Repodaki en son yapılan değişiklikleri (commitleri) listeler.
        
        Args:
            limit: Kaç commit getirileceği
        Returns:
            Commit listesi (string formatında)
        """
        if not self.is_active:
            return "❌ GitHub reposuna bağlantı kurulamadı."
            
        try:
            commits = self.repo.get_commits()[:limit]
            result = [f"📊 Son {limit} Değişiklik (Commit):"]
            
            for commit in commits:
                date_str = commit.commit.author.date.strftime("%Y-%m-%d %H:%M")
                author = commit.commit.author.name
                msg = commit.commit.message.split('\n')[0] # Sadece başlığı al
                sha = commit.sha[:7] # Kısa hash
                
                result.append(f"- [{date_str}] {author} ({sha}): {msg}")
                
            return "\n".join(result)
            
        except Exception as e:
            return f"❌ Commitler çekilirken hata oluştu: {str(e)}"