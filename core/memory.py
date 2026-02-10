import sqlite3
import logging
import threading
import hashlib
import re
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# --- YAPILANDIRMA VE FALLBACK ---
try:
    from config import Config
except ImportError:
    # Bağımsız çalışma durumu için Fallback
    class Config:
        WORK_DIR = Path.cwd()
        USE_GPU = False

# --- LOGLAMA ---
logger = logging.getLogger("LotusAI.Memory")

# --- KÜTÜPHANE KONTROLLERİ ---

# PDF İşleme
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("⚠️ PyPDF2 bulunamadı. PDF okuma devre dışı. (pip install PyPDF2)")

# Word İşleme
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("⚠️ python-docx bulunamadı. Word okuma devre dışı. (pip install python-docx)")

# Vektör Veritabanı (ChromaDB) ve GPU Desteği
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    import torch
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.error("⚠️ ChromaDB veya bağımlılıkları bulunamadı. Uzun süreli hafıza ve RAG devre dışı. (pip install chromadb sentence-transformers torch)")

class MemoryManager:
    """
    LotusAI Gelişmiş Hafıza ve RAG (Retrieval-Augmented Generation) Sistemi.
    
    Yetenekler:
    - Kısa Süreli Hafıza: SQLite (Hızlı erişim, sıralı sohbet akışı).
    - Uzun Süreli Hafıza: ChromaDB (Anlamsal arama, geçmiş hatırlatma).
    - GPU Hızlandırma: Config ve Donanım uyumlu embedding işlemleri.
    - Bilgi Bankası (RAG): PDF, DOCX, TXT, MD belgelerinin vektörel indekslenmesi.
    """
    
    def __init__(self):
        # Yollar (Config üzerinden dinamik olarak alınır)
        self.work_dir = Path(getattr(Config, "WORK_DIR", "./data"))
        self.db_path = self.work_dir / "lotus_system.db"
        self.docs_path = self.work_dir / "documents"
        self.vector_db_path = self.work_dir / "lotus_vector_db"
        
        self.lock = threading.Lock()
        
        # Gerekli klasörlerin varlığından emin ol
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)

        self._init_sqlite()
        
        self.chroma_client = None
        self.history_collection = None
        self.docs_collection = None
        self.embedding_fn = None
        
        # Donanım Durumu (Merkezi Config + Kütüphane Kontrolü)
        config_gpu = getattr(Config, "USE_GPU", False)
        
        if CHROMA_AVAILABLE:
            try:
                # 1. GPU/CPU Kontrolü ve Cihaz Seçimi
                # Config GPU'ya izin veriyorsa VE Torch CUDA görüyorsa
                if config_gpu and torch.cuda.is_available():
                    device = "cuda"
                    logger.info(f"🚀 GPU Algılandı: {torch.cuda.get_device_name(0)}. Hafıza işlemleri GPU üzerinde çalışacak.")
                else:
                    device = "cpu"
                    if config_gpu:
                        logger.warning("⚠️ Config GPU açık dedi ancak Torch/CUDA bulunamadı. CPU'ya geçiliyor.")
                    else:
                        logger.info("ℹ️ Hafıza işlemleri CPU modunda (Config veya donanım kısıtlaması).")

                # 2. Embedding Fonksiyonunu Hazırla
                # 'all-MiniLM-L6-v2' hızlıdır, daha kaliteli sonuçlar için 'paraphrase-multilingual-MiniLM-L12-v2' tercih edilebilir.
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2",
                    device=device
                )

                # 3. Kalıcı ChromaDB istemcisi
                self.chroma_client = chromadb.PersistentClient(path=str(self.vector_db_path))
                
                # 4. Koleksiyonları oluştur veya mevcut olanları yükle
                self.history_collection = self.chroma_client.get_or_create_collection(
                    name="chat_history", 
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self.embedding_fn
                )
                self.docs_collection = self.chroma_client.get_or_create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=self.embedding_fn
                )
                
                doc_count = self.docs_collection.count()
                logger.info(f"✅ Hafıza Sistemi Aktif: {doc_count} belge parçası yüklü.")
                
            except Exception as e:
                logger.error(f"❌ ChromaDB Başlatma Hatası: {e}")
                self.chroma_client = None

    def _init_sqlite(self):
        """SQLite veritabanı şemasını hazırlar ve tabloları oluşturur."""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=10)
                c = conn.cursor()
                # Sohbet Geçmişi Tablosu
                c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                             (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                              agent TEXT, 
                              role TEXT, 
                              message TEXT, 
                              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
                
                # İşlenen Dosyalar Tablosu (Tekrar işlemeyi önlemek için)
                c.execute('''CREATE TABLE IF NOT EXISTS processed_files 
                             (filename TEXT PRIMARY KEY, 
                              processed_date DATETIME, 
                              file_hash TEXT,
                              file_size INTEGER)''')
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                logger.error(f"❌ SQLite Başlatma Hatası: {e}")

    # --- 1. BELGE İŞLEME (RAG INGESTION) ---

    def ingest_documents(self) -> str:
        """Belgeleri tarar ve vektör veritabanına indeksler."""
        if not CHROMA_AVAILABLE or not self.docs_collection:
            return "Vektör veritabanı (ChromaDB) yüklü veya aktif değil."

        new_files_count = 0
        supported_extensions = ['*.txt', '*.pdf', '*.md', '*.docx']
        
        for pattern in supported_extensions:
            for file_path in self.docs_path.glob(pattern):
                filename = file_path.name
                
                try:
                    file_content_raw = file_path.read_bytes()
                    file_hash = hashlib.md5(file_content_raw).hexdigest()
                    file_size = file_path.stat().st_size
                    
                    if self._is_file_processed(filename, file_hash):
                        continue

                    content = ""
                    if file_path.suffix in ['.txt', '.md']:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    elif file_path.suffix == '.pdf' and PDF_AVAILABLE:
                        with open(file_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page in reader.pages:
                                text = page.extract_text()
                                if text: content += text + "\n"
                                
                    elif file_path.suffix == '.docx' and DOCX_AVAILABLE:
                        doc = docx.Document(file_path)
                        content = "\n".join([para.text for para in doc.paragraphs])

                    if content.strip():
                        chunks = self._smart_chunk_text(content, chunk_size=800, overlap=150)
                        if not chunks: continue

                        ids = [f"{filename}_{i}_{file_hash[:8]}" for i in range(len(chunks))]
                        metadatas = [
                            {
                                "source": filename, 
                                "chunk_id": i, 
                                "date": datetime.now().isoformat(),
                                "type": file_path.suffix[1:],
                                "hash": file_hash
                            } 
                            for i in range(len(chunks))
                        ]
                        
                        # Vektör veritabanına toplu ekleme
                        self.docs_collection.add(
                            documents=chunks, 
                            metadatas=metadatas, 
                            ids=ids
                        )
                        
                        self._mark_file_as_processed(filename, file_hash, file_size)
                        new_files_count += 1
                        logger.info(f"📄 Bilgi Bankasına Eklendi: {filename} ({len(chunks)} parça)")

                except Exception as e:
                    logger.error(f"❌ Dosya işleme hatası ({filename}): {e}")

        if new_files_count > 0:
            return f"Başarılı: {new_files_count} yeni belge bilgi bankasına eklendi."
        return "Bilgi bankası güncel, yeni dosya bulunamadı."

    def _smart_chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Metni anlamsal bütünlüğü bozmadan parçalara ayırır."""
        if not text: return []
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                # Cümle sonunu bulmaya çalış (.!?)
                last_punctuation = -1
                for punct in ".!?":
                    pos = text.rfind(punct, start + (chunk_size // 2), end)
                    if pos > last_punctuation:
                        last_punctuation = pos
                if last_punctuation != -1:
                    end = last_punctuation + 1
            
            chunk = text[start:end].strip()
            if len(chunk) > 10:
                chunks.append(chunk)
            
            # Bir sonraki parça için başlangıç noktasını ayarla (overlap kadar geri git)
            start = end - overlap
            if start < 0: start = 0
            if end >= text_len: break
            
        return chunks

    def _is_file_processed(self, filename: str, file_hash: str) -> bool:
        """Dosyanın daha önce işlenip işlenmediğini kontrol eder."""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                c = conn.cursor()
                c.execute("SELECT 1 FROM processed_files WHERE filename=? AND file_hash=?", (filename, file_hash))
                res = c.fetchone()
                conn.close()
                return res is not None
            except: return False

    def _mark_file_as_processed(self, filename: str, file_hash: str, file_size: int):
        """Dosyayı işlenmiş olarak işaretler."""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                c = conn.cursor()
                c.execute("""INSERT OR REPLACE INTO processed_files 
                             (filename, processed_date, file_hash, file_size) 
                             VALUES (?, ?, ?, ?)""", 
                          (filename, datetime.now().isoformat(), file_hash, file_size))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"❌ Veritabanı işaretleme hatası: {e}")

    # --- 2. SOHBET KAYIT VE ERİŞİM ---

    def save(self, agent: str, role: str, message: str):
        """Mesajı kaydeder. ChromaDB kısmı GPU ile vektörlenir (Varsa)."""
        if not message or not message.strip(): return
        timestamp = datetime.now()
        
        # 1. SQLite Kaydı
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=10)
                c = conn.cursor()
                c.execute("INSERT INTO chat_history (agent, role, message, timestamp) VALUES (?, ?, ?, ?)", 
                          (agent, role, message, timestamp.isoformat()))
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                logger.error(f"❌ Hafıza Kayıt Hatası: {e}")
            
        # 2. ChromaDB Vektör Kaydı
        if CHROMA_AVAILABLE and self.history_collection and len(message) > 15:
            try:
                meta = {"agent": str(agent), "role": str(role), "time": str(timestamp.isoformat())}
                # Benzersiz ID oluştur
                unique_id = f"{agent}_{timestamp.timestamp()}_{hashlib.md5(message.encode()).hexdigest()[:6]}"
                
                self.history_collection.add(
                    documents=[message], 
                    metadatas=[meta], 
                    ids=[unique_id]
                )
            except Exception as e:
                logger.debug(f"⚠️ Vektör geçmiş kaydı başarısız: {e}")

    def load_context(self, agent: str, query: str, limit: int = 10) -> Tuple[List[Dict], str, str]:
        """Ajan için GPU destekli anlamsal arama ile bağlam hazırlar."""
        recent = []
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=10)
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("""SELECT role, message FROM chat_history 
                             WHERE agent=? OR agent='SYSTEM' 
                             ORDER BY id DESC LIMIT ?""", (agent, limit))
                rows = c.fetchall()
                conn.close()
                recent = [{"role": r["role"], "content": r["message"]} for r in reversed(rows)]
            except: pass
        
        long_term_history = ""
        relevant_docs = ""
        
        if CHROMA_AVAILABLE:
            # 1. Benzer eski konuşmalar
            if self.history_collection:
                try:
                    res = self.history_collection.query(
                        query_texts=[query], 
                        n_results=3, 
                        where={"agent": agent}
                    )
                    if res['documents'] and res['documents'][0]:
                        long_term_history = "\n".join([f"• {d}" for d in res['documents'][0]])
                except: pass

            # 2. Bilgi Bankasında Ara
            if self.docs_collection:
                try:
                    res_docs = self.docs_collection.query(query_texts=[query], n_results=4)
                    if res_docs['documents'] and res_docs['documents'][0]:
                        docs_content = []
                        for i, doc_text in enumerate(res_docs['documents'][0]):
                            src = res_docs['metadatas'][0][i].get('source', 'Bilinmiyor')
                            docs_content.append(f"[{src}]: {doc_text}")
                        relevant_docs = "\n\n".join(docs_content)
                except Exception as e:
                    logger.error(f"❌ RAG Sorgu Hatası: {e}")
            
        return recent, long_term_history, relevant_docs

    def get_agent_history_for_web(self, agent_name: str, limit: int = 40) -> List[Dict]:
        """Web arayüzü için sohbet geçmişini formatlar."""
        target_agent = "ATLAS" if agent_name == "GENEL" else agent_name
        rows = []
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("""SELECT agent, role, message, timestamp FROM chat_history 
                             WHERE agent=? ORDER BY id DESC LIMIT ?""", 
                          (target_agent, limit))
                rows = c.fetchall()
                conn.close()
            except: return []

        return [{
            "sender": "Siz" if r["role"] == "user" else r["agent"],
            "text": r["message"],
            "type": "user" if r["role"] == "user" else "agent",
            "time": r["timestamp"][11:16] if r["timestamp"] else ""
        } for r in reversed(rows)]

    def clear_history(self, only_chat: bool = True) -> bool:
        """Hafızayı temizler. only_chat=False ise bilgi bankası da silinir."""
        with self.lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                c = conn.cursor()
                c.execute("DELETE FROM chat_history")
                if not only_chat:
                    c.execute("DELETE FROM processed_files")
                conn.commit()
                conn.close()
                
                if CHROMA_AVAILABLE and self.chroma_client:
                    try:
                        self.chroma_client.delete_collection("chat_history")
                        self.history_collection = self.chroma_client.get_or_create_collection(
                            name="chat_history", 
                            embedding_function=self.embedding_fn
                        )
                        
                        if not only_chat:
                            self.chroma_client.delete_collection("documents")
                            self.docs_collection = self.chroma_client.get_or_create_collection(
                                name="documents", 
                                embedding_function=self.embedding_fn
                            )
                    except Exception as ve:
                        logger.warning(f"⚠️ Vektör temizleme uyarısı: {ve}")
                
                logger.info(f"🧹 Hafıza temizlendi (Mod: {'Sohbet' if only_chat else 'Tam'})")
                return True
            except Exception as e:
                logger.error(f"❌ Hafıza temizleme hatası: {e}")
                return False

# import sqlite3
# import logging
# import threading
# import hashlib
# import re
# import os
# from pathlib import Path
# from datetime import datetime
# from typing import List, Dict, Tuple, Any, Optional
# from config import Config

# # --- LOGLAMA ---
# logger = logging.getLogger("LotusAI.Memory")

# # --- KÜTÜPHANE KONTROLLERİ ---
# # PDF İşleme
# try:
#     import PyPDF2
#     PDF_AVAILABLE = True
# except ImportError:
#     PDF_AVAILABLE = False
#     logger.warning("⚠️ PyPDF2 bulunamadı. PDF okuma devre dışı. (pip install PyPDF2)")

# # Word İşleme
# try:
#     import docx
#     DOCX_AVAILABLE = True
# except ImportError:
#     DOCX_AVAILABLE = False
#     logger.warning("⚠️ python-docx bulunamadı. Word okuma devre dışı. (pip install python-docx)")

# # Vektör Veritabanı (ChromaDB)
# try:
#     import chromadb
#     from chromadb.config import Settings
#     CHROMA_AVAILABLE = True
# except ImportError:
#     CHROMA_AVAILABLE = False
#     logger.error("⚠️ ChromaDB bulunamadı. Uzun süreli hafıza ve RAG devre dışı. (pip install chromadb)")

# class MemoryManager:
#     """
#     LotusAI Gelişmiş Hafıza ve RAG (Retrieval-Augmented Generation) Sistemi.
    
#     Yetenekler:
#     - Kısa Süreli Hafıza: SQLite (Hızlı erişim, sıralı sohbet akışı).
#     - Uzun Süreli Hafıza: ChromaDB (Anlamsal arama, geçmiş hatırlatma).
#     - Bilgi Bankası (RAG): PDF, DOCX, TXT, MD belgelerinin vektörel indekslenmesi.
#     - Akıllı Metin Parçalama: Anlamsal bütünlüğü koruyan chunking algoritması.
#     - Bağlam Yönetimi: LLM limitlerine uygun context hazırlama.
#     """
    
#     def __init__(self):
#         # Yollar (Config üzerinden dinamik olarak alınır)
#         self.work_dir = Path(getattr(Config, "WORK_DIR", "./data"))
#         self.db_path = self.work_dir / "lotus_system.db"
#         self.docs_path = self.work_dir / "documents"
#         self.vector_db_path = self.work_dir / "lotus_vector_db"
        
#         self.lock = threading.Lock()
        
#         # Gerekli klasörlerin varlığından emin ol
#         self.docs_path.mkdir(parents=True, exist_ok=True)
#         self.vector_db_path.mkdir(parents=True, exist_ok=True)

#         self._init_sqlite()
        
#         self.chroma_client = None
#         self.history_collection = None
#         self.docs_collection = None
        
#         if CHROMA_AVAILABLE:
#             try:
#                 # Kalıcı ChromaDB istemcisi
#                 self.chroma_client = chromadb.PersistentClient(path=str(self.vector_db_path))
                
#                 # Koleksiyonları oluştur veya mevcut olanları yükle
#                 self.history_collection = self.chroma_client.get_or_create_collection(
#                     name="chat_history", 
#                     metadata={"hnsw:space": "cosine"} 
#                 )
#                 self.docs_collection = self.chroma_client.get_or_create_collection(
#                     name="documents",
#                     metadata={"hnsw:space": "cosine"}
#                 )
                
#                 logger.info(f"✅ Hafıza Sistemi Aktif: {self.docs_collection.count()} belge parçası yüklü.")
#             except Exception as e:
#                 logger.error(f"❌ ChromaDB Başlatma Hatası: {e}")

#     def _init_sqlite(self):
#         """SQLite veritabanı şemasını hazırlar ve tabloları oluşturur."""
#         with self.lock:
#             try:
#                 conn = sqlite3.connect(str(self.db_path), timeout=10)
#                 c = conn.cursor()
#                 # Sohbet geçmişi tablosu (Hızlı erişim ve UI için)
#                 c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
#                              (id INTEGER PRIMARY KEY AUTOINCREMENT, 
#                               agent TEXT, 
#                               role TEXT, 
#                               message TEXT, 
#                               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
                
#                 # İşlenen dosyaların takibi (Mükerrer işlemeyi önlemek için)
#                 c.execute('''CREATE TABLE IF NOT EXISTS processed_files 
#                              (filename TEXT PRIMARY KEY, 
#                               processed_date DATETIME, 
#                               file_hash TEXT,
#                               file_size INTEGER)''')
#                 conn.commit()
#                 conn.close()
#             except sqlite3.Error as e:
#                 logger.error(f"❌ SQLite Başlatma Hatası: {e}")

#     # --- 1. BELGE İŞLEME (RAG INGESTION) ---

#     def ingest_documents(self) -> str:
#         """
#         'documents/' klasöründeki yeni veya güncellenmiş belgeleri tarar, 
#         içeriklerini çıkarır ve vektör veritabanına indeksler.
#         """
#         if not CHROMA_AVAILABLE or not self.docs_collection:
#             return "Vektör veritabanı (ChromaDB) yüklü veya aktif değil."

#         new_files_count = 0
#         supported_extensions = ['*.txt', '*.pdf', '*.md', '*.docx']
        
#         for pattern in supported_extensions:
#             for file_path in self.docs_path.glob(pattern):
#                 filename = file_path.name
                
#                 try:
#                     # Dosya değişikliğini kontrol et (Hash ve Boyut ile)
#                     file_content_raw = file_path.read_bytes()
#                     file_hash = hashlib.md5(file_content_raw).hexdigest()
#                     file_size = file_path.stat().st_size
                    
#                     if self._is_file_processed(filename, file_hash):
#                         continue

#                     content = ""
#                     # Dosya tipine göre metin çıkarma
#                     if file_path.suffix in ['.txt', '.md']:
#                         content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
#                     elif file_path.suffix == '.pdf' and PDF_AVAILABLE:
#                         with open(file_path, 'rb') as f:
#                             reader = PyPDF2.PdfReader(f)
#                             for page in reader.pages:
#                                 text = page.extract_text()
#                                 if text: content += text + "\n"
                                
#                     elif file_path.suffix == '.docx' and DOCX_AVAILABLE:
#                         doc = docx.Document(file_path)
#                         content = "\n".join([para.text for para in doc.paragraphs])

#                     if content.strip():
#                         # Metni yönetilebilir parçalara ayır
#                         chunks = self._smart_chunk_text(content, chunk_size=800, overlap=150)
#                         if not chunks: continue

#                         ids = [f"{filename}_{i}_{file_hash[:8]}" for i in range(len(chunks))]
#                         metadatas = [
#                             {
#                                 "source": filename, 
#                                 "chunk_id": i, 
#                                 "date": datetime.now().isoformat(),
#                                 "type": file_path.suffix[1:],
#                                 "hash": file_hash
#                             } 
#                             for i in range(len(chunks))
#                         ]
                        
#                         # Vektör veritabanına toplu ekleme
#                         self.docs_collection.add(
#                             documents=chunks, 
#                             metadatas=metadatas, 
#                             ids=ids
#                         )
                        
#                         self._mark_file_as_processed(filename, file_hash, file_size)
#                         new_files_count += 1
#                         logger.info(f"📄 Bilgi Bankasına Eklendi: {filename} ({len(chunks)} parça)")

#                 except Exception as e:
#                     logger.error(f"❌ Dosya işleme hatası ({filename}): {e}")

#         if new_files_count > 0:
#             return f"Başarılı: {new_files_count} yeni belge bilgi bankasına eklendi."
#         return "Bilgi bankası güncel, yeni dosya bulunamadı."

#     def _smart_chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
#         """
#         Metni anlamsal bütünlüğü bozmadan (cümleleri bölmeden) parçalara ayırır.
#         """
#         if not text: return []
        
#         # Fazla boşlukları ve satır sonlarını temizle
#         text = re.sub(r'\s+', ' ', text).strip()
        
#         chunks = []
#         start = 0
#         text_len = len(text)
        
#         while start < text_len:
#             end = start + chunk_size
            
#             if end < text_len:
#                 # Cümle sonu karakterlerini ara (son 150 karakter içinde)
#                 # Bu, bir düşüncenin ortadan bölünmesini engeller.
#                 last_punctuation = -1
#                 for punct in ".!?":
#                     pos = text.rfind(punct, start + (chunk_size // 2), end)
#                     if pos > last_punctuation:
#                         last_punctuation = pos
                
#                 if last_punctuation != -1:
#                     end = last_punctuation + 1
            
#             chunk = text[start:end].strip()
#             if len(chunk) > 10: # Çok kısa anlamsız parçaları atla
#                 chunks.append(chunk)
            
#             # Overlap ile bir sonraki parça için geri git (bağlam sürekliliği)
#             start = end - overlap
#             if start < 0: start = 0
#             if end >= text_len: break
            
#         return chunks

#     def _is_file_processed(self, filename: str, file_hash: str) -> bool:
#         """Dosyanın veritabanında aynı hash ile kayıtlı olup olmadığını kontrol eder."""
#         with self.lock:
#             try:
#                 conn = sqlite3.connect(str(self.db_path))
#                 c = conn.cursor()
#                 c.execute("SELECT 1 FROM processed_files WHERE filename=? AND file_hash=?", (filename, file_hash))
#                 res = c.fetchone()
#                 conn.close()
#                 return res is not None
#             except: return False

#     def _mark_file_as_processed(self, filename: str, file_hash: str, file_size: int):
#         """İşlenen dosyayı SQLite tablosuna kaydeder."""
#         with self.lock:
#             try:
#                 conn = sqlite3.connect(str(self.db_path))
#                 c = conn.cursor()
#                 c.execute("""INSERT OR REPLACE INTO processed_files 
#                              (filename, processed_date, file_hash, file_size) 
#                              VALUES (?, ?, ?, ?)""", 
#                           (filename, datetime.now().isoformat(), file_hash, file_size))
#                 conn.commit()
#                 conn.close()
#             except Exception as e:
#                 logger.error(f"❌ Veritabanı işaretleme hatası: {e}")

#     # --- 2. SOHBET KAYIT VE ERİŞİM ---

#     def save(self, agent: str, role: str, message: str):
#         """
#         Mesajı hem SQLite'a (Sıralı hafıza) hem ChromaDB'ye (Anlamsal hafıza) kaydeder.
#         """
#         if not message or not message.strip(): return
        
#         timestamp = datetime.now()
        
#         # 1. SQLite Kaydı
#         with self.lock:
#             try:
#                 conn = sqlite3.connect(str(self.db_path), timeout=10)
#                 c = conn.cursor()
#                 c.execute("INSERT INTO chat_history (agent, role, message, timestamp) VALUES (?, ?, ?, ?)", 
#                           (agent, role, message, timestamp.isoformat()))
#                 conn.commit()
#                 conn.close()
#             except sqlite3.Error as e:
#                 logger.error(f"❌ Hafıza Kayıt Hatası: {e}")
            
#         # 2. ChromaDB Vektör Kaydı (Sadece anlamlı uzunluktaki mesajlar için)
#         if CHROMA_AVAILABLE and self.history_collection and len(message) > 15:
#             try:
#                 meta = {"agent": str(agent), "role": str(role), "time": str(timestamp.isoformat())}
#                 unique_id = f"{agent}_{timestamp.timestamp()}_{hashlib.md5(message.encode()).hexdigest()[:6]}"
                
#                 self.history_collection.add(
#                     documents=[message], 
#                     metadatas=[meta], 
#                     ids=[unique_id]
#                 )
#             except Exception as e:
#                 logger.debug(f"⚠️ Vektör geçmiş kaydı başarısız: {e}")

#     def load_context(self, agent: str, query: str, limit: int = 10) -> Tuple[List[Dict], str, str]:
#         """
#         Ajan için 3 katmanlı bağlam hazırlar:
#         1. Yakın geçmiş (Kısa süreli)
#         2. Benzer eski konuşmalar (Anlamsal uzun süreli)
#         3. Bilgi bankası belgeleri (RAG)
#         """
#         # A. Kısa Süreli Hafıza
#         recent = []
#         with self.lock:
#             try:
#                 conn = sqlite3.connect(str(self.db_path), timeout=10)
#                 conn.row_factory = sqlite3.Row
#                 c = conn.cursor()
#                 # Ajanın kendi geçmişini ve genel sistem mesajlarını getir
#                 c.execute("""SELECT role, message FROM chat_history 
#                              WHERE agent=? OR agent='SYSTEM' 
#                              ORDER BY id DESC LIMIT ?""", (agent, limit))
#                 rows = c.fetchall()
#                 conn.close()
#                 recent = [{"role": r["role"], "content": r["message"]} for r in reversed(rows)]
#             except: pass
        
#         # B. Vektörel Hafıza ve RAG
#         long_term_history = ""
#         relevant_docs = ""
        
#         if CHROMA_AVAILABLE:
#             # 1. Benzer eski konuşmaları bul
#             if self.history_collection:
#                 try:
#                     res = self.history_collection.query(
#                         query_texts=[query], 
#                         n_results=3, 
#                         where={"agent": agent}
#                     )
#                     if res['documents'] and res['documents'][0]:
#                         long_term_history = "\n".join([f"• {d}" for d in res['documents'][0]])
#                 except: pass

#             # 2. Bilgi Bankasında (RAG) ara
#             if self.docs_collection:
#                 try:
#                     res_docs = self.docs_collection.query(query_texts=[query], n_results=4)
#                     if res_docs['documents'] and res_docs['documents'][0]:
#                         docs_content = []
#                         for i, doc_text in enumerate(res_docs['documents'][0]):
#                             src = res_docs['metadatas'][0][i].get('source', 'Bilinmiyor')
#                             docs_content.append(f"[{src}]: {doc_text}")
#                         relevant_docs = "\n\n".join(docs_content)
#                 except Exception as e:
#                     logger.error(f"❌ RAG Sorgu Hatası: {e}")
            
#         return recent, long_term_history, relevant_docs

#     # --- 3. YÖNETİM VE WEB YARDIMCILARI ---

#     def get_agent_history_for_web(self, agent_name: str, limit: int = 40) -> List[Dict]:
#         """Web arayüzünde mesajları görüntülemek için temiz veri seti döner."""
#         target_agent = "ATLAS" if agent_name == "GENEL" else agent_name
        
#         rows = []
#         with self.lock:
#             try:
#                 conn = sqlite3.connect(str(self.db_path))
#                 conn.row_factory = sqlite3.Row
#                 c = conn.cursor()
#                 c.execute("""SELECT agent, role, message, timestamp FROM chat_history 
#                              WHERE agent=? ORDER BY id DESC LIMIT ?""", 
#                           (target_agent, limit))
#                 rows = c.fetchall()
#                 conn.close()
#             except: return []

#         return [{
#             "sender": "Siz" if r["role"] == "user" else r["agent"],
#             "text": r["message"],
#             "type": "user" if r["role"] == "user" else "agent",
#             "time": r["timestamp"][11:16] if r["timestamp"] else ""
#         } for r in reversed(rows)]

#     def clear_history(self, only_chat: bool = True) -> bool:
#         """
#         Sistemi sıfırlar. 
#         only_chat=True ise sadece yazışmaları, False ise dökümanları da siler.
#         """
#         with self.lock:
#             try:
#                 conn = sqlite3.connect(str(self.db_path))
#                 c = conn.cursor()
#                 c.execute("DELETE FROM chat_history")
#                 if not only_chat:
#                     c.execute("DELETE FROM processed_files")
#                     # Döküman klasöründeki dosyalar fiziksel olarak silinmez, sadece indeksi temizlenir.
#                 conn.commit()
#                 conn.close()
                
#                 if CHROMA_AVAILABLE:
#                     self.chroma_client.delete_collection("chat_history")
#                     self.history_collection = self.chroma_client.get_or_create_collection(name="chat_history")
                    
#                     if not only_chat:
#                         self.chroma_client.delete_collection("documents")
#                         self.docs_collection = self.chroma_client.get_or_create_collection(name="documents")
                
#                 logger.info(f"🧹 Hafıza temizlendi (Mod: {'Sohbet' if only_chat else 'Tam'})")
#                 return True
#             except Exception as e:
#                 logger.error(f"❌ Hafıza temizleme hatası: {e}")
#                 return False