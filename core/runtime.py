# core/runtime.py
import queue
from concurrent.futures import ThreadPoolExecutor

class RuntimeContext:
    """Tüm global değişkenlerin merkezi yönetimi."""
    msg_queue = queue.Queue()
    messaging_manager = None # Main loop başladığında atanacak
    engine = None 
    loop = None
    security_instance = None 
    state_manager = None
    
    # Web Durumları
    active_web_agent = "ATLAS"
    voice_mode_active = False
    executor = ThreadPoolExecutor(max_workers=5)