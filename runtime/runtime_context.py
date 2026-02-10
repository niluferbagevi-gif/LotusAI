import queue
from concurrent.futures import ThreadPoolExecutor

from managers.messaging import MessagingManager


class RuntimeContext:
    """Tüm çalışma anı (runtime) durumunu merkezileştirir."""

    msg_queue = queue.Queue()
    messaging_manager = MessagingManager()
    engine = None
    loop = None
    security_instance = None
    state_manager = None

    # Web durumları
    active_web_agent = "ATLAS"
    voice_mode_active = False
    executor = ThreadPoolExecutor(max_workers=5)
