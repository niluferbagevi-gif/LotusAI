from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

from sidar_clone_starter.config import Settings
from sidar_clone_starter.core.memory import ConversationMemory
from sidar_clone_starter.managers.code_manager import CodeManager
from sidar_clone_starter.managers.system_health import SystemHealthManager


class ErrorType(str, Enum):
    GPU_OOM = "gpu_oom"
    IMPORT_ERROR = "import_error"
    FILE_NOT_FOUND = "file_not_found"
    SYNTAX_ERROR = "syntax_error"
    UNKNOWN = "unknown"


@dataclass
class EngineerMetrics:
    files_read: int = 0
    files_written: int = 0
    syntax_checks: int = 0
    syntax_errors: int = 0


class EngineerAgent:
    ERROR_PATTERNS = {
        "CUDA out of memory": ErrorType.GPU_OOM,
        "ImportError": ErrorType.IMPORT_ERROR,
        "ModuleNotFoundError": ErrorType.IMPORT_ERROR,
        "FileNotFoundError": ErrorType.FILE_NOT_FOUND,
        "SyntaxError": ErrorType.SYNTAX_ERROR,
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.code = CodeManager(settings)
        self.health = SystemHealthManager()
        self.memory = ConversationMemory()
        self.metrics = EngineerMetrics()

    def analyze_error(self, traceback_text: str) -> Dict[str, Any]:
        for pattern, error_type in self.ERROR_PATTERNS.items():
            if pattern in traceback_text:
                return {"error_type": error_type.value, "matched_pattern": pattern}
        return {"error_type": ErrorType.UNKNOWN.value, "matched_pattern": None}

    def run_system_audit(self) -> Dict[str, Any]:
        return self.health.collect()

    def handle(self, command: str) -> Dict[str, Any]:
        text = command.lower()

        if "audit" in text or "denet" in text:
            return {"ok": True, "action": "audit", "data": self.run_system_audit()}

        if text.startswith("read "):
            target = command[5:].strip()
            content = self.code.read_file(target)
            self.memory.last_file = target
            self.metrics.files_read += 1
            return {"ok": True, "action": "read", "file": target, "content": content}

        if text.startswith("write "):
            parts = command.split("|", 1)
            if len(parts) != 2:
                return {"ok": False, "error": "Use: write <path> | <content>"}

            left, content = parts
            file_path = left.replace("write", "", 1).strip()
            result = self.code.write_file(file_path, content)
            if result.get("ok"):
                self.metrics.files_written += 1
            return {"ok": result.get("ok", False), "action": "write", "result": result}

        return {"ok": False, "error": "Unknown command. Use read/write/audit."}
