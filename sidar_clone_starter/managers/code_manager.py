import ast
import json
from pathlib import Path
from typing import Dict, Any

from sidar_clone_starter.config import Settings
from sidar_clone_starter.core.access import can_write


class CodeManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def read_file(self, file_path: str) -> str:
        path = Path(file_path)
        return path.read_text(encoding="utf-8")

    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not can_write(self.settings.access_level, path, self.settings.sandbox_dir):
            return {"ok": False, "error": "Write permission denied by access level."}

        path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(path)}

    def check_python_syntax(self, source_code: str) -> Dict[str, Any]:
        try:
            ast.parse(source_code)
            return {"ok": True}
        except SyntaxError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "line": exc.lineno,
                "offset": exc.offset,
            }

    def check_json_syntax(self, raw_json: str) -> Dict[str, Any]:
        try:
            json.loads(raw_json)
            return {"ok": True}
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "line": exc.lineno,
                "col": exc.colno,
            }
