"""
Sidar Project - Birim Testleri
pytest ile çalıştır: cd sidar_project && pytest tests/ -v
"""

import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# Proje kökünü yola ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from managers.security import SecurityManager, RESTRICTED, SANDBOX, FULL
from managers.code_manager import CodeManager
from managers.system_health import SystemHealthManager
from core.memory import ConversationMemory


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def tmp_base(tmp_path):
    """Geçici proje kökü oluştur."""
    (tmp_path / "temp").mkdir()
    return tmp_path


@pytest.fixture
def security_sandbox(tmp_base):
    return SecurityManager("sandbox", tmp_base)


@pytest.fixture
def security_full(tmp_base):
    return SecurityManager("full", tmp_base)


@pytest.fixture
def security_restricted(tmp_base):
    return SecurityManager("restricted", tmp_base)


@pytest.fixture
def code_manager_sandbox(security_sandbox, tmp_base):
    return CodeManager(security_sandbox, tmp_base)


@pytest.fixture
def code_manager_full(security_full, tmp_base):
    return CodeManager(security_full, tmp_base)


# ─────────────────────────────────────────────
#  GÜVENLİK TESTLERİ
# ─────────────────────────────────────────────

class TestSecurityManager:
    def test_restricted_cannot_write(self, security_restricted):
        assert security_restricted.can_write("/tmp/test.py") is False

    def test_restricted_can_read(self, security_restricted):
        assert security_restricted.can_read() is True

    def test_restricted_cannot_execute(self, security_restricted):
        assert security_restricted.can_execute() is False

    def test_sandbox_can_write_to_temp(self, security_sandbox, tmp_base):
        safe_path = str(tmp_base / "temp" / "test.py")
        assert security_sandbox.can_write(safe_path) is True

    def test_sandbox_cannot_write_outside_temp(self, security_sandbox, tmp_base):
        outside = str(tmp_base / "outside.py")
        assert security_sandbox.can_write(outside) is False

    def test_sandbox_cannot_execute(self, security_sandbox):
        assert security_sandbox.can_execute() is False

    def test_full_can_write_anywhere(self, security_full, tmp_base):
        assert security_full.can_write(str(tmp_base / "anything.py")) is True

    def test_full_can_execute(self, security_full):
        assert security_full.can_execute() is True

    def test_level_values(self, tmp_base):
        assert SecurityManager("restricted", tmp_base).level == RESTRICTED
        assert SecurityManager("sandbox", tmp_base).level == SANDBOX
        assert SecurityManager("full", tmp_base).level == FULL


# ─────────────────────────────────────────────
#  KOD YÖNETİCİSİ TESTLERİ
# ─────────────────────────────────────────────

class TestCodeManager:
    def test_read_existing_file(self, code_manager_sandbox, tmp_base):
        test_file = tmp_base / "hello.py"
        test_file.write_text("print('hello')", encoding="utf-8")
        ok, content = code_manager_sandbox.read_file(str(test_file))
        assert ok is True
        assert "print" in content

    def test_read_nonexistent_file(self, code_manager_sandbox):
        ok, msg = code_manager_sandbox.read_file("/nonexistent/path/file.py")
        assert ok is False
        assert "bulunamadı" in msg.lower()

    def test_write_to_temp_sandbox(self, code_manager_sandbox, tmp_base):
        path = str(tmp_base / "temp" / "test_output.py")
        ok, msg = code_manager_sandbox.write_file(path, "x = 1\n", validate=True)
        assert ok is True
        assert Path(path).exists()

    def test_write_rejected_outside_temp_sandbox(self, code_manager_sandbox, tmp_base):
        path = str(tmp_base / "not_allowed.py")
        ok, msg = code_manager_sandbox.write_file(path, "x = 1\n")
        assert ok is False
        assert "openclaw" in msg.lower() or "yetki" in msg.lower()

    def test_valid_python_syntax(self, code_manager_sandbox):
        ok, msg = code_manager_sandbox.validate_python_syntax("def foo(): return 42")
        assert ok is True

    def test_invalid_python_syntax(self, code_manager_sandbox):
        ok, msg = code_manager_sandbox.validate_python_syntax("def foo( return 42")
        assert ok is False
        assert "sözdizimi" in msg.lower() or "syntax" in msg.lower()

    def test_valid_json(self, code_manager_sandbox):
        ok, msg = code_manager_sandbox.validate_json('{"key": "value"}')
        assert ok is True

    def test_invalid_json(self, code_manager_sandbox):
        ok, msg = code_manager_sandbox.validate_json('{key: value}')
        assert ok is False

    def test_list_directory(self, code_manager_sandbox, tmp_base):
        ok, listing = code_manager_sandbox.list_directory(str(tmp_base))
        assert ok is True
        assert "temp" in listing

    def test_audit_project(self, code_manager_full, tmp_base):
        # Geçerli python dosyası oluştur
        (tmp_base / "sample.py").write_text("x = 1\n", encoding="utf-8")
        report = code_manager_full.audit_project(str(tmp_base))
        assert "Denetim Raporu" in report

    def test_metrics_increment(self, code_manager_sandbox, tmp_base):
        test_file = tmp_base / "metric_test.py"
        test_file.write_text("pass\n", encoding="utf-8")
        code_manager_sandbox.read_file(str(test_file))
        code_manager_sandbox.validate_python_syntax("x = 1")
        m = code_manager_sandbox.get_metrics()
        assert m["files_read"] >= 1
        assert m["syntax_checks"] >= 1

    def test_no_write_on_syntax_error(self, code_manager_full, tmp_base):
        path = str(tmp_base / "bad.py")
        ok, msg = code_manager_full.write_file(path, "def bad( pass", validate=True)
        assert ok is False
        assert not Path(path).exists()


# ─────────────────────────────────────────────
#  SİSTEM SAĞLIĞI TESTLERİ
# ─────────────────────────────────────────────

class TestSystemHealthManager:
    def test_full_report_runs(self):
        health = SystemHealthManager(use_gpu=False)
        report = health.full_report()
        assert "Sistem Sağlık Raporu" in report

    def test_gpu_optimize_no_crash(self):
        health = SystemHealthManager(use_gpu=False)
        result = health.optimize_gpu_memory()
        assert "GC" in result or "temizlendi" in result.lower()

    def test_memory_info(self):
        health = SystemHealthManager(use_gpu=False)
        mem = health.get_memory_info()
        # psutil kuruluysa dolu, değilse boş dict
        if mem:
            assert "total_gb" in mem
            assert mem["total_gb"] > 0


# ─────────────────────────────────────────────
#  BELLEK TESTLERİ
# ─────────────────────────────────────────────

class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory(max_turns=5)
        mem.add("user", "Merhaba")
        mem.add("assistant", "Selam!")
        history = mem.get_history()
        assert len(history) == 2

    def test_max_turns_window(self):
        mem = ConversationMemory(max_turns=2)
        for i in range(10):
            mem.add("user", f"mesaj {i}")
            mem.add("assistant", f"yanıt {i}")
        # max_turns * 2 = 4 mesaj tutulmalı
        assert len(mem) <= 4

    def test_last_file_tracking(self):
        mem = ConversationMemory()
        assert mem.get_last_file() is None
        mem.set_last_file("/path/to/file.py")
        assert mem.get_last_file() == "/path/to/file.py"

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("user", "test")
        mem.set_last_file("test.py")
        mem.clear()
        assert len(mem) == 0
        assert mem.get_last_file() is None

    def test_messages_for_llm(self):
        mem = ConversationMemory()
        mem.add("user", "soru")
        mem.add("assistant", "yanıt")
        msgs = mem.get_messages_for_llm()
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert "timestamp" not in msgs[0]
