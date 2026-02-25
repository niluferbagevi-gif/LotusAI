from pathlib import Path

from sidar_clone_starter.agents.engineer import EngineerAgent
from sidar_clone_starter.config import Settings, AccessLevel


def build_agent(tmp_path: Path, level: AccessLevel = AccessLevel.SANDBOX) -> EngineerAgent:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    settings = Settings(access_level=level, sandbox_dir=sandbox)
    return EngineerAgent(settings)


def test_write_and_read_in_sandbox(tmp_path: Path):
    agent = build_agent(tmp_path)
    file_path = tmp_path / "sandbox" / "demo.py"

    result = agent.handle(f"write {file_path} | print('ok')")
    assert result["ok"] is True

    read_result = agent.handle(f"read {file_path}")
    assert read_result["ok"] is True
    assert "print('ok')" in read_result["content"]


def test_write_denied_in_restricted(tmp_path: Path):
    agent = build_agent(tmp_path, level=AccessLevel.RESTRICTED)
    file_path = tmp_path / "sandbox" / "demo.py"

    result = agent.handle(f"write {file_path} | print('x')")
    assert result["ok"] is False


def test_error_analysis():
    settings = Settings()
    agent = EngineerAgent(settings)

    output = agent.analyze_error("Traceback... SyntaxError: invalid syntax")
    assert output["error_type"] == "syntax_error"
