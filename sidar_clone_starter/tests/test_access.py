from pathlib import Path

from sidar_clone_starter.config import AccessLevel
from sidar_clone_starter.core.access import can_write


def test_restricted_cannot_write(tmp_path: Path):
    target = tmp_path / "a.txt"
    sandbox = tmp_path / "sandbox"
    assert can_write(AccessLevel.RESTRICTED, target, sandbox) is False


def test_sandbox_can_write_only_inside_sandbox(tmp_path: Path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    inside = sandbox / "ok.txt"
    outside = tmp_path / "no.txt"

    assert can_write(AccessLevel.SANDBOX, inside, sandbox) is True
    assert can_write(AccessLevel.SANDBOX, outside, sandbox) is False


def test_full_can_write_anywhere(tmp_path: Path):
    target = tmp_path / "x.txt"
    sandbox = tmp_path / "sandbox"
    assert can_write(AccessLevel.FULL, target, sandbox) is True
