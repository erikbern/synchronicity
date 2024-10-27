import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Windows can't fork")
def test_fork_restarts_loop():
    p = subprocess.Popen(
        [sys.executable, Path(__file__).parent / "support" / "_forker.py"],
        encoding="utf8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = p.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        p.kill()
        assert False, "Fork process hanged"

    assert p.returncode == 0
    assert stdout == "done\ndone\n"
