import pytest
import signal
import subprocess
import sys
from pathlib import Path

class PopenWithCtrlC(subprocess.Popen):
    def __init__(self, *args, creationflags=0, **kwargs):
        if sys.platform == "win32":
            # needed on windows to separate ctrl-c lifecycle of subprocess from parent:
            creationflags = creationflags | subprocess.CREATE_NEW_CONSOLE  # type: ignore

        super().__init__(*args, **kwargs, creationflags=creationflags)

    def send_ctrl_c(self):
        # platform independent way to replicate the behavior of Ctrl-C:ing a cli app
        if sys.platform == "win32":
            # windows doesn't support sigint, and subprocess.CTRL_C_EVENT has a bunch
            # of gotchas since it's bound to a console which is the same for the parent
            # process by default, and can't be sent using the python standard library
            # to a separate process's console
            import console_ctrl

            console_ctrl.send_ctrl_c(self.pid)  # noqa [E731]
        else:
            self.send_signal(signal.SIGINT)


def test_shutdown():
    # We run it in a separate process so we can simulate interrupting it
    fn = Path(__file__).parent / "support" / "_shutdown.py"
    p = PopenWithCtrlC(
        [sys.executable, fn],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PYTHONUNBUFFERED": "1"},
    )
    for i in range(2):  # this number doesn't matter, it's a while loop
        assert p.stdout.readline() == b"running\n"
    p.send_ctrl_c()
    assert p.stdout.readline() == b"cancelled\n"
    assert p.stdout.readline() == b"handled cancellation\n"
    assert p.stdout.readline() == b"exit async\n"
    assert (
        p.stdout.readline() == b"keyboard interrupt\n"
    )  # we want the keyboard interrupt to come *after* the running function has been cancelled!

    stderr_content = p.stderr.read()
    assert b"Traceback" not in stderr_content


def test_keyboard_interrupt_reraised_as_is(synchronizer):
    @synchronizer.create_blocking
    async def a():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        a()


def test_shutdown_during_ctx_mgr_setup():
    # We run it in a separate process so we can simulate interrupting it
    fn = Path(__file__).parent / "support" / "_shutdown_ctx_mgr.py"
    p = PopenWithCtrlC(
        [sys.executable, fn, "enter"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PYTHONUNBUFFERED": "1"},
    )
    for i in range(2):  # this number doesn't matter, it's a while loop
        assert p.stdout.readline() == b"enter\n"
    p.send_ctrl_c()
    assert p.stdout.readline() == b"exit\n"
    assert p.stdout.readline() == b"keyboard interrupt\n"
    assert p.stderr.read() == b""


def test_shutdown_during_ctx_mgr_yield():
    # We run it in a separate process so we can simulate interrupting it
    fn = Path(__file__).parent / "support" / "_shutdown_ctx_mgr.py"
    p = PopenWithCtrlC(
        [sys.executable, fn, "yield"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PYTHONUNBUFFERED": "1"},
    )
    for i in range(2):  # this number doesn't matter, it's a while loop
        assert p.stdout.readline() == b"in ctx\n"
    p.send_ctrl_c()
    assert p.stdout.readline() == b"exit\n"
    assert p.stdout.readline() == b"keyboard interrupt\n"
    assert p.stderr.read() == b""


def test_shutdown_during_async_run():
    fn = Path(__file__).parent / "support" / "_shutdown_async_run.py"
    p = PopenWithCtrlC(
        [sys.executable, fn],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PYTHONUNBUFFERED": "1"},
        encoding="utf-8",
    )
    for i in range(2):  # this number doesn't matter, it's a while loop
        assert p.stdout.readline() == "running\n"
    p.send_ctrl_c()
    assert p.stdout.readline() == "cancelled\n"
    assert p.stdout.readline() == "handled cancellation\n"
    assert p.stdout.readline() == "exit async\n"
    assert (
        p.stdout.readline() == "keyboard interrupt\n"
    )  # we want the keyboard interrupt to come *after* the running function has been cancelled!

    stderr_content = p.stderr.read()
    assert "Traceback" not in stderr_content
