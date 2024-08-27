import asyncio
import os
import pytest
import signal
import subprocess
import sys
import threading
import time

import synchronicity
from synchronicity.exceptions import SynchronizerShutdown


def test_shutdown():
    # We run it in a separate process so we can simulate interrupting it
    fn = os.path.join(os.path.dirname(__file__), "_shutdown.py")
    p = subprocess.Popen(
        [sys.executable, fn],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PYTHONUNBUFFERED": "1"},
    )
    for i in range(3):  # this number doesn't matter, it's a while loop
        assert p.stdout.readline() == b"running\n"
    p.send_signal(signal.SIGINT)
    assert p.stdout.readline() == b"cancelled\n"
    assert p.stdout.readline() == b"stopping\n"
    assert p.stdout.readline() == b"exiting\n"
    stderr_content = p.stderr.read()
    assert b"Traceback" not in stderr_content


def test_shutdown_raises_shutdown_error():
    s = synchronicity.Synchronizer()

    @s.create_blocking
    async def wrapped():
        await asyncio.sleep(10)

    def shut_down_soon():
        s._get_loop(start=True)  # ensure loop is running
        time.sleep(0.1)
        s._close_loop()

    t = threading.Thread(target=shut_down_soon)
    t.start()

    with pytest.raises(SynchronizerShutdown):
        wrapped()

    t.join()


@pytest.mark.asyncio
async def test_shutdown_raises_shutdown_error_async():
    s = synchronicity.Synchronizer()

    @s.create_blocking
    async def wrapped():
        await asyncio.sleep(10)

    @s.create_blocking
    async def supercall():
        try:
            # loop-internal calls should propagate the CancelledError
            return await wrapped.aio()
        except asyncio.CancelledError:
            raise  # expected
        except BaseException:
            raise Exception("asyncio.CancelledError is expected internally")

    def shut_down_soon():
        s._get_loop(start=True)  # ensure loop is running
        time.sleep(0.1)
        s._close_loop()

    t = threading.Thread(target=shut_down_soon)
    t.start()

    with pytest.raises(SynchronizerShutdown):
        # calls from outside of the synchronizer loop should get the SynchronizerShutdown
        await supercall.aio()

    t.join()
