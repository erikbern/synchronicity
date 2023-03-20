import asyncio
import pytest

from synchronicity import Interface, Synchronizer


def test_is_synchronized():
    s = Synchronizer()

    class Foo:
        pass

    BlockingFoo = s.create_blocking(Foo)
    assert s.is_synchronized(Foo) == False
    assert s.is_synchronized(BlockingFoo) == True
