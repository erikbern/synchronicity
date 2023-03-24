import collections.abc
import contextlib
import functools
import inspect
import typing

from .exceptions import UserCodeException
from .interface import Interface
from contextlib import asynccontextmanager as _asynccontextmanager


def type_compat_wraps(func, interface: Interface, new_annotations=None):
    """Like functools.wraps but maintains `inspect.iscoroutinefunction` and allows custom type annotations overrides

    Use this when the wrapper function is non-async but returns the coroutine resulting
    from calling the underlying wrapped `func`. This will make sure that the wrapper
    is still an async function in that case, and can be inspected as such.

    Note: Does not forward async generator information other than explicit annotations
    """
    if inspect.iscoroutinefunction(func) and interface == Interface.ASYNC:

        def asyncfunc_deco(user_wrapper):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await user_wrapper(*args, **kwargs)
                except UserCodeException as uc_exc:
                    raise uc_exc.exc from None

            if new_annotations:
                wrapper.__annotations__ = new_annotations
            return wrapper

        return asyncfunc_deco
    else:

        def blockingfunc_deco(user_wrapper):
            wrapped = functools.wraps(func)(user_wrapper)

            if new_annotations:
                wrapped.__annotations__ = new_annotations

            return wrapped

        return blockingfunc_deco


YIELD_TYPE = typing.TypeVar("YIELD_TYPE")
SEND_TYPE = typing.TypeVar("SEND_TYPE")


def asynccontextmanager(
    f: typing.AsyncGenerator[YIELD_TYPE, SEND_TYPE]
) -> typing.Callable[[], typing.AsyncContextManager[YIELD_TYPE]]:
    """Wrapper around contextlib.asynccontextmanager that sets correct type annotations

    The standard library one doesn't
    """
    acm_factory = _asynccontextmanager(f)

    old_ret = acm_factory.__annotations__.pop("return", None)
    if old_ret is not None:
        if old_ret.__origin__ in [
            collections.abc.AsyncGenerator,
            collections.abc.AsyncIterator,
            collections.abc.AsyncIterator,
        ]:
            acm_factory.__annotations__["return"] = typing.AsyncContextManager[
                old_ret.__args__[0]
            ]
        elif old_ret.__origin__ == contextlib.AbstractAsyncContextManager:
            # if the standard lib fixes the annotations in the future, lets not break it...
            return acm_factory
    else:
        raise ValueError(
            "To use the fixed @asynccontextmanager, make sure to properly annotate your wrapped function as an AsyncGenerator"
        )

    return acm_factory
