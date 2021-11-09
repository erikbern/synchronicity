class UserCodeException(Exception):
    """This is used to wrap and unwrap exceptions in "user code".

    This lets us have cleaner tracebacks without all the internal synchronicity stuff."""

    def __init__(self, exc):
        # There's always going to be one place inside synchronicity where we
        # catch the exception. We can always safely remove that frame from the
        # traceback.
        self.exc = exc.with_traceback(exc.__traceback__.tb_next)


def wrap_coro_exception(coro):
    async def coro_wrapped():
        try:
            await coro
        except Exception as exc:
            raise UserCodeException(exc)

    return coro_wrapped()