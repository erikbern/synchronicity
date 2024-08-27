import asyncio
import atexit
import collections.abc
import concurrent.futures
import contextlib
import functools
import inspect
import platform
import threading
import types
import typing
import warnings
from typing import ForwardRef, Optional

import typing_extensions

from synchronicity.annotations import evaluated_annotation
from synchronicity.combined_types import FunctionWithAio, MethodWithAio

from .async_wrap import wraps_by_interface
from .callback import Callback
from .exceptions import SynchronizerShutdown, UserCodeException, unwrap_coro_exception, wrap_coro_exception
from .interface import Interface

_BUILTIN_ASYNC_METHODS = {
    "__aiter__": "__iter__",
    "__aenter__": "__enter__",
    "__aexit__": "__exit__",
    "__anext__": "__next__",
}

IGNORED_ATTRIBUTES = (
    # the "zope" lib monkey patches in some non-introspectable stuff on stdlib abc.ABC.
    # Ignoring __provides__ fixes an incompatibility with `channels[daphne]`,
    # where Synchronizer creation fails when wrapping contextlib._AsyncGeneratorContextManager
    "__provides__",
)

_RETURN_FUTURE_KWARG = "_future"

# Default names for classes
_CLASS_PREFIXES = {
    Interface.BLOCKING: "Blocking",
    Interface.ASYNC: "Async",
}

# Default names for functions
_FUNCTION_PREFIXES = {
    Interface.BLOCKING: "blocking_",
    Interface.ASYNC: "async_",  # deprecated, will be removed soon!
    Interface._ASYNC_WITH_BLOCKING_TYPES: "aio_",
}

TARGET_INTERFACE_ATTR = "_sync_target_interface"
SYNCHRONIZER_ATTR = "_sync_synchronizer"


ASYNC_GENERIC_ORIGINS = (
    collections.abc.Awaitable,
    collections.abc.Coroutine,
    collections.abc.AsyncIterator,
    collections.abc.AsyncIterable,
    collections.abc.AsyncGenerator,
    contextlib.AbstractAsyncContextManager,
)


def _type_requires_aio_usage(annotation, declaration_module):
    if isinstance(annotation, ForwardRef):
        annotation = annotation.__forward_arg__
    if isinstance(annotation, str):
        try:
            annotation = evaluated_annotation(annotation, declaration_module=declaration_module)
        except Exception:
            # TODO: this will be incorrect in special case of `arg: "Awaitable[some_forward_ref_type]"`,
            #       but its a hard problem to solve without passing around globals everywhere
            return False

    if hasattr(annotation, "__origin__"):
        if annotation.__origin__ in ASYNC_GENERIC_ORIGINS:  # type: ignore
            return True
        # recurse for generic subtypes
        for a in getattr(annotation, "__args__", ()):
            if _type_requires_aio_usage(a, declaration_module):
                return True
    return False


def should_have_aio_interface(func):
    # determines if a blocking function gets an .aio attribute with an async interface to the function or not
    if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
        return True
    # check annotations if they contain any async entities that would need an event loop to be translated:
    # This catches things like vanilla functions returning Coroutines
    annos = getattr(func, "__annotations__", {})
    for anno in annos.values():
        if _type_requires_aio_usage(anno, func.__module__):
            return True
    return False


class Synchronizer:
    """Helps you offer a blocking (synchronous) interface to asynchronous code."""

    _stopping: asyncio.Event

    def __init__(
        self,
        multiwrap_warning=False,
        async_leakage_warning=True,
    ):
        self._multiwrap_warning = multiwrap_warning
        self._async_leakage_warning = async_leakage_warning
        self._loop = None
        self._loop_creation_lock = threading.Lock()
        self._thread = None
        self._stopping = None

        if platform.system() == "Windows":
            # default event loop policy on windows spits out errors when
            # closing the event loop, so use WindowsSelectorEventLoopPolicy instead
            # https://stackoverflow.com/questions/45600579/asyncio-event-loop-is-closed-when-getting-loop
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Special attribute we use to go from wrapped <-> original
        self._wrapped_attr = "_sync_wrapped_%d" % id(self)
        self._original_attr = "_sync_original_%d" % id(self)

        # Special attribute to mark something as non-wrappable
        self._nowrap_attr = "_sync_nonwrap_%d" % id(self)
        self._input_translation_attr = "_sync_input_translation_%d" % id(self)
        self._output_translation_attr = "_sync_output_translation_%d" % id(self)

        # Prep a synchronized context manager in case one is returned and needs translation
        self._ctx_mgr_cls = contextlib._AsyncGeneratorContextManager
        self.create_async(self._ctx_mgr_cls)
        self.create_blocking(self._ctx_mgr_cls)

        atexit.register(self._close_loop)

    _PICKLE_ATTRS = [
        "_multiwrap_warning",
        "_async_leakage_warning",
    ]

    def __getstate__(self):
        return dict([(attr, getattr(self, attr)) for attr in self._PICKLE_ATTRS])

    def __setstate__(self, d):
        for attr in self._PICKLE_ATTRS:
            setattr(self, attr, d[attr])

    def _start_loop(self):
        with self._loop_creation_lock:
            if self._loop and self._loop.is_running():
                return self._loop

            is_ready = threading.Event()

            def thread_inner():
                async def loop_inner():
                    self._loop = asyncio.get_running_loop()
                    self._stopping = asyncio.Event()
                    is_ready.set()
                    await self._stopping.wait()  # wait until told to stop

                try:
                    asyncio.run(loop_inner())
                except RuntimeError as exc:
                    # Python 3.12 raises a RuntimeError when new threads are created at shutdown.
                    # Swallowing it here is innocuous, but ideally we will revisit this after
                    # refactoring the shutdown handlers that modal uses to avoid triggering it.
                    if "can't create new thread at interpreter shutdown" not in str(exc):
                        raise exc

            self._thread = threading.Thread(target=thread_inner, daemon=True)
            self._thread.start()
            is_ready.wait()  # TODO: this might block for a very short time
            return self._loop

    def _close_loop(self):
        if self._thread is not None:
            if not self._loop.is_closed():
                # This also serves the purpose of waking up an idle loop
                self._loop.call_soon_threadsafe(self._stopping.set)
            self._thread.join()
            self._thread = None

    def __del__(self):
        self._close_loop()

    def _get_loop(self, start=False):
        if self._loop is None and start:
            return self._start_loop()
        return self._loop

    def _get_running_loop(self):
        # TODO: delete this method
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return

    def _is_inside_loop(self):
        loop = self._get_loop()
        if loop is None:
            return False
        if threading.current_thread() != self._thread:
            # gevent does something bad that causes asyncio.get_running_loop() to return self._loop
            return False
        current_loop = self._get_running_loop()
        return loop == current_loop

    def _wrap_check_async_leakage(self, coro):
        """Check if a coroutine returns another coroutine (or an async generator) and warn.

        The reason this is important to catch is that otherwise even synchronized code might end up
        "leaking" async code into the caller.
        """
        if not self._async_leakage_warning:
            return coro

        async def coro_wrapped():
            value = await coro
            # TODO: we should include the name of the original function here
            if inspect.iscoroutine(value):
                warnings.warn(f"Potential async leakage: coroutine returned a coroutine {value}.")
            elif inspect.isasyncgen(value):
                warnings.warn(f"Potential async leakage: Coroutine returned an async generator {value}.")
            return value

        return coro_wrapped()

    def _wrap_instance(self, obj, interface):
        # Takes an object and creates a new proxy object for it
        cls = obj.__class__
        cls_dct = cls.__dict__
        interfaces = cls_dct[self._wrapped_attr]
        if interface not in interfaces:
            raise RuntimeError(f"Class {cls} has not synchronized {interface}.")
        interface_cls = interfaces[interface]
        new_obj = interface_cls.__new__(interface_cls)
        # Store a reference to the original object
        new_obj.__dict__[self._original_attr] = obj
        new_obj.__dict__[SYNCHRONIZER_ATTR] = self
        new_obj.__dict__[TARGET_INTERFACE_ATTR] = interface
        return new_obj

    def _translate_scalar_in(self, obj):
        # If it's an external object, translate it to the internal type
        if hasattr(obj, "__dict__"):
            if inspect.isclass(obj):  # TODO: functions?
                return obj.__dict__.get(self._original_attr, obj)
            else:
                return obj.__dict__.get(self._original_attr, obj)
        else:
            return obj

    def _translate_scalar_out(self, obj, interface):
        if interface == Interface._ASYNC_WITH_BLOCKING_TYPES:
            interface = Interface.BLOCKING

        # If it's an internal object, translate it to the external interface
        if inspect.isclass(obj):  # TODO: functions?
            cls_dct = obj.__dict__
            if self._wrapped_attr in cls_dct:
                return cls_dct[self._wrapped_attr][interface]
            else:
                return obj
        elif isinstance(obj, (typing.TypeVar, typing_extensions.ParamSpec)):
            if hasattr(obj, self._wrapped_attr):
                return getattr(obj, self._wrapped_attr)[interface]
            else:
                return obj
        else:
            cls_dct = obj.__class__.__dict__
            if self._wrapped_attr in cls_dct:
                # This is an *instance* of a synchronized class, translate its type
                return self._wrap(obj, interface)
            else:
                return obj

    def _recurse_map(self, mapper, obj):
        if type(obj) == list:  # noqa: E721
            return list(self._recurse_map(mapper, item) for item in obj)
        elif type(obj) == tuple:  # noqa: E721
            return tuple(self._recurse_map(mapper, item) for item in obj)
        elif type(obj) == dict:  # noqa: E721
            return dict((key, self._recurse_map(mapper, item)) for key, item in obj.items())
        else:
            return mapper(obj)

    def _translate_in(self, obj):
        return self._recurse_map(self._translate_scalar_in, obj)

    def _translate_out(self, obj, interface):
        return self._recurse_map(lambda scalar: self._translate_scalar_out(scalar, interface), obj)

    def _translate_coro_out(self, coro, interface, original_func):
        async def unwrap_coro():
            res = await coro
            if getattr(original_func, self._output_translation_attr, True):
                return self._translate_out(res, interface)
            return res

        return unwrap_coro()

    def _run_function_sync(self, coro, interface, original_func):
        if self._is_inside_loop():
            raise Exception("Deadlock detected: calling a sync function from the synchronizer loop")

        coro = wrap_coro_exception(coro)
        coro = self._wrap_check_async_leakage(coro)
        loop = self._get_loop(start=True)
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            value = fut.result()
        except concurrent.futures.CancelledError:
            if not self._loop or self._stopping.is_set():
                # this allows differentiate between wrapped code raising concurrent.futures.CancelledError
                # and synchronicity itself raising it due to the event loop shutting down while waiting
                # for the synchronizer.
                raise SynchronizerShutdown()
            raise

        if getattr(original_func, self._output_translation_attr, True):
            return self._translate_out(value, interface)
        return value

    def _run_function_sync_future(self, coro, interface, original_func):
        coro = wrap_coro_exception(coro)
        coro = self._wrap_check_async_leakage(coro)
        loop = self._get_loop(start=True)
        # For futures, we unwrap the result at this point, not in f_wrapped
        coro = unwrap_coro_exception(coro)
        coro = self._translate_coro_out(coro, interface, original_func)
        return asyncio.run_coroutine_threadsafe(coro, loop)

    async def _run_function_async(self, coro, interface, original_func):
        coro = wrap_coro_exception(coro)
        coro = self._wrap_check_async_leakage(coro)
        loop = self._get_loop(start=True)

        if self._is_inside_loop():
            value = await coro
        else:
            try:
                c_fut = asyncio.run_coroutine_threadsafe(coro, loop)
                a_fut = asyncio.wrap_future(c_fut)
                value = await a_fut
            except asyncio.CancelledError:
                if not self._loop or self._stopping.is_set():
                    raise SynchronizerShutdown()
                raise

        if getattr(original_func, self._output_translation_attr, True):
            return self._translate_out(value, interface)
        return value

    def _run_generator_sync(self, gen, interface, original_func):
        value, is_exc = None, False
        while True:
            try:
                if is_exc:
                    value = self._run_function_sync(gen.athrow(value), interface, original_func)
                else:
                    value = self._run_function_sync(gen.asend(value), interface, original_func)
            except UserCodeException as uc_exc:
                raise uc_exc.exc from None
            except StopAsyncIteration:
                break
            try:
                value = yield value
                is_exc = False
            except BaseException as exc:
                value = exc
                is_exc = True

    async def _run_generator_async(self, gen, interface, original_func):
        value, is_exc = None, False
        while True:
            try:
                if is_exc:
                    value = await self._run_function_async(gen.athrow(value), interface, original_func)
                else:
                    value = await self._run_function_async(gen.asend(value), interface, original_func)
            except UserCodeException as uc_exc:
                raise uc_exc.exc from None
            except StopAsyncIteration:
                break
            try:
                value = yield value
                is_exc = False
            except BaseException as exc:
                value = exc
                is_exc = True

    def create_callback(self, f, interface):
        return Callback(self, f, interface)

    def _update_wrapper(self, f_wrapped, f, name=None, interface=None, target_module=None):
        """Very similar to functools.update_wrapper"""
        functools.update_wrapper(f_wrapped, f)
        if name is not None:
            f_wrapped.__name__ = name
            f_wrapped.__qualname__ = name
        if target_module is not None:
            f_wrapped.__module__ = target_module
        setattr(f_wrapped, SYNCHRONIZER_ATTR, self)
        setattr(f_wrapped, TARGET_INTERFACE_ATTR, interface)

    def _wrap_callable(
        self,
        f,
        interface,
        name=None,
        allow_futures=True,
        unwrap_user_excs=True,
        target_module=None,
        include_aio_interface=True,
    ):
        if hasattr(f, self._original_attr):
            if self._multiwrap_warning:
                warnings.warn(f"Function {f} is already wrapped, but getting wrapped again")
            return f

        if name is None:
            _name = _FUNCTION_PREFIXES[interface] + f.__name__
        else:
            _name = name

        is_coroutinefunction = inspect.iscoroutinefunction(f)

        @wraps_by_interface(interface, f)
        def f_wrapped(*args, **kwargs):
            return_future = kwargs.pop(_RETURN_FUTURE_KWARG, False)

            # If this gets called with an argument that represents an external type,
            # translate it into an internal type
            if getattr(f, self._input_translation_attr, True):
                args = self._translate_in(args)
                kwargs = self._translate_in(kwargs)

            # Call the function
            res = f(*args, **kwargs)

            # Figure out if this is a coroutine or something
            is_coroutine = inspect.iscoroutine(res)
            is_asyncgen = inspect.isasyncgen(res)

            if return_future:
                if not allow_futures:
                    raise Exception("Can not return future for this function")
                elif is_coroutine:
                    return self._run_function_sync_future(res, interface, f)
                elif is_asyncgen:
                    raise Exception("Can not return futures for generators")
                else:
                    return res
            elif is_coroutine:
                if interface in (Interface.ASYNC, Interface._ASYNC_WITH_BLOCKING_TYPES):
                    coro = self._run_function_async(res, interface, f)
                    if not is_coroutinefunction:
                        # If this is a non-async function that returns a coroutine,
                        # then this is the exit point, and we need to unwrap any
                        # wrapped exception here. Otherwise, the exit point is
                        # in async_wrap.py
                        coro = unwrap_coro_exception(coro)
                    return coro
                elif interface == Interface.BLOCKING:
                    # This is the exit point, so we need to unwrap the exception here
                    try:
                        return self._run_function_sync(res, interface, f)
                    except StopAsyncIteration:
                        # this is a special case for handling __next__ wrappers around
                        # __anext__ that raises StopAsyncIteration
                        raise StopIteration()
                    except UserCodeException as uc_exc:
                        # Used to skip a frame when called from `proxy_method`.
                        if unwrap_user_excs and not (Interface.BLOCKING and include_aio_interface):
                            raise uc_exc.exc from None
                        else:
                            raise uc_exc
            elif is_asyncgen:
                # Note that the _run_generator_* functions handle their own
                # unwrapping of exceptions (this happens during yielding)
                if interface in (Interface.ASYNC, Interface._ASYNC_WITH_BLOCKING_TYPES):
                    return self._run_generator_async(res, interface, f)
                elif interface == Interface.BLOCKING:
                    return self._run_generator_sync(res, interface, f)
            else:
                if inspect.isfunction(res) or isinstance(res, functools.partial):  # TODO: HACKY HACK
                    # TODO: this is needed for decorator wrappers that returns functions
                    # Maybe a bit of a hacky special case that deserves its own decorator
                    @wraps_by_interface(interface, res)
                    def f_wrapped(*args, **kwargs):
                        args = self._translate_in(args)
                        kwargs = self._translate_in(kwargs)
                        f_res = res(*args, **kwargs)
                        if getattr(f, self._output_translation_attr, True):
                            return self._translate_out(f_res, interface)
                        else:
                            return f_res

                    return f_wrapped

                if getattr(f, self._output_translation_attr, True):
                    return self._translate_out(res, interface)
                else:
                    return res

        self._update_wrapper(f_wrapped, f, _name, interface, target_module=target_module)
        setattr(f_wrapped, self._original_attr, f)

        if interface == Interface.BLOCKING and include_aio_interface and should_have_aio_interface(f):
            # special async interface
            # this async interface returns *blocking* instances of wrapped objects, not async ones:
            async_interface = self._wrap_callable(
                f,
                interface=Interface._ASYNC_WITH_BLOCKING_TYPES,
                name=name,
                allow_futures=allow_futures,
                unwrap_user_excs=unwrap_user_excs,
                target_module=target_module,
            )
            f_wrapped = FunctionWithAio(f_wrapped, async_interface, self)
            self._update_wrapper(f_wrapped, f, _name, interface, target_module=target_module)
            setattr(f_wrapped, self._original_attr, f)

        return f_wrapped

    def _wrap_proxy_method(
        synchronizer_self,
        method,
        interface,
        allow_futures=True,
        include_aio_interface=True,
    ):
        if getattr(method, synchronizer_self._nowrap_attr, None):
            # This method is marked as non-wrappable
            return method

        wrapped_method = synchronizer_self._wrap_callable(
            method,
            interface,
            allow_futures=allow_futures,
            unwrap_user_excs=False,
        )

        @wraps_by_interface(interface, wrapped_method)
        def proxy_method(self, *args, **kwargs):
            instance = self.__dict__[synchronizer_self._original_attr]
            try:
                return wrapped_method(instance, *args, **kwargs)
            except UserCodeException as uc_exc:
                raise uc_exc.exc from None

        if interface == Interface.BLOCKING and include_aio_interface and should_have_aio_interface(method):
            async_proxy_method = synchronizer_self._wrap_proxy_method(
                method, Interface._ASYNC_WITH_BLOCKING_TYPES, allow_futures
            )
            return MethodWithAio(proxy_method, async_proxy_method, synchronizer_self)

        return proxy_method

    def _wrap_proxy_staticmethod(self, method, interface):
        orig_function = method.__func__
        method = self._wrap_callable(orig_function, interface)
        if isinstance(method, FunctionWithAio):
            return method  # no need to wrap a FunctionWithAio in a staticmethod, as it won't get bound anyways
        return staticmethod(method)

    def _wrap_proxy_classmethod(self, orig_classmethod, interface):
        orig_func = orig_classmethod.__func__
        method = self._wrap_callable(orig_func, interface, include_aio_interface=False)

        if interface == Interface.BLOCKING and should_have_aio_interface(orig_func):
            async_method = self._wrap_callable(orig_func, Interface._ASYNC_WITH_BLOCKING_TYPES)
            return MethodWithAio(method, async_method, self, is_classmethod=True)

        return classmethod(method)

    def _wrap_proxy_property(self, prop, interface):
        kwargs = {}
        for attr in ["fget", "fset", "fdel"]:
            if getattr(prop, attr):
                func = getattr(prop, attr)
                kwargs[attr] = self._wrap_proxy_method(
                    func, interface, allow_futures=False, include_aio_interface=False
                )
        return property(**kwargs)

    def _wrap_proxy_constructor(synchronizer_self, cls, interface):
        """Returns a custom __init__ for the subclass."""

        def my_init(self, *args, **kwargs):
            # Create base instance
            args = synchronizer_self._translate_in(args)
            kwargs = synchronizer_self._translate_in(kwargs)
            instance = cls(*args, **kwargs)

            # Register self as the wrapped one
            interface_instances = {interface: self}
            instance.__dict__[synchronizer_self._wrapped_attr] = interface_instances

            # Store a reference to the original object
            self.__dict__[synchronizer_self._original_attr] = instance

        synchronizer_self._update_wrapper(my_init, cls.__init__, interface=interface)
        setattr(my_init, synchronizer_self._original_attr, cls.__init__)
        return my_init

    def _wrap_class(self, cls, interface, name, target_module=None):
        new_bases = []
        for base in cls.__dict__.get("__orig_bases__", cls.__bases__):
            base_is_generic = hasattr(base, "__origin__")
            if base is object or (base_is_generic and base.__origin__ == typing.Generic):
                new_bases.append(base)  # no need to wrap these, just add them as base classes
            else:
                if base_is_generic:
                    wrapped_generic = self._wrap(base.__origin__, interface, require_already_wrapped=(name is not None))
                    new_bases.append(wrapped_generic.__class_getitem__(*base.__args__))
                else:
                    new_bases.append(self._wrap(base, interface, require_already_wrapped=(name is not None)))

        bases = tuple(new_bases)
        new_dict = {self._original_attr: cls}
        if cls is not None:
            new_dict["__init__"] = self._wrap_proxy_constructor(cls, interface)

        for k, v in cls.__dict__.items():
            if k in _BUILTIN_ASYNC_METHODS:
                k_sync = _BUILTIN_ASYNC_METHODS[k]
                if interface == Interface.BLOCKING:
                    new_dict[k_sync] = self._wrap_proxy_method(
                        v,
                        interface,
                        allow_futures=False,
                        include_aio_interface=False,
                    )
                    new_dict[k] = self._wrap_proxy_method(
                        v,
                        Interface._ASYNC_WITH_BLOCKING_TYPES,
                        allow_futures=False,
                    )
                elif interface == Interface.ASYNC:
                    new_dict[k] = self._wrap_proxy_method(v, interface, allow_futures=False)
            elif k in ("__new__", "__init__"):
                # Skip custom constructor in the wrapped class
                # Instead, delegate to the base class constructor and wrap it
                pass
            elif k in IGNORED_ATTRIBUTES:
                pass
            elif isinstance(v, staticmethod):
                # TODO(erikbern): this feels pretty hacky
                new_dict[k] = self._wrap_proxy_staticmethod(v, interface)
            elif isinstance(v, classmethod):
                new_dict[k] = self._wrap_proxy_classmethod(v, interface)
            elif isinstance(v, property):
                new_dict[k] = self._wrap_proxy_property(v, interface)
            elif isinstance(v, MethodWithAio):
                # if library defines its own "synchronicity-like" interface we transfer it "as is" to the wrapper
                new_dict[k] = v
            elif callable(v):
                new_dict[k] = self._wrap_proxy_method(v, interface)

        if name is None:
            name = _CLASS_PREFIXES[interface] + cls.__name__

        new_cls = types.new_class(name, bases, exec_body=lambda ns: ns.update(new_dict))
        new_cls.__module__ = cls.__module__ if target_module is None else target_module
        new_cls.__doc__ = cls.__doc__
        if "__annotations__" in cls.__dict__:
            new_cls.__annotations__ = cls.__annotations__  # transfer annotations

        setattr(new_cls, TARGET_INTERFACE_ATTR, interface)
        setattr(new_cls, SYNCHRONIZER_ATTR, self)
        return new_cls

    def _wrap(
        self,
        obj,
        interface,
        name=None,
        require_already_wrapped=False,
        target_module=None,
    ):
        # This method works for classes, functions, and instances
        # It wraps the object, and caches the wrapped object

        # Get the list of existing interfaces
        if hasattr(obj, "__dict__"):
            if self._wrapped_attr not in obj.__dict__:
                if isinstance(obj.__dict__, dict):
                    # This works for instances
                    obj.__dict__.setdefault(self._wrapped_attr, {})
                else:
                    # This works for classes & functions
                    setattr(obj, self._wrapped_attr, {})
            interfaces = obj.__dict__[self._wrapped_attr]
        else:
            # e.g., TypeVar in Python>=3.12
            if not hasattr(obj, self._wrapped_attr):
                setattr(obj, self._wrapped_attr, {})
            interfaces = getattr(obj, self._wrapped_attr)

        # If this is already wrapped, return the existing interface
        if interface in interfaces:
            if self._multiwrap_warning:
                warnings.warn(f"Object {obj} is already wrapped, but getting wrapped again")
            return interfaces[interface]

        if require_already_wrapped:
            # This happens if a class has a custom name but its base class doesn't
            raise RuntimeError(f"{obj} needs to be serialized explicitly with a custom name")

        # Wrap object (different cases based on the type)
        if inspect.isclass(obj):
            new_obj = self._wrap_class(
                obj,
                interface,
                name,
                target_module=target_module,
            )
        elif inspect.isfunction(obj):
            new_obj = self._wrap_callable(obj, interface, name, target_module=target_module)
        elif isinstance(obj, typing_extensions.ParamSpec):
            new_obj = self._wrap_param_spec(obj, interface, name, target_module)
        elif isinstance(obj, typing.TypeVar):
            new_obj = self._wrap_type_var(obj, interface, name, target_module)
        elif self._wrapped_attr in obj.__class__.__dict__:
            new_obj = self._wrap_instance(obj, interface)
        else:
            raise Exception("Argument %s is not a class or a callable" % obj)

        # Store the interface on the obj and return
        interfaces[interface] = new_obj
        return new_obj

    def _wrap_type_var(self, obj, interface, name, target_module):
        # TypeVar translation is needed only for type stub generation, in case the
        # "bound" attribute refers to a translatable type.

        # Creates a new identical TypeVar, marked with synchronicity's special attributes
        # This lets type stubs "translate" the `bounds` attribute on emitted type vars
        # if picked up from module scope and in generics using the base implementation type

        # TODO(elias): Refactor - since this isn't used for live apps, move type stub generation into genstub
        new_obj = typing.TypeVar(name, bound=obj.__bound__)  # noqa
        setattr(new_obj, self._original_attr, obj)
        setattr(new_obj, SYNCHRONIZER_ATTR, self)
        setattr(new_obj, TARGET_INTERFACE_ATTR, interface)
        new_obj.__module__ = target_module
        if not hasattr(obj, self._wrapped_attr):
            setattr(obj, self._wrapped_attr, {})
        getattr(obj, self._wrapped_attr)[interface] = new_obj
        return new_obj

    def _wrap_param_spec(self, obj, interface, name, target_module):
        # TODO(elias): Refactor - since this isn't used for live apps, move type stub generation into genstub
        new_obj = typing_extensions.ParamSpec(name)  # noqa
        setattr(new_obj, self._original_attr, obj)
        setattr(new_obj, SYNCHRONIZER_ATTR, self)
        setattr(new_obj, TARGET_INTERFACE_ATTR, interface)
        new_obj.__module__ = target_module
        if not hasattr(obj, self._wrapped_attr):
            setattr(obj, self._wrapped_attr, {})
        getattr(obj, self._wrapped_attr)[interface] = new_obj
        return new_obj

    def nowrap(self, obj):
        setattr(obj, self._nowrap_attr, True)
        return obj

    def no_input_translation(self, obj):
        setattr(obj, self._input_translation_attr, False)
        return obj

    def no_output_translation(self, obj):
        setattr(obj, self._output_translation_attr, False)
        return obj

    def no_io_translation(self, obj):
        return self.no_input_translation(self.no_output_translation(obj))

    # New interface that (almost) doesn't mutate objects
    def create_blocking(self, obj, name: Optional[str] = None, target_module: Optional[str] = None):
        wrapped = self._wrap(obj, Interface.BLOCKING, name, target_module=target_module)
        return wrapped

    def create_async(self, obj, name: Optional[str] = None, target_module: Optional[str] = None):
        wrapped = self._wrap(obj, Interface.ASYNC, name, target_module=target_module)
        return wrapped

    def is_synchronized(self, obj):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            return hasattr(obj, self._original_attr)
        else:
            return hasattr(obj.__class__, self._original_attr)
