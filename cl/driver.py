import ctypes
import platform
import json
import os
from contextlib import contextmanager

module_path = os.path.abspath(os.path.dirname(__file__))

from .cregistry import CRegistry


def generate_len_mappings(signature):
    len_map = []
    argnames = list(signature.keys())

    for i in range(len(argnames)):
        if argnames[i].startswith('num_'):
            try:
                len_map.append((argnames[i], argnames[i + 1]))
            except:
                pass
    return len_map


def apply_auto_len(formatted_args, len_map):
    for k, v in len_map:
        if k not in formatted_args.arguments and v in formatted_args.arguments:
            formatted_args.arguments[k] = len(formatted_args.arguments[v])


class CL(CRegistry):
    def __init__(self, registry=None, library=None, *args, **kwargs):
        if library is None:
            if platform.system() == 'Windows':
                self.library = ctypes.cdll.LoadLibrary('OpenCL.dll')

            elif platform.system() == "Darwin":
                self.library = ctypes.cdll.LoadLibrary('OpenCL.framework/OpenCL')

            else:
                self.library = ctypes.cdll.LoadLibrary('libOpenCL.so')

        if registry is None:
            with open(os.path.join(module_path, "cl.json"), 'r') as fh:
                registry = json.load(fh)

        super().__init__(registry, *args, **kwargs)
        self.func_cache = {}
        self.error_mode = True
        self.apply_defaults = True
        self.auto_len = True

    @contextmanager
    def check_error(self):
        error_mode = self.error_mode
        self.set_error_mode(True)
        try:
            yield
        finally:
            self.set_error_mode(error_mode)

    def check(self):
        for k in self.registry:
            getattr(self, k)

    def set_error_mode(self, mode):
        assert isinstance(mode, bool)
        self.error_mode = mode

    def __getattr__(self, item):
        if item in self.func_cache:
            return self.func_cache[item]

        if hasattr(self.library, item):
            func = self(item)(getattr(self.library, item))
            type(func).__call__ = self.decorate(func, item)
            self.func_cache[item] = func
            return func

        return self(item)

    def catch_exception(self, error_code):
        if error_code.value:
            raise Exception(self.ErrorCodes[error_code.value])

    def decorate(self, func, name):
        inner = func.__call__
        sig = func.__signature__
        error_argument = 'errcode_ret' in sig.parameters
        len_map = generate_len_mappings(sig.parameters)

        def wrapper(instance, *args, **kwargs):
            formatted_args = sig.bind(*args, **kwargs)
            post_exec_hook = lambda result: None

            if self.auto_len:
                apply_auto_len(formatted_args, len_map)

            if self.apply_defaults:
                formatted_args.apply_defaults()

            if self.error_mode:
                if error_argument:
                    errcode_ret_ptr = formatted_args.arguments.get('errcode_ret')
                    if not errcode_ret_ptr:
                        errcode_ret = self.cl_int()
                        errcode_ret_ptr = ctypes.byref(errcode_ret)
                        formatted_args.arguments['errcode_ret'] = errcode_ret_ptr

                    errcode_ret = ctypes.cast(errcode_ret_ptr, ctypes.POINTER(self.cl_int)).contents
                    post_exec_hook = lambda result: self.catch_exception(errcode_ret)

                else:
                    post_exec_hook = lambda result: self.catch_exception(result)

            result = inner(*formatted_args.args)
            post_exec_hook(result)
            return result

        return wrapper

