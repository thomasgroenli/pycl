import ctypes
from inspect import Parameter, Signature
from . import registry

class CRegistry(registry.SimpleRegistry):
    class CommandProto:
        @classmethod
        def type_or_any(cls, argtype):
            return argtype if not issubclass(argtype, cls) else ctypes.c_void_p


    def list(*args):
        return args

    def zipdict(keys, values):
        return {k: v for k, v in zip(keys, values)}

    def expr(self, code):
        return eval(code, {}, self)

    def function(self, argnames, code):
        return lambda *args: eval(code, {k: v for k, v in zip(argnames, args)}, self)

    def parent(self):
        return self.__getitem__

    def type(base, ptr_count, array_size):
        for i in range(ptr_count):
            base = ctypes.POINTER(base)

        if array_size > 0:
            base = base * array_size

        return base

    def define(name, type_info):
        return type(name, (type_info,), {})

    def struct(name, members, anonymous):
        return type(name, (ctypes.Structure,), {"_anonymous_": anonymous, "_fields_": members})

    def union(name, members, anonymous):
        return type(name, (ctypes.Union,), {"_anonymous_": anonymous, "_fields_": members})

    def command(name, rettype, args):
        class Command(CRegistry.CommandProto):
            parameters = {argname: argtype for argname, argtype in args}
            argtypes = [CRegistry.CommandProto.type_or_any(argtype) for argtype in parameters.values()]
            restype = rettype
            __signature__ = Signature([Parameter(argname, Parameter.POSITIONAL_OR_KEYWORD,
                                                 default=CRegistry.CommandProto.type_or_any(argtype)(),
                                                 annotation=argtype)
                                       for argname, argtype in args],
                                      return_annotation=rettype)


            def __init__(self, func_ptr):
                self.func_ptr = func_ptr
                self.func_ptr.argtypes = self.argtypes
                self.func_ptr.restype = self.restype

            def __call__(self, *args):
                return self.func_ptr(*args)

        Command.__qualname__ = name
        return Command
