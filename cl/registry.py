import builtins
import logging
from collections.abc import Sequence, ByteString, Mapping

logging.basicConfig()


class Registry(Mapping):
    def __init__(self, registry=None, parent=None, *, cache=None, log_level=logging.INFO):
        if registry is None:
            registry = {}

        if cache is None:
            cache = {}

        self._registry = registry
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(log_level)
        self._cache = cache
        self._parent = parent

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def __contains__(self, item):
        return item in self._registry

    def __call__(self, item):
        # Cache lookup
        if item in self._cache:
            return self._cache[item]

        # Generate from registry
        elif item in self:
            #self._logger.info(f"{item} -> {self._registry[item]}")
            self._cache[item] = self._generate(self._registry[item])
            return self._cache[item]

        raise LookupError(item)

    def __getitem__(self, item):
        try:
            return self(item)
        except LookupError as error:
            raise KeyError(error)

    def _generate(self, obj):
        # Return instance
        if obj is None:
            return self

        # Return function call
        if isinstance(obj, Sequence):
            type_spec, *args, kwargs = obj
            return self._generate(type_spec)(*(self._generate(arg) for arg in args),
                                             **self._generate(kwargs))

        # Return mapping
        elif isinstance(obj, Mapping):
            return self.__class__(registry=obj, parent=self)

        # Return literal
        else:
            return obj


class SimpleRegistry(Registry):
    def _generate(self, obj):
        # Simplify string generation
        if isinstance(obj, str):
            return obj

        # Lookup class attribute for generation
        if isinstance(obj, Sequence):
            type_spec, *args = obj
            if isinstance(type_spec, str):
                type_spec = type.__getattribute__(self.__class__, type_spec)
            return self._generate(type_spec)(*(self._generate(arg) for arg in args))

        return super()._generate(obj)

