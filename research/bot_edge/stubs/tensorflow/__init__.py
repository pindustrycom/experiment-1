# stub: nateemma utils import tensorflow at module level; the indicator strategies never call it
import types, sys
class _Any(types.ModuleType):
    def __getattr__(self, k):
        if k.startswith('__'): raise AttributeError(k)
        return _Any(self.__name__ + '.' + k)
    def __call__(self, *a, **k): return _Any('call')
    def __mro_entries__(self, bases): return (object,)
_root = _Any('tensorflow'); sys.modules['tensorflow'] = _root
for n in ('keras','keras.layers','keras.models','keras.optimizers','keras.callbacks','keras.backend','keras.losses','keras.metrics','keras.regularizers','keras.utils'):
    sys.modules['tensorflow.'+n] = _Any('tensorflow.'+n)
