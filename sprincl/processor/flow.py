from typing import Iterable as IterableBase

from ..base import Param
from .base import Processor


class IterableMeta(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, IterableBase) and all(isinstance(obj, self.__subtype__) for obj in instance)


class Iterable(metaclass=IterableMeta):
    __subtype__ = object

    def __class_getitem__(cls, params):
        if not isinstance(params, tuple):
            params = (params,)
        if not params:
            raise TypeError("At least one subtypes must be specified!")
        if not all(isinstance(obj, type) for obj in params):
            raise TypeError("Subtypes must be types!")
        return type('{}[{}]'.format(cls.__name__, params), (cls,), {'__subtype__': params})


class Shaper(Processor):
    shape = Param(Iterable[int])

    def function(self, data):
        return tuple(data[index] for index in self.shape)


class GroupProcessor(Processor):
    children = Param(Iterable[Processor], default=tuple())


class Parallel(GroupProcessor):
    def function(self, data):
        result = []
        for child, element in zip(self.children, data):
            out = child(element)
            result.append(out)
        return result


class Sequential(GroupProcessor):
    def function(self, data):
        for child in self.children:
            data = child(data)
        return data
