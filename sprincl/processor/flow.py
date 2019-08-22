from typing import Iterable

from ..base import Param


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
