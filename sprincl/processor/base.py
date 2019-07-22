"""Base classes Param and Processor.

"""
from ..tracker import MetaTracker


class Param(object):
    """Parameters to be tracked in Processors

    """
    def __init__(self, dtype, default=None):
        self.dtype = dtype
        self.default = default


class Processor(object, metaclass=MetaTracker.sub('MetaProcessor', Param, 'params')):
    """Base class of processors of tasks in a pipeline instance

    """
    is_output = Param(bool, None)
    is_checkpoint = Param(bool, False)

    def __init__(self, **kwargs):
        for key, param in self.params.items():
            setattr(self, key, kwargs.get(key, param.default))
        self.checkpoint_data = None

    def function(self, data):
        raise NotImplementedError

    def __call__(self, data):
        out = self.function(data)
        if self.is_checkpoint:
            self.checkpoint_data = out
        return out

    def param_values(self):
        return {key: getattr(self, key) for key in self.params}

    def copy(self):
        new = type(self)(**self.param_values())
        new.checkpoint_data = self.checkpoint_data
        return new

    # TODO: this is not yet clean chaining, we have to find the common base, which is something like the second-to-top
    # class
    # def __add__(self, other):
    #     common_base = type(self)
    #     if not isinstance(other, common_base)
    #         raise TypeError("Processor-chaining: expected instance of type '{}', got '{}'."
    #                         .format(common_base, type(other))
    #                     )
    #     if not isinstance(self, ChainedProcessor):
    #         chained_proc = type("Chained{}".format(common_base),
    #                            [common_base, ChainedProcessor], {})(chain=[self, other], **self.param_values())
    #     else:
    #         chain = self.chain + [other]
    #         chained_proc = type(self)(**self.param_values())
    #     return chained_proc


# TODO: Chained Processors
# class ChainedProcessor(Processor):
#     chain = Param(list, [])
#
#     def function(self, data):
#         for proc in self.chain:
#             data = proc(data)
#         return data


class FunctionProcessor(Processor):
    """Processor instance initialized with a supplied function

    """
    function = Param(callable, (lambda self, data: data))


def ensure_processor(proc, **kwargs):
    """Make sure argument is a Processor and, if it is not, but callable, make it a FunctionProcessor

    """
    if not isinstance(proc, Processor):
        if callable(proc):
            proc = FunctionProcessor(function=proc)
        else:
            raise TypeError('Supplied processor {} is neither a Processor, nor callable!')
    for key, val in kwargs.items():
        if getattr(proc, key, None) is None:
            setattr(proc, key, val)
    return proc