"""Base classes Task and Pipeline.

"""
from collections import OrderedDict

from ..processor.base import ensure_processor, Processor
from ..plugboard import Slot, Plug


class TaskPlug(Plug):
    """Plug to ensure all contained objects are Processors"""
    def __init__(self, slot, obj=None, default=None, **kwargs):
        if default is not None:
            default = ensure_processor(default, **kwargs)
        if obj is not None:
            obj = ensure_processor(obj)
        super().__init__(slot, obj=obj, default=default, **kwargs)

    # pylint: disable=no-member
    @Plug.obj.setter
    def obj(self, value):
        if value is not None:
            value = ensure_processor(value)
        # pylint: disable=no-member
        Plug.obj.fset(self, value)

    # pylint: disable=no-member
    @Plug.default.setter
    def default(self, value):
        if value is not None:
            value = ensure_processor(value)
        # pylint: disable=no-member
        Plug.default.fset(self, value)


class Task(Slot):
    """A single item in a :obj:`Pipeline` task scheme. Tasks are slots that ensure all contained objects in Plugs and
    own default values are Processors.

    Attributes
    ----------
    proc_type : type
        Class of the :obj:`Processor` allowed for this :obj:`Task`.
    default : :obj:`Processor` or :obj:`types.FunctionType` or :obj:`types.MethodType`
        Default :obj:`Processor` to use if no Processor is assigned. If a :obj:`types.FunctionType` or
        :obj:`types.MethodType`, an appropriate :obj:`FunctionProcessor` will be created.
    proc_kwargs : dict
        Keyword arguments to overwrite on the supplied Processor.

    """
    def __init__(self, proc_type=Processor, default=(lambda data: data), **kwargs):
        """Configure :obj:`Task` instance.

        Parameters
        ----------
        proc_type : type
            Class of the :obj:`Processor` allowed for this :obj:`Task`.
        default : :obj:`Processor` or :obj:`types.FunctionType` or :obj:`types.MethodType` Default :obj:`Processor` to
            use if no Processor is assigned. If a :obj:`types.FunctionType` or :obj:`types.MethodType`, an appropriate
            :obj:`FunctionProcessor` will be created.
        **kwargs :
            Keyword arguments to overwrite on the supplied Processor.

        Raises
        ------
        TypeError
            If the default :obj:`Processor` is not of type `proc_type`.

        """
        if not issubclass(proc_type, Processor):
            raise TypeError("Only sub-classes of Processors are allowed!")
        if default is not None:
            default = ensure_processor(default, **kwargs)
        super().__init__(dtype=proc_type, default=default)

    # pylint: disable=no-member
    @Slot.default.setter
    def default(self, value):
        if value is not None:
            value = ensure_processor(value)
        # pylint: disable=no-member
        Task.default.fset(self, value)

    def __call__(self, obj=None, default=None):
        return TaskPlug(self, obj=obj, default=default)


class Pipeline(Processor):
    """Abstract base class for all pipelines using MetaPipeline's tracked Task attributes

    Attributes
    ----------
    task_scheme : :obj:`collections.OrderedDict`
        OrderedDict of Tasks which can be filled with :obj:`Processor`.
    processes : :obj:`collections.OrderedDict`
        OrderedDict of the :obj:`Processor` that filled the `task_scheme`.

    """

    def checkpoint_processes(self):
        """Find the checkpoint :obj:`Processor` closest to output and return an
        :obj:`collections.OrderedDict` of that and all following :obj:`Processor`.

        Returns
        -------
        :obj:`collections.OrderedDict`
            The :obj:`Processor` that is the checkpoint closest to output and all its following
            :obj:`Processor` in an :obj:`OrderedDict`.

        Raises
        ------
        RuntimeError
            If there is not a single checkpoint.

        """
        checkpoint_process_list = []
        for key, proc in reversed(self.collect_attr(Task).items()):
            checkpoint_process_list.append((key, proc))
            if proc.is_checkpoint:
                break
        if checkpoint_process_list and not checkpoint_process_list[-1][1].is_checkpoint:
            raise RuntimeError("No checkpoint defined!")
        checkpoint_processes = OrderedDict(checkpoint_process_list[::-1])
        return checkpoint_processes

    def from_checkpoint(self):
        """Re-evaluate from last checkpointed :obj:`Processor` using its respective output.

        Returns
        -------
        object
            Output of the whole pipeline, starting from checkpointed :obj:`Processor` closest to output.

        Raises
        ------
        RuntimeError
            If the checkpointed :obj:`Processor` closest to output does not have any
            `checkpoint_data` store, i.e. the :obj:`Processor` was never called once after being declared a checkpoint.

        """
        checkpoint_processes = self.checkpoint_processes()
        data_iter = iter(checkpoint_processes.values())
        data = next(data_iter).checkpoint_data
        if data is None:
            raise RuntimeError("No checkpoint data! Run whole pipeline first!")
        for proc in data_iter:
            data = proc(data)
        return data

    def function(self, data):
        """Propagate `data` through the whole pipeline from front to back, calling all Processors in series.

        Attributes
        ----------
        data :
            The pipeline input passed to the first :obj:`Processor` in `self.processors`. Type depends on first
            :obj:`Processor`.

        Returns
        -------
        object
            Output of all :obj:`Processor` that are flagged as pipeline outputs. If no processors
            are flagged as outputs, return the output of the last processor.

        """
        outputs = []
        for proc in self.collect_attr(Task).values():
            data = proc(data)
            if proc.is_output:
                outputs.append(data)
        if not outputs:
            return data
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def __repr__(self):
        """Represent a Pipeline as str.

        Example
        -------
        >>> MyPipeline()
        MyPipeline(
            FunctionProcessor(function=(lambda x: x.mean(1)),) -> output:np.ndarray
            SciPyPDist(metric=sqeuclidean) -> output:np.ndarray
            RadialBasisFunction(sigma=0.1) -> output:np.ndarray
            MyProcess(stuff=3, func=Param(FunctionType, lambda x: x**2)) -> output:np.ndarray
        )
        """
        pipeline = '\n    '.join([proc.__repr__() for proc in self.collect_attr(Task).values()])
        return f'{self.__class__.__name__}(\n    {pipeline}\n)'
