"""Base classes Task and Pipeline.

"""
from collections import OrderedDict

from ..processor.base import ensure_processor, Processor
from ..plugboard import Slot, Plug, Plugboard


class TaskPlug(Plug):
    """Plug to ensure all contained objects are Processors"""
    def __init__(self, slot, obj=None, default=None, **kwargs):
        if default is not None:
            default = ensure_processor(default, **kwargs)
        if obj is not None:
            obj = ensure_processor(obj)
        super().__init__(slot, obj=obj, default=default, **kwargs)

    @Plug.obj.setter
    def obj(self, value):
        if value is not None:
            value = ensure_processor(value)
        Plug.obj.fset(self, value)

    @Plug.default.setter
    def default(self, value):
        if value is not None:
            value = ensure_processor(value)
        Plug.default.fset(self, value)


class Task(Slot):
    """A single item in a :obj:`Pipeline` task scheme. Tasks are slots that ensure all contained objects in Plugs and
    own default values are Processors.

    Attributes
    ----------
    proc_type : type
        Class of the :obj:`Processor`s allowed for this :obj:`Task`.
    default : :obj:`Processor` or :obj:`types.FunctionType` or :obj:`types.MethodType` Default :obj:`Processor` to use
        if no Processor is assigned. If a :obj:`types.FunctionType` or :obj:`types.MethodType`, an appropriate
        :obj:`FunctionProcessor` will be created.
    proc_kwargs : dict
        Keyword arguments to overwrite on the supplied Processor.

    """
    def __init__(self, proc_type=Processor, default=(lambda data: data), **kwargs):
        """Configure :obj:`Task` instance.

        Parameters
        ----------
        proc_type : type
            Class of the :obj:`Processor`s allowed for this :obj:`Task`.
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

    @Slot.default.setter
    def default(self, value):
        if value is not None:
            value = ensure_processor(value)
        Task.default.fset(self, value)

    def __call__(self, obj=None, default=None):
        return TaskPlug(self, obj=obj, default=default)


class Pipeline(Processor):
    """Abstract base class for all pipelines using MetaPipeline's tracked Task attributes

    Attributes
    ----------
    task_scheme : :obj:`collections.OrderedDict`
        OrderedDict of Tasks which can be filled with :obj:`Processor`s.
    processes : :obj:`collections.OrderedDict`
        OrderedDict of the :obj:`Processor`s that filled the `task_scheme`.

    """
    def __init__(self, **kwargs):
        """Configure :obj:`Pipeline` instance

        Parameters
        ----------
        **kwargs :
            Keyword arguments with keys named identical to the pipeline tasks, and values of type :obj:`Processor` to
            fill these :obj:`Task`s.

        Raises
        ------
        TypeError
            If some :obj:`Processor` does not have a matching type of the :obj:`Task` it has be assigned to.

        """
        super().__init__(**kwargs)

    def checkpoint_processes(self):
        """Find the checkpoint :obj:`Processor` closest to output in `self.processes` and return an
        :obj:`collections.OrderedDict` of that and all following :obj:`Processor`s in `self.processes`.

        Returns
        -------
        :obj:`collections.OrderedDict`
            The :obj:`Processor` that is the checkpoint closest to output in `self.processes` and all its following
            :obj:`Processor`s in an :obj:`OrderedDict`.

        Raises
        ------
        RuntimeError
            If there is not a single checkpoint in `self.processes`

        """
        checkpoint_process_list = []
        for key, proc in reversed(self.collect_attr(Task).items()):
            checkpoint_process_list.append((key, proc))
            if proc.is_checkpoint:
                break
        if not proc.is_checkpoint:
            raise RuntimeError("No checkpoint defined!")
        checkpoint_processes = OrderedDict(checkpoint_process_list[::-1])
        return checkpoint_processes

    def from_checkpoint(self):
        """Re-evaluate from last checkpointed :obj:`Processor` using its respective output.

        Returns
        -------
        object
            Output of the whole pipeline, starting from checkpointed :obj:`Processor` closest to output in
            `self.processes`.

        Raises
        ------
        RuntimeError
            If the checkpointed :obj:`Processor` closest to output in `self.processes` does not have any
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
        """Propagate `data` through the whole pipeline from front to back, calling all `self.processes` in series.

        Attributes
        ----------
        data :
            The pipeline input passed to the first :obj:`Processor` in `self.processors`. Type depends on first
            :obj:`Processor`.

        Returns
        -------
        object
            Output of all :obj:`Processor`s in `self.processes` that are flagged as pipeline outputs. If no processors
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
        """Example of the pipeline representation:
        MyPipeline(
            FunctionProcessor(function=(lambda x: x.mean(1)),) -> output:np.ndarray
            SciPyPDist(metric=sqeuclidean) -> output:np.ndarray
            RadialBasisFunction(sigma=0.1) -> output:np.ndarray
            MyProcess(stuff=3, func=Param(FunctionType, lambda x: x**2)) -> output:np.ndarray
        )
        """
        pipeline = '\n    '.join([proc.__repr__() for proc in self.processes.values()])
        return '{}(\n    {}\n)'.format(self.__class__.__name__, pipeline)
