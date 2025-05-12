"""A module that contains core CoRelAy classes, such as :py:class:`~corelay.base.Param`."""

import typing

from corelay.plugboard import Slot
from corelay.utils import get_fully_qualified_name


class Param(Slot):
    """A single parameter slot, which can be used to parameterize implementations of :py:class:`~corelay.processor.base.Processor`. The instances of
    :py:class:`Param` in a :py:class:`~corelay.processor.base.Processor` implementation are tracked by a :py:class:`~corelay.tracker.Tracker`.
    """

    def __init__(
        self,
        dtype: type[typing.Any] | tuple[type[typing.Any], ...],
        default: typing.Any = None,
        mandatory: bool = False,
        positional: bool = False,
        identifier: bool = False
    ) -> None:
        """Initializes a new :py:class:`Param` instance, and configures its type and default value of the parameter.

        Args:
            dtype (type[typing.Any] | tuple[type[typing.Any], ...]): The allowed type(s) of the parameter. This can be a single :py:class:`type` or a
                :py:class:`tuple` of multiple :py:class:`type` instances.
            default (typing.Any): The default value of the parameter. This must be an instance of one of the types specified in the ``dtype``
                parameter.
            mandatory (bool): A value indicating whether this parameter is mandatory. If :py:obj:`True`, the default value will be removed. Defaults
                to :py:obj:`False`.
            positional (bool): A value indicating whether this parameter will have to be passed as a positional argument to
                :py:meth:`Processor.__init__ <corelay.processor.base.Processor.__init__>`. If :py:obj:`True`, the parameter will be passed as a
                positional argument to :py:meth:`Processor.__init__ <corelay.processor.base.Processor.__init__>`, otherwise it will be passed as a
                keyword argument. Defaults to :py:obj:`False`.
            identifier (bool): A value indicating whether this parameter should be used to identify a
                :py:class:`~corelay.processor.base.Processor`. If :py:obj:`True`, the parameter will be used to identify a
                :py:class:`~corelay.processor.base.Processor`. This is useful for distinguishing processors, when caching their outputs. Defaults to
                :py:obj:`False`.
        """

        super().__init__(dtype, default)

        if mandatory:
            del self.default

        self._mandatory = mandatory
        self._positional = positional
        self._identifier = identifier

    def __repr__(self) -> str:
        """Returns a :py:class:`str` representation of the :py:class:`Param` instance.

        Returns:
            str: Returns a :py:class:`str` representation of the :py:class:`Param` instance.
        """

        # Sphinx AutoDoc uses __repr__ when it encounters the metadata of typing.Annotated; this is a reasonable thing to do, but then it tries to
        # resolve the resulting string as types for cross-referencing, which is not possible with the default implementation of __repr__; to be able
        # to get proper documentation, the fully-qualified name of the class is returned, because this enable Sphinx AutoDoc to reference the class in
        # the documentation
        return f'~{get_fully_qualified_name(self)}'

    @property
    def is_positional(self) -> bool:
        """Gets or sets a value indicating whether this parameter can be assigned as a positional argument to
        :py:meth:`Processor.__init__ <corelay.processor.base.Processor.__init__>`.

        Returns:
            bool: Returns :py:obj:`True` if this parameter can be assigned as a positional argument to
            :py:meth:`Processor.__init__ <corelay.processor.base.Processor.__init__>` and :py:obj:`False` otherwise.
        """

        return self._positional

    @property
    def is_identifier(self) -> bool:
        """Gets or sets a value indicating whether this parameter should be used to identify a :py:class:`~corelay.processor.base.Processor`.

        Returns:
            bool: Returns :py:obj:`True` if this parameter should be used to identify a :py:class:`~corelay.processor.base.Processor`
            and :py:obj:`False` otherwise.
        """

        return self._identifier
