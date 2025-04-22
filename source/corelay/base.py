"""A module that contains basic CoRelAy classes, such as ``Param``."""

from typing import Any

from corelay.plugboard import Slot


class Param(Slot):
    """A single parameter, whose instances are tracked by a ``MetaTracker``."""

    def __init__(
        self,
        dtype: type | tuple[type, ...],
        default: Any = None,
        mandatory: bool = False,
        positional: bool = False,
        identifier: bool = False
    ) -> None:
        """Initializes a new ``Param`` instance, and configures its type and default value of the parameter.

        Args:
            dtype (type | tuple[type, ...]): The allowed type(s) of the parameter. This can be a single type or a tuple of types.
            default (Any): The default value of the parameter. This must be an instance of one of the types specified in ``dtype``.
            mandatory (bool): A value indicating whether this parameter is mandatory. If `True`, the default value will be removed.
            positional (bool): A value indicating whether this parameter will have to be passed as a positional argument to ``Processor.__init__``. If
                `True`, the parameter will be passed as a positional argument to ``Processor.__init__``, otherwise it will be passed as a keyword
                argument.
            identifier (bool): A value indicating whether this parameter should be used to identify a ``Processor``. If `True`, the parameter will be
                used to identify a ``Processor``. This is useful for distinguishing processors, when caching their outputs.
        """

        super().__init__(dtype, default)

        if mandatory:
            del self.default

        self._positional = positional
        self._identifier = identifier

    @property
    def is_positional(self) -> bool:
        """Gets or sets a value indicating whether this parameter can be assigned as a positional argument to ``Processor.__init__``.

        Returns:
            bool: Returns `True` if this parameter can be assigned as a positional argument to ``Processor.__init__`` and `False` otherwise.
        """

        return self._positional

    @property
    def is_identifier(self) -> bool:
        """Gets or sets a value indicating whether this parameter should be used to identify a ``Processor``.

        Returns:
            bool: Returns `True` if this parameter should be used to identify a ``Processor`` and `False` otherwise.
        """

        return self._identifier
