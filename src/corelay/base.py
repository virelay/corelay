"""Module containing basic CoRelAy classes, such as Param"""
from .plugboard import Slot


class Param(Slot):
    """A single parameter, which instances are to be tracked by a `MetaTracker`.

    Attributes
    ----------
    dtype : type or tuple of type
        Allowed type(s) of the parameter.
    default : :obj:`dtype`
        Default parameter value, should be an instance of (one of) :obj:`dtype`.
    positional : bool
        Optional, whether the Parameter can also be set as a positional Parameter.
    identifier : bool
        Optional, whether the Parameter should be used to identify a Processor.

    """
    def __init__(self, dtype, default=None, mandatory=False, positional=False, identifier=False):
        """Configure type and default value of parameter.

        Parameters
        ----------
        dtype : type or tuple of type
            Allowed type(s) of the parameter.
        default : :obj:`dtype`
            Default parameter value, should be an instance of (one of) :obj:`dtype`.

        """
        super().__init__(dtype, default)
        if mandatory:
            del self.default
        self._positional = positional
        self._identifier = identifier

        # allowed_dtypes = (type, FunctionType, BuiltinFunctionType)
        # if not all(isinstance(x, allowed_dtypes) for x in self.dtype):
        #     raise TypeError(
        #         "Following dtypes: {} are not in the allowed types {}.".format(self.dtype, allowed_dtypes)
        #     )

    @property
    def is_positional(self):
        """Whether this param can be assigned as a positional argument to Processor.__init__"""
        return self._positional

    @property
    def is_identifier(self):
        """Whether this param should be used to identify a Processor"""
        return self._identifier
