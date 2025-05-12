"""A module that contains the :py:class:`~corelay.plugboard.Plugboard` class. Plugboards manage instances of :py:class:`~corelay.plugboard.Slot`,
which describe values. A :py:class:`~corelay.plugboard.Slot` can be filled using a :py:class:`~corelay.plugboard.Plug`, which represents a concrete
value.
"""

import typing
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, LambdaType, MethodType

import numpy

from corelay.tracker import Tracker
from corelay.utils import get_fully_qualified_name


class EmptyInit:
    """:py:class:`EmptyInit` is a class intended to be inherited as a last step down the MRO, to catch any remaining positional and/or keyword
    arguments and thus raise proper exceptions.
    """

    def __init__(self) -> None:
        """Initializes a new instance of :py:class:`EmptyInit`.

        Note:
            This is not intended to be called directly, but rather by the constructor of the next class up in the inheritance hierarchy. The
            super-delegation is not unnecessary, even if PyLint claims so, since this is causes Python to raise a more informative exception when the
            user tries to pass more keyword arguments than are accepted by the constructors in the inheritance hierarchy. If the constructor of the
            penultimate class in inheritance hierarchy calls this constructor and there are still keyword arguments left, Python will raise a
            :py:class:`TypeError` with a message like ":py:class:`TypeError`: `object.__init__()` takes exactly one argument (the instance to
            initialize)". This may confuse users, but if the constructor call is delegated to the next class up (i.e., :py:class:`object`), the
            exception will be more informative and say something like: ":py:class:`TypeError`: `Empty.__init__()` takes 0 positional arguments but 1
            was given".
        """

        # pylint: disable=useless-super-delegation
        super().__init__()


class Slot(EmptyInit):
    """Slots are descriptors that contain objects in a container called :py:class:`Plug`. Instances of the :py:class:`Slot` class have a
    :py:attr:`~Slot.dtype` and a :py:attr:`~Slot.default` value, which are enforced to be consistent. When a :py:class:`Slot` instance is accessed in
    a class, it will return the contained object of its :py:class:`Plug` container. When accessing or assigning :py:class:`Slot` instances in a class
    that have never been accessed before, a :py:class:`Plug` object is stored in the class' ``__dict__`` under the same name the :py:class:`Slot` was
    assigned to in the class. Slots may have their :py:attr:`~Slot.default` value set to :py:obj:`None`, in which case setting plugs belonging to it
    must have either a :py:attr:`Plug.default` value, or an explicit :py:attr:`Plug.obj` value on their own. Calling a :py:class:`Slot` instance
    creates a corresponding :py:class:`Plug` container instance.

    Note:
        See `https://docs.python.org/3/howto/descriptor.html` for more information on descriptors.

    See Also:
        * :py:class:`Plugboard`
        * :py:class:`Plug`
    """

    def __init__(
        self,
        dtype: type | tuple[type, ...] = object,
        default: typing.Any = None,
        **kwargs: typing.Any
    ) -> None:
        """Initializes a new :py:class:`Slot` instance. Configures that data type and the default value of the slot. A consistency check is performed
        to ensure that the default value is of the correct type.

        Args:
            dtype (type | tuple[type, ...]): The data type of the slot. This can be a single type or a tuple of types. The value of the plug, as well
                as the default value, must be of this type or one of the types in the tuple. Defaults to :py:class:`object`.
            default (typing.Any): The default value of the plug. The default value must be an instance of the specified data type
                :py:attr:`~Slot.dtype` or one of the types in the tuple. If no default value is set, it will be :py:obj:`None`. When a plug is created
                without an explicit value, it will use this default value. Defaults to :py:obj:`None`.
            **kwargs (typing.Any): Additional keyword arguments that are passed to the parent class constructor. This is done for cooperativity's
                sake, as the next class one step up in the inheritance hierarchy will be `EmptyInit`, which does not accept any additional keyword
                arguments and will raise an exception if any are passed.
        """

        super().__init__(**kwargs)

        self._dtype: tuple[type, ...] = dtype if isinstance(dtype, tuple) else (dtype,)
        self._default = default
        self.__name__ = ''

        self._consistent()

    _function_types = (
        LambdaType,
        MethodType,
        BuiltinFunctionType,
        BuiltinMethodType,
        numpy.ufunc,
        type(numpy.max)
    )
    """Contains all types that may represent a function. This is necessary, since many functions are not of type :py:class:`~types.FunctionType`,
    e.g., lambda expressions, methods, built-in functions, built-in methods, and NumPy universal functions. Also, since NumPy 1.26, NumPy array
    functions are no longer actual functions; so, for example, something like
    `pooling_function: Annotated[FunctionType, Param(FunctionType, numpy.sum)]` would not work anymore. When the user sets the :py:attr:`~Slot.dtype`
    to :py:class:`~types.FunctionType` or :py:class:`~types.FunctionType` is an element of the tuple, then the types in this tuple are added to the
    data types that the consistency check checks against. This way, the user does not have to worry about the fact that many functions are not of type
    :py:class:`~types.FunctionType`.
    """

    def _consistent(self) -> None:
        """Checks whether :py:attr:`~Slot.dtype` and :py:attr:`~Slot.default` are consistent, i.e., :py:attr:`~Slot.default` is either :py:obj:`None`,
        of type :py:attr:`~Slot.dtype`, or one of the types in the tuple :py:attr:`~Slot.dtype`.

        Raises:
            TypeError: The :py:attr:`~Slot.default` value is not consistent with the :py:attr:`~Slot.dtype`, i.e., it is neither :py:obj:`None`, nor
                of the type :py:attr:`~Slot.dtype` or one of the types in the tuple :py:attr:`~Slot.dtype`.
        """

        if (
            not isinstance(self.dtype, type) and
            not (
                isinstance(self.dtype, tuple) and
                all(isinstance(element, type) for element in self.dtype)
            )
        ):
            raise TypeError(
                f'The data type of the "{type(self).__name__}" object "{self.__name__}" is neither a type, nor a tuple of types, but  "{self.dtype}".'
            )

        # If the user sets the dtype to FunctionType, then we should add some more types to check against, because many functions are not of type
        # FunctionType, e.g., expressions, methods, built-in functions, built-in methods, NumPy universal functions, and NumPy array functions
        effective_dtypes = self.dtype if isinstance(self.dtype, tuple) else (self.dtype,)
        if FunctionType in effective_dtypes:
            effective_dtypes = effective_dtypes + Slot._function_types

        if self.default is not None and not isinstance(self.default, effective_dtypes):
            raise TypeError(
                f'The data type of the default value of the "{type(self).__name__}" object "{self.__name__}" is not of type "{self.dtype}".'
            )

    def get_plug(self, instance: typing.Any, obj: typing.Any = None, default: typing.Any = None) -> 'Plug':
        """Gets a corresponding :py:class:`Plug` that can be used to access the ``__dict__`` of the :py:class:`Slot` instance . In case a new
        :py:class:`Plug` has to be created, the ``obj`` and ``default`` parameters may be specified.

        Args:
            instance (typing.Any): An instance of the class the :py:class:`Slot` was assigned in.
            obj (typing.Any): The object value to write to newly created :py:class:`Plug` instances.
            default (typing.Any): The default value to write to newly created :py:class:`Plug` instances.

        Returns:
            Plug: Returns a :py:class:`Plug` container. If a :py:class:`Plug` instance already exists in the instance's ``__dict__`` it is returned.
            Otherwise, a new :py:class:`Plug` container is created, which is also appended to the instance's ``__dict__``. If a new :py:class:`Plug`
            instance is created, the :py:attr:`~Plug.obj` and :py:attr:`~Plug.default` values are set to the values passed in.
        """

        try:
            plug: Plug = instance.__dict__[self.__name__]
        except KeyError:
            plug = self(obj=obj, default=default)
            instance.__dict__[self.__name__] = plug
        return plug

    def __set_name__(self, owner: type, name: str) -> None:  # pylint: disable=unused-argument
        """Is invoked, when the :py:class:`Slot` is assigned to a class or instance attribute. Sets the name of the slot when assigned under a class.
        Necessary to write the correct ``__dict__`` entry in the parent class.

        Args:
            owner (type): The parent class the :py:class:`Slot` was assigned in.
            name (str): The name under which the :py:class:`Slot` was assigned in the parent class.
        """

        self.__name__ = name

    def __get__(self, instance: typing.Any, owner: type) -> 'Slot | typing.Any':  # pylint: disable=unused-argument
        """Is invoked, when the class or instance attribute the :py:class:`Slot` was assigned to is read. When the :py:class:`Slot` is accessed from a
        class, the :py:class:`Slot` instance itself is returned. If accessed using an instance, the corresponding :py:class:`Plug` container's value
        is returned.

        Args:
            instance (typing.Any): The instance of the parent class the :py:class:`Slot` was assigned in.
            owner (type):  The parent class the :py:class:`Slot` was defined in.

        Returns:
            Slot | typing.Any: Returns the value of the Plug container.
        """

        return self if instance is None else self.get_plug(instance).obj

    def __set__(self, instance: typing.Any, value: typing.Any) -> None:
        """Is invoked, when the class or instance attribute the :py:class:`Slot` was assigned to is written. Sets the instance's :py:class:`Plug`
        container object value.

        Args:
            instance (typing.Any): The instance of the parent class the :py:class:`Slot` was assigned in.
            value (typing.Any): The value to set the Plug container's object value to.
        """

        self.get_plug(instance, obj=value).obj = value

    def __delete__(self, instance: typing.Any) -> None:
        """Is invoked, when the class or instance attribute the :py:class:`Slot` was assigned to is deleted. Deletes the instance's :py:class:`Plug`
        container object value if it exists, enforcing the use of its default value.

        Args:
            instance (typing.Any): The instance of the parent class the :py:class:`Slot` was assigned in.
        """

        del self.get_plug(instance).obj

    def __repr__(self) -> str:
        """Returns a :py:class:`str` representation of the :py:class:`Slot` instance.

        Returns:
            str: Returns a :py:class:`str` representation of the :py:class:`Slot` instance.
        """

        # Sphinx AutoDoc uses __repr__ when it encounters the metadata of typing.Annotated; this is a reasonable thing to do, but then it tries to
        # resolve the resulting string as types for cross-referencing, which is not possible with the default implementation of __repr__; to be able
        # to get proper documentation, the fully-qualified name of the class is returned, because this enable Sphinx AutoDoc to reference the class in
        # the documentation
        return f'~{get_fully_qualified_name(self)}'

    @property
    def default(self) -> typing.Any:
        """Gets or sets the default value of the :py:class:`Slot`.

        Returns:
            typing.Any: Returns the slot's default value. If not set, :py:obj:`None` is returned.
        """

        return self._default

    @default.setter
    def default(self, value: typing.Any) -> None:
        """Gets or sets the default value of the :py:class:`Slot`. Checks the new default value for consistency with the :py:attr:`~Slot.dtype`.

        Args:
            value (typing.Any): The new default value to set. If not set, :py:obj:`None` is returned.

        Raises:
            TypeError: The default value is not consistent with the :py:attr:`~Slot.dtype`, i.e., it is neither :py:obj:`None`, nor of the type
                :py:attr:`~Slot.dtype` or one of the types in the tuple :py:attr:`~Slot.dtype`.
        """

        original_default_value = self._default
        self._default = value

        try:
            self._consistent()
        except TypeError as exception:
            self._default = original_default_value
            raise TypeError(
                f'The data type of the default value of the "{type(self).__name__}" object "{self.__name__}" is not of type "{self.dtype}".'
            ) from exception

    @default.deleter
    def default(self) -> None:
        """Deletes the default value of the :py:class:`Slot`."""

        self._default = None

    @property
    def dtype(self) -> type | tuple[type, ...]:
        """Gets or sets the slot's :py:attr:`~Slot.dtype`.

        Returns:
            type | tuple[type, ...]: Returns the slot's :py:attr:`~Slot.dtype`. If not set, :py:obj:`None` is returned.
        """

        return self._dtype

    @dtype.setter
    def dtype(self, value: type | tuple[type, ...]) -> None:
        """Gets or sets the slot's :py:attr:`~Slot.dtype`. Checks the default value of the slot for consistency with the new :py:attr:`~Slot.dtype`.

        Args:
            value (type | tuple[type, ...]): The new :py:attr:`~Slot.dtype` to set. If not set, :py:obj:`None` is returned.

        Raises:
            TypeError: The default value is not consistent with the :py:attr:`~Slot.dtype`, i.e., it is neither :py:obj:`None`, nor of the type
                :py:attr:`~Slot.dtype` or one of the types in the tuple :py:attr:`~Slot.dtype`.
        """

        original_dtype_value = self._dtype
        self._dtype = value if isinstance(value, tuple) else (value,)

        try:
            self._consistent()
        except TypeError as exception:
            self._dtype = original_dtype_value
            raise TypeError(
                f'The data type of the dtype value of the "{type(self).__name__}" object "{self.__name__}" is not of type "{self.dtype}".'
            ) from exception

    @property
    def optional(self) -> bool:
        """Gets a value indicating whether the :py:class:`Slot` is optional.

        Returns:
            bool: Returns :py:obj:`True` if the :py:class:`Slot` is optional, i.e., it has a default value, and :py:obj:`False` otherwise.
        """

        return self.default is not None

    def __call__(self, obj: typing.Any = None, default: typing.Any = None) -> 'Plug':
        """Create a new corresponding Plug container

        Args:
            obj (typing.Any): A value to initialize the newly created :py:class:`Plug` container's object value to. Defaults to :py:obj:`None`.
            default (typing.Any): A value to initialize the newly created :py:class:`Plug` container's default value to. Defaults to :py:obj:`None`.

        Returns:
            Plug: Returns a newly created :py:class:`Plug` container instance, obeying the type and optionality constraints.
        """

        return Plug(self, obj=obj, default=default)


class Plug(EmptyInit):
    """Container class to fill slots associated with a certain instance. The instance is usually of type :py:class:`Plugboard`, but may be of any kind
    of type.

    See Also:
        * :py:class:`Slot`
        * :py:class:`Plugboard`
    """

    def __init__(self, slot: Slot, obj: typing.Any = None, default: typing.Any = None, **kwargs: typing.Any) -> None:
        """Initializes a new :py:class:`Plug` instance and checks for consistency.

        Args:
            slot (Slot): The :py:class:`Slot` instance to associate with this :py:class:`Plug`.
            obj (typing.Any): An explicitly defined object held in the :py:class:`Plug` container. If not set, :py:attr:`~Plug.default` is returned as
                its value.
            default (typing.Any): A plug-dependent lower-priority object held in the :py:class:`Plug` container. If not set, :py:attr:`~Plug.fallback`
                is returned.
            **kwargs (typing.Any): Keyword arguments passed down to the base class constructor, for cooperativity's sake. In normal cases, this next
                class will be :py:class:`EmptyInit`, which accepts no more keyword arguments and will raise an exception.
        """

        super().__init__(**kwargs)

        self._obj = obj
        self._slot = slot
        self._default = default

        self._consistent()

    _function_types = (
        LambdaType,
        MethodType,
        BuiltinFunctionType,
        BuiltinMethodType,
        numpy.ufunc,
        type(numpy.max)
    )
    """Contains all types that may represent a function. This is necessary, since many functions are not of type :py:class:`~types.FunctionType`,
    e.g., lambda expressions, methods, built-in functions, built-in methods, and NumPy universal functions. Also, since NumPy 1.26, NumPy array
    functions are no longer actual functions; so, for example, something like
    `pooling_function: Annotated[FunctionType, Param(FunctionType, numpy.sum)]` would not work anymore. When the user sets the :py:attr:`~Slot.dtype`
    to :py:class:`~types.FunctionType` or :py:class:`~types.FunctionType` is an element of the tuple, then the types in this tuple are added to the
    data types that the consistency check checks against. This way, the user does not have to worry about the fact that many functions are not of type
    :py:class:`~types.FunctionType`.
    """

    def _consistent(self) -> None:
        """Checks whether all values are consistent, i.e., at least one of :py:attr:`~Plug.obj`, :py:attr:`~Plug.default`, or
        :py:attr:`~Plug.fallback` is set and of the data type specified in `slot.dtype`, or one of the types in the tuple `slot.dtype`.

        Raises:
            TypeError: None of :py:attr:`~Plug.obj`, :py:attr:`~Plug.default`, or :py:attr:`~Plug.fallback` is set, or the value is not consistent
                with the `slot.dtype` or one of the types in the tuple `slot.dtype`.
        """

        if self.obj is None:
            raise TypeError(f'"{type(self.slot).__name__}" object "{self.slot.__name__}" is mandatory, yet it has been accessed without being set.')

        # If the user sets the dtype to FunctionType, then we should add some more types to check against, because many functions are not of type
        # FunctionType, e.g., expressions, methods, built-in functions, built-in methods, NumPy universal functions, and NumPy array functions
        effective_dtypes = self.slot.dtype if isinstance(self.slot.dtype, tuple) else (self.slot.dtype,)
        if FunctionType in effective_dtypes:
            effective_dtypes = effective_dtypes + Plug._function_types

        if not isinstance(self.obj, effective_dtypes):
            raise TypeError(f'"{type(self.slot).__name__}" object "{self.slot.__name__}" value "{self.obj}" is not of type "{self.slot.dtype}".')

    @property
    def slot(self) -> Slot:
        """Gets or sets the associated :py:class:`Slot`.

        Returns:
            Slot: Returns the associated :py:class:`Slot`. If not set, :py:obj:`None` is returned.
        """

        return self._slot

    @slot.setter
    def slot(self, value: Slot) -> None:
        """Gets or sets associated :py:class:`Slot` and checks for consistency.

        Args:
            value (Slot): The new :py:class:`Slot` to set.
        """

        self._slot = value
        self._consistent()

    @property
    def dtype(self) -> type | tuple[type, ...]:
        """Gets the :py:attr:`Slot.dtype` of the associated :py:class:`Slot`. The :py:attr:`~Plug.dtype` property is non-mutable.

        Returns:
            type | tuple[type, ...]: Returns the :py:attr:`Slot.dtype` of the associated :py:class:`Slot`.
        """

        return self.slot.dtype[0] if isinstance(self.slot.dtype, tuple) and len(self.slot.dtype) == 1 else self.slot.dtype

    @property
    def optional(self) -> bool:
        """Gets a value indicating whether the :py:class:`Plug` container has a default value. The :py:attr:`Plug.optional` property is non-mutable.

        Returns:
            bool: Returns :py:obj:`True` if the :py:class:`Plug` container has a default value, and :py:obj:`False` otherwise.
        """

        return self.default is not None

    @property
    def fallback(self) -> typing.Any:
        """Gets the default value of the associated :py:class:`Slot`. The :py:attr:`~Plug.fallback` property is non-mutable.

        Returns:
            typing.Any: Returns the default value of the associated :py:class:`Slot`.
        """

        return self.slot.default

    @property
    def obj(self) -> typing.Any:
        """Gets or sets the value of the object contained in the :py:class:`Plug`. If the :py:class:`Plug` does not contain an object value,
        :py:attr:`~Plug.default` is retrieved instead.

        Returns:
            typing.Any: Returns the object value contained in the :py:class:`Plug`. If not set, :py:attr:`~Plug.default` is returned.
        """

        if self._obj is None:
            return self.default
        return self._obj

    @obj.setter
    def obj(self, value: typing.Any) -> None:
        """Gets or sets the value of the object contained in the :py:class:`Plug` and checks for consistency.

        Args:
            value (typing.Any): The new object value to set.

        Raises:
            TypeError: The object value is not consistent with the :py:attr:`~Plug.dtype`, i.e., it is neither :py:obj:`None`, nor of the type
                :py:attr:`~Plug.dtype` or one of the types in the tuple :py:attr:`~Plug.dtype`.
        """

        original_obj_value = self._obj
        self._obj = value

        try:
            self._consistent()
        except TypeError as exception:
            self._obj = original_obj_value
            raise TypeError(f'The data type of the obj value of the "{type(self).__name__}" object is not of type "{self.dtype}".') from exception

    @obj.deleter
    def obj(self) -> None:
        """Deletes the value of the object contained in the :py:class:`Plug` by setting it to :py:obj:`None`."""

        self.obj = None

    @property
    def default(self) -> typing.Any:
        """Gets or sets the default value of the :py:class:`Plug`. If the :py:attr:`~Plug.default` value is not set, then the
        :py:attr:`~Plug.fallback` value is retrieved instead.

        Returns:
            typing.Any: Returns the default value of the :py:class:`Plug`. If not set, :py:attr:`~Plug.fallback` is returned.
        """

        if self._default is None:
            return self.fallback
        return self._default

    @default.setter
    def default(self, value: typing.Any) -> None:
        """Gets or sets the default value of the :py:class:`Plug` and checks for consistency.

        Args:
            value (typing.Any): The new default value to set.

        Raises:
            TypeError: The default value is not consistent with the :py:attr:`~Plug.dtype`, i.e., it is neither :py:obj:`None`, nor of the type
                :py:attr:`~Plug.dtype` or one of the types in the tuple :py:attr:`~Plug.dtype`.
        """

        original_default_value = self._default
        self._default = value

        try:
            self._consistent()
        except TypeError as exception:
            self._default = original_default_value
            raise TypeError(f'The data type of the default value of the "{type(self).__name__}" object is not of type "{self.dtype}".') from exception

    @default.deleter
    def default(self) -> None:
        """Deletes the default value of the :py:class:`Plug` by setting it to :py:obj:`None`."""

        self.default = None


class SlotDefaultAccess:
    """A proxy-object descriptor class to access the default values of the owning class of a :py:class:`Slot`, since :py:class:`Slot` instances cannot
    be returned except by accessing a classes' ``__dict__``.

    See Also:
        * :py:class:`Slot`
        * :py:class:`Plugboard`
        * :py:class:`Plug`
    """

    def __init__(self, instance: Tracker | typing.Any = None) -> None:
        """Initializes a new :py:class:`SlotDefaultAccess` instance.

        Args:
            instance (Tracker | typing.Any): The instance of the class the :py:class:`SlotDefaultAccess` is associated with.
        """

        # We cannot just assign self._instance here, because this would cause __get__ to be called, which in turn would create a new instance of
        # SlotDefaultAccess, which would call __init__ again, which would call __get__ again, and so on, resulting in an infinite recursion; this is
        # circumvented by using object.__setattr__ instead, which does not call __get__
        self._instance: Tracker | typing.Any
        object.__setattr__(self, '_instance', instance)

    def _get_plug(self, name: str, default: typing.Any = None) -> Plug:
        """Gets the :py:class:`Plug` of the instance of the associated :py:class:`Slot`-owning class by name, by calling the :py:meth:`~Slot.get_plug`
        method of the :py:class:`Slot`.

        Args:
            name (str):  The name of the :py:class:`Slot`.
            default (typing.Any): The default value to set if the :py:class:`Plug` associated with the :py:class:`Slot` does not exist yet.

        Raises:
            AttributeError: There is no attribute in the associated owner class of this name of type :py:class:`Slot`.

        Returns:
            Plug: Returns the :py:class:`Plug` container associated with the instance of the :py:class:`Slot`-owning class and name.
        """

        if self._instance is None:
            raise AttributeError('The instance of the class the SlotDefaultAccess is associated with is not set.')
        slot = getattr(type(self._instance), name)
        if not isinstance(slot, Slot):
            raise AttributeError(f'"{type(self._instance)}" object has no attribute "{name}" of type "{Slot}", it is of type "{type(slot)}".')
        return slot.get_plug(self._instance, default=default)

    def __get__(self, instance: typing.Any, owner: typing.Any) -> 'SlotDefaultAccess':  # pylint: disable=unused-argument
        """Gets a new instance of :py:class:`SlotDefaultAccess`, initialized with the provided instance value.

        Args:
            instance (typing.Any): The instance of the class the :py:class:`SlotDefaultAccess` is associated with.
            owner (typing.Any): The owner class of the :py:class:`SlotDefaultAccess`.

        Returns:
            SlotDefaultAccess: Returns a new instance of :py:class:`SlotDefaultAccess` initialized with the provided instance value.
        """

        return type(self)(instance)

    def __set__(self, instance: typing.Any, value: dict[str, typing.Any]) -> None:
        """Sets the default values of the associated owner class instance's slots by assigning a the values of the :py:class:`dict` specified in
        ``value``.

        Args:
            instance (typing.Any): The instance of the class the :py:class:`SlotDefaultAccess` is associated with.
            value (dict[str, typing.Any]): A :py:class:`dict` containing the default values to set for the associated owner class instance's slots.

        Raises:
            TypeError: The ``value`` is not a :py:class:`dict`.
        """

        if not isinstance(value, dict):
            raise TypeError('Can only directly set default values using a dict.')

        slot_default_accessor = type(self)(instance)
        for attribute_name, attribute_default_value in value.items():
            setattr(slot_default_accessor, attribute_name, attribute_default_value)

    def __getattr__(self, name: str) -> typing.Any:
        """Gets the default value of the of the slot with the specified ``name`` that associated with the owner class instance.

        Args:
            name (str): The name of the slot to get the default value for.

        Raises:
            AttributeError: There is no slot with the specified ``name`` in the associated owner class instance.

        Returns:
            typing.Any: Returns the default value of the slot with the specified ``name`` that associated with the owner class instance.
        """

        try:
            return self._get_plug(name).default
        except AttributeError as exception:
            raise AttributeError(f'"{type(self._instance)}" object has no attribute "{name}" of type "{Slot}".') from exception

    def __setattr__(self, name: str, value: typing.Any) -> None:
        """Sets the default value of the slot with the specified ``name`` that associated with the owner class instance.

        Args:
            name (str): The name of the slot to set the default value for.
            value (typing.Any): The new default value to set.

        Raises:
            AttributeError: There is no slot with the specified ``name`` in the associated owner class instance.
            TypeError: The default value is not consistent with the :py:attr:`~Plug.dtype` of the :py:class:`Plug`, i.e., it is neither
                :py:obj:`None`, nor of the type :py:attr:`~Plug.dtype` or one of the types in the tuple :py:attr:`~Plug.dtype`.
        """

        try:
            self._get_plug(name, default=value).default = value
        except AttributeError as exception:
            raise AttributeError(f'"{type(self._instance)}" object has no attribute "{name}" of type "{Slot}".') from exception
        except TypeError as exception:
            slot: Slot | None = getattr(type(self._instance), name, None)
            if slot is not None:
                raise TypeError(
                    f'The data type of the default value of the "{type(self._instance)}" object is not of type "{slot.dtype}".'
                ) from exception
            raise TypeError(f'The data type of the default value of the "{type(self._instance)}" object is invalid.') from exception

    def __delattr__(self, name: str) -> None:
        """Deletes the default value of the slot with the specified ``name`` that associated with the owner class instance.

        Args:
            name (str): The name of the slot to delete the default value for.

        Raises:
            AttributeError: There is no slot with the specified ``name`` in the associated owner class instance.
        """

        try:
            del self._get_plug(name).default
        except AttributeError as exception:
            raise AttributeError(f'"{type(self._instance)}" object has no attribute "{name}" of type "{Slot}".') from exception

    def __dir__(self) -> list[str]:
        """Returns a list of all slots of the associated owner class instance.

        Returns:
            list[str]: Returns a list of all slots of the associated owner class instance.
        """

        return list(type(self._instance).collect(Slot))


class Plugboard(Tracker, EmptyInit):
    """Optional Manager class for slots. Uses :py:class:`SlotDefaultAccess` to access :py:class:`Plug` default values. Also initializes
    :py:class:`Plug` container object values during instantiation by keywords.

    See Also:
        * :py:class:`Slot`
        * :py:class:`SlotDefaultAccess`
        * :py:class:`Plug`
    """

    default = SlotDefaultAccess()
    """Contains a proxy object to access the default values of the owning class of a :py:class:`Plug`."""

    def __init__(self, **kwargs: typing.Any) -> None:
        """Initializes a new :py:class:`Plugboard` instance and initializes the slots via the keyword arguments passed in.

        Args:
            **kwargs (typing.Any): The keyword arguments that are used to initialize slots. Only keyword arguments which correspond to the slot
                attribute names of the class are processed. All other keyword arguments are passed to the constructor of the next class in the
                inheritance hierarchy.
        """

        slots = self.collect(Slot)
        non_slot_kwargs = {attribute_name: attribute_value for attribute_name, attribute_value in kwargs.items() if attribute_name not in slots}
        super().__init__(**non_slot_kwargs)

        for attribute_name, attribute_value in kwargs.items():
            setattr(self, attribute_name, attribute_value)

    def reset_defaults(self) -> None:
        """Deletes the default values of all plugs of this instance."""

        # Please note, that although, self.default has not been initialized with an instance of the owning class, delattr will cause __get__ to be
        # called on self.default, which will return a new instance of SlotDefaultAccess, which is initialized with the instance of the owning class;
        # so at runtime, this will magically work, even though it looks like it should not
        for attribute_name in self.collect(Slot):
            delattr(self.default, attribute_name)

    def update_defaults(self, **kwargs: typing.Any) -> None:
        """Updates the default values of all plugs of this instance using the keyword arguments.

        Args:
            **kwargs (typing.Any): The keyword arguments that are used to update the default values of the plugs.
        """

        for attribute_name, new_attribute_default_value in kwargs.items():
            setattr(self.default, attribute_name, new_attribute_default_value)
