"""A module that contains plugboards, which  are classes that contains slots filled using plugs."""

from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, LambdaType, MethodType
from typing import Any

import numpy

from corelay.tracker import Tracker


class EmptyInit:
    """Empty Init is a class intended to be inherited as a last step down the MRO, to catch any remaining positional and/or keyword arguments and thus
    raise proper Exceptions.
    """

    def __init__(self) -> None:
        """Initializes a new instance of EmptyInit.

        Note:
            This is not intended to be called directly, but rather by the constructor of the next class up in the inheritance hierarchy. The
            super-delegation is not unnecessary, even if PyLint claims so, since this is causes Python to raise a more informative exception when the
            user tries to pass more keyword arguments than are accepted by the constructors in the inheritance hierarchy. If the constructor of the
            penultimate class in inheritance hierarchy calls this constructor and there are still keyword arguments left, Python will raise a
            TypeError with a message like: "TypeError: object.__init__() takes exactly one argument (the instance to initialize)". This may confuse
            users, but if the constructor call is delegated to the next class up (i.e., object), the exception will be more informative and say
            something like: "TypeError: Empty.__init__() takes 0 positional arguments but 1 was given".
        """

        # pylint: disable=useless-super-delegation
        super().__init__()


class Slot(EmptyInit):
    """Slots are descriptors that contain objects in a container called ``Plug``. Instances of the ``Slot`` class have a ``dtype`` and a ``default``
    value, which are enforced to be consistent. When a ``Slot`` instance is accessed in a class, it will return the contained object of its ``Plug``
    container. When accessing or assigning ``Slot`` instances in a class that have never been accessed before, a ``Plug`` object is stored in the
    class' ``__dict__`` under the same name the ``Slot`` was assigned to in the class. Slots may have their default value set to `None`, in which case
    setting Plugs belonging to it must have either a default value, or an explicit ``obj`` value on their own. Calling a ``Slot`` instance creates a
    corresponding ``Plug`` container instance.

    Note:
        See `https://docs.python.org/3/howto/descriptor.html` for more information on descriptors.

    See Also:
        :obj:`Plugboard`
        :obj:`Plug`
    """

    def __init__(
        self,
        dtype: type | tuple[type, ...] = object,
        default: Any = None,
        **kwargs: Any
    ) -> None:
        """Initializes a new instance of ``Slot``. Configures that data type and the default value of the slot. A consistency check is performed to
        ensure that the default value is of the correct type.

        Args:
            dtype (type | tuple[type, ...], optional): The data type of the slot. This can be a single type or a tuple of types. The value of the
                plug, as well as the default value, must be of this type or one of the types in the tuple. Defaults to ``object``.
            default (Any, optional): The default value of the plug. The default value must be an instance of the specified data type ``dtype`` or one
                of the types in the tuple. If no default value is set, it will be `None`. When a plug is created without an explicit value, it will
                use this default value. Defaults to `None`.
            **kwargs (Any): Additional keyword arguments that are passed to the parent class constructor. This is done for cooperativity's sake, as
                the next class one step up in the inheritance hierarchy will be `EmptyInit`, which does not accept any additional keyword arguments
                and will raise an exception if any are passed.
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
    """Contains all types that may represent a function. This is necessary, since many functions are not of type ``FunctionType``, e.g., lambda
    expressions, methods, built-in functions, built-in methods, and NumPy universal functions. Also, since NumPy 1.26, NumPy array functions are no
    longer actual functions; so, for example, something like `pooling_function: Annotated[FunctionType, Param(FunctionType, numpy.sum)]` would not
    work anymore. When the user sets the dtype to ``FunctionType`` or ``FunctionType`` is an element of the tuple, then the types in this tuple are
    added to the data types that ``_consistent`` checks against. This way, the user does not have to worry about the fact that many functions are not
    of type ``FunctionType``.
    """

    def _consistent(self) -> None:
        """Checks whether ``dtype`` and ``default`` are consistent, i.e., ``default`` is either `None`, of type ``dtype``, or one of the types in the
        tuple ``dtype``.

        Raises:
            TypeError: The ``default`` value is not consistent with the ``dtype``, i.e., it is neither `None`, nor of the type ``dtype`` or one of the
                types in the tuple ``dtype``.
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

    def get_plug(self, instance: Any, obj: Any = None, default: Any = None) -> 'Plug':
        """Gets a corresponding ``Plug`` that can be used to access the ``__dict__`` of the ``Slot`` instance . In case a new ``Plug`` has to be
        created, a ``obj`` and ``default`` value may be specified.

        Args:
            instance (Any): An instance of the class the ``Slot`` was assigned in.
            obj (Any): The object value to write to newly created ``Plug`` instances.
            default (Any): The default value to write to newly created ``Plug`` instances.

        Returns:
            Plug: Returns a ``Plug`` container. If a ``Plug`` instance already exists in the instance's ``__dict__`` it is returned. Otherwise, a new
            ``Plug`` container is created, which is also appended to the instance's ``__dict__``. If a new ``Plug`` instance is created, the ``obj``
            and ``default`` values are set to the values passed in.
        """

        try:
            plug: Plug = instance.__dict__[self.__name__]
        except KeyError:
            plug = self(obj=obj, default=default)
            instance.__dict__[self.__name__] = plug
        return plug

    def __set_name__(self, owner: type, name: str) -> None:  # pylint: disable=unused-argument
        """Is invoked, when the ``Slot`` is assigned to a class or instance attribute. Sets the name of the slot when assigned under a class.
        Necessary to write the correct ``__dict__`` entry in the parent class.

        Args:
            owner (type): The parent class the ``Slot`` was assigned in.
            name (str): The name under which the ``Slot`` was assigned in the parent class.
        """

        self.__name__ = name

    def __get__(self, instance: Any, owner: type) -> 'Slot | Any':  # pylint: disable=unused-argument
        """Is invoked, when the class or instance attribute the ``Slot`` was assigned to is read. When the ``Slot`` is accessed from a class, the
        ``Slot`` instance itself is returned. If accessed using an instance, the corresponding ``Plug`` container's value is returned.

        Args:
            instance (Any): The instance of the parent class the ``Slot`` was assigned in.
            owner (type):  The parent class the ``Slot`` was defined in.

        Returns:
            Slot | Any: Returns the value of the Plug container.
        """

        return self if instance is None else self.get_plug(instance).obj

    def __set__(self, instance: Any, value: Any) -> None:
        """Is invoked, when the class or instance attribute the ``Slot`` was assigned to is written. Sets the instance's ``Plug`` container object
        value.

        Args:
            instance (Any): The instance of the parent class the ``Slot`` was assigned in.
            value (Any): The value to set the Plug container's object value to.
        """

        self.get_plug(instance, obj=value).obj = value

    def __delete__(self, instance: Any) -> None:
        """Is invoked, when the class or instance attribute the ``Slot`` was assigned to is deleted. Deletes the instance's ``Plug`` container object
        value if it exists, enforcing the use of its default value.

        Args:
            instance (Any): The instance of the parent class the ``Slot`` was assigned in.
        """

        del self.get_plug(instance).obj

    @property
    def default(self) -> Any:
        """Gets or sets the default value of the ``Slot``.

        Returns:
            Any: Returns the slot's default value. If not set, `None` is returned.
        """

        return self._default

    @default.setter
    def default(self, value: Any) -> None:
        """Gets or sets the default value of the ``Slot``. Checks the new default value for consistency with the ``dtype``.

        Args:
            value (Any): The new default value to set. If not set, `None` is returned.

        Raises:
            TypeError: The default value is not consistent with the ``dtype``, i.e., it is neither `None`, nor of the type ``dtype`` or one of the
                types in the tuple ``dtype``.
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
        """Deletes the default value of the ``Slot``."""

        self._default = None

    @property
    def dtype(self) -> type | tuple[type, ...]:
        """Gets or sets the slot's ``dtype``.

        Returns:
            type | tuple[type, ...]: Returns the slot's ``dtype``. If not set, `None` is returned.
        """

        return self._dtype

    @dtype.setter
    def dtype(self, value: type | tuple[type, ...]) -> None:
        """Gets or sets the slot's ``dtype``. Checks the default value of the slot for consistency with the new ``dtype``.

        Args:
            value (type | tuple[type, ...]): The new ``dtype`` to set. If not set, `None` is returned.

        Raises:
            TypeError: The default value is not consistent with the ``dtype``, i.e., it is neither `None`, nor of the type ``dtype`` or one of the
                types in the tuple ``dtype``.
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
        """Gets a value indicating whether the ``Slot`` is optional.

        Returns:
            bool: Returns `True` if the ``Slot`` is optional, i.e., it has a default value, and `False` otherwise.
        """

        return self.default is not None

    def __call__(self, obj: Any = None, default: Any = None) -> 'Plug':
        """Create a new corresponding Plug container

        Args:
            obj (Any): A value to initialize the newly created ``Plug`` container's object value to. Defaults to `None`.
            default (Any): A value to initialize the newly created ``Plug`` container's default value to. Defaults to `None`.

        Returns:
            Plug: Returns a newly created ``Plug`` container instance, obeying the type and optionality constraints.
        """

        return Plug(self, obj=obj, default=default)


class Plug(EmptyInit):
    """Container class to fill slots associated with a certain instance. The instance is usually of type ``Plugboard``, but may be of any kind of
    type.

    See Also:
        :obj:`Slot`
        :obj:`Plugboard`
    """

    def __init__(self, slot: Slot, obj: Any = None, default: Any = None, **kwargs: Any) -> None:
        """Initializes a new ``Plug`` instance and checks for consistency.

        Args:
            slot (Slot): The ``Slot`` instance to associate with this ``Plug``.
            obj (Any): An explicitly defined object held in the ``Plug`` container. If not set, ``default`` is returned as its value.
            default (Any): A plug-dependent lower-priority object held in the ``Plug`` container. If not set, ``fallback`` is returned.
            **kwargs (Any): Keyword arguments passed down to the base class constructor, for cooperativity's sake. In normal cases, this next class
                will be `EmptyInit`, which accepts no more keyword arguments and will raise an exception.
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
    """Contains all types that may represent a function. This is necessary, since many functions are not of type ``FunctionType``, e.g., lambda
    expressions, methods, built-in functions, built-in methods, and NumPy universal functions. Also, since NumPy 1.26, NumPy array functions are no
    longer actual functions; so, for example, something like `pooling_function: Annotated[FunctionType, Param(FunctionType, numpy.sum)]` would not
    work anymore. When the user sets the dtype to ``FunctionType`` or ``FunctionType`` is an element of the tuple, then the types in this tuple are
    added to the data types that ``_consistent`` checks against. This way, the user does not have to worry about the fact that many functions are not
    of type ``FunctionType``.
    """

    def _consistent(self) -> None:
        """Checks whether all values are consistent, i.e., at least one of ``obj``, ``default``, or ``fallback`` is set and of the data type specified
        in `slot.dtype`, or one of the types in the tuple `slot.dtype`.

        Raises:
            TypeError: None of ``obj``, ``default``, or ``fallback`` is set, or the value is not consistent with the `slot.dtype` or one of the types
                in the tuple `slot.dtype`.
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
        """Gets or sets the associated ``Slot``.

        Returns:
            Slot: Returns the associated ``Slot``. If not set, `None` is returned.
        """

        return self._slot

    @slot.setter
    def slot(self, value: Slot) -> None:
        """Gets or sets associated ``Slot`` and checks for consistency.

        Args:
            value (Slot): The new ``Slot`` to set.
        """

        self._slot = value
        self._consistent()

    @property
    def dtype(self) -> type | tuple[type, ...]:
        """Gets the ``dtype`` of the associated ``Slot``. The ``dtype`` property is non-mutable.

        Returns:
            type | tuple[type, ...]: Returns the ``dtype`` of the associated ``Slot``.
        """

        return self.slot.dtype[0] if isinstance(self.slot.dtype, tuple) and len(self.slot.dtype) == 1 else self.slot.dtype

    @property
    def optional(self) -> bool:
        """Gets a value indicating whether the ``Plug`` container has a default value. The ``optional`` property is non-mutable.

        Returns:
            bool: Returns `True` if the ``Plug`` container has a default value, and `False` otherwise.
        """

        return self.default is not None

    @property
    def fallback(self) -> Any:
        """Gets the default value of the associated ``Slot``. The ``fallback`` property is non-mutable.

        Returns:
            Any: Returns the default value of the associated ``Slot``.
        """

        return self.slot.default

    @property
    def obj(self) -> Any:
        """Gets or sets the value of the object contained in the ``Plug``. If the ``Plug`` does not contain an object value, ``default`` is retrieved
        instead.

        Returns:
            Any: Returns the object value contained in the ``Plug``. If not set, ``default`` is returned.
        """

        if self._obj is None:
            return self.default
        return self._obj

    @obj.setter
    def obj(self, value: Any) -> None:
        """Gets or sets the value of the object contained in the ``Plug`` and checks for consistency.

        Args:
            value (Any): The new object value to set.

        Raises:
            TypeError: The object value is not consistent with the ``dtype``, i.e., it is neither `None`, nor of the type ``dtype`` or one of the
                types in the tuple ``dtype``.
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
        """Deletes the value of the object contained in the ``Plug`` by setting it to `None`."""

        self.obj = None

    @property
    def default(self) -> Any:
        """Gets or sets the default value of the ``Plug``. If the ``default`` value is not set, then the ``fallback`` value is retrieved instead.

        Returns:
            Any: Returns the default value of the ``Plug``. If not set, ``fallback`` is returned.
        """

        if self._default is None:
            return self.fallback
        return self._default

    @default.setter
    def default(self, value: Any) -> None:
        """Gets or sets the default value of the ``Plug`` and checks for consistency.

        Args:
            value (Any): The new default value to set.

        Raises:
            TypeError: The default value is not consistent with the ``dtype``, i.e., it is neither `None`, nor of the type ``dtype`` or one of the
                types in the tuple ``dtype``.
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
        """Deletes the default value of the ``Plug`` by setting it to `None`."""

        self.default = None


class SlotDefaultAccess:
    """A proxy-object descriptor class to access the default values of the owning class of a ``Slot``, since ``Slot`` instances cannot be returned
    except by accessing a classes' ``__dict__``.

    See Also:
        :obj:`Slot`
        :obj:`Plugboard`
        :obj:`Plug`
    """

    def __init__(self, instance: Tracker | Any = None) -> None:
        """Initializes a new ``SlotDefaultAccess`` instance.

        Args:
            instance (Tracker | Any): The instance of the class the ``SlotDefaultAccess`` is associated with.
        """

        # We cannot just assign self._instance here, because this would cause __get__ to be called, which in turn would create a new instance of
        # SlotDefaultAccess, which would call __init__ again, which would call __get__ again, and so on, resulting in an infinite recursion; this is
        # circumvented by using object.__setattr__ instead, which does not call __get__
        self._instance: Tracker | Any
        object.__setattr__(self, '_instance', instance)

    def _get_plug(self, name: str, default: Any = None) -> Plug:
        """Gets the ``Plug`` of the instance of the associated ``Slot``-owning class by name, by calling the ``get_plug`` method of the ``Slot``.

        Args:
            name (str):  The name of the ``Slot``.
            default (Any): The default value to set if the ``Plug`` associated with the ``Slot`` does not exist yet.

        Raises:
            AttributeError: There is no attribute in the associated owner class of this name of type ``Slot``.

        Returns:
            Plug: Returns the ``Plug`` container associated with the instance of the ``Slot``-owning class and name.
        """

        if self._instance is None:
            raise AttributeError('The instance of the class the SlotDefaultAccess is associated with is not set.')
        slot = getattr(type(self._instance), name)
        if not isinstance(slot, Slot):
            raise AttributeError(f'"{type(self._instance)}" object has no attribute "{name}" of type "{Slot}", it is of type "{type(slot)}".')
        return slot.get_plug(self._instance, default=default)

    def __get__(self, instance: Any, owner: Any) -> 'SlotDefaultAccess':  # pylint: disable=unused-argument
        """Gets a new instance of ``SlotDefaultAccess``, initialized with the provided instance value.

        Args:
            instance (Any): The instance of the class the ``SlotDefaultAccess`` is associated with.
            owner (Any): The owner class of the ``SlotDefaultAccess``.

        Returns:
            SlotDefaultAccess: Returns a new instance of ``SlotDefaultAccess`` initialized with the provided instance value.
        """

        return type(self)(instance)

    def __set__(self, instance: Any, value: dict[str, Any]) -> None:
        """Sets the default values of the associated owner class instance's slots by assigning a the values of the dictionary specified
        in ``value``.

        Args:
            instance (Any): The instance of the class the ``SlotDefaultAccess`` is associated with.
            value (dict[str, Any]): A dictionary containing the default values to set for the associated owner class instance's slots.

        Raises:
            TypeError: The ``value`` is not a dictionary.
        """

        if not isinstance(value, dict):
            raise TypeError('Can only directly set default values using a dict.')

        slot_default_accessor = type(self)(instance)
        for attribute_name, attribute_default_value in value.items():
            setattr(slot_default_accessor, attribute_name, attribute_default_value)

    def __getattr__(self, name: str) -> Any:
        """Gets the default value of the of the slot with the specified ``name`` that associated with the owner class instance.

        Args:
            name (str): The name of the slot to get the default value for.

        Raises:
            AttributeError: There is no slot with the specified ``name`` in the associated owner class instance.

        Returns:
            Any: Returns the default value of the slot with the specified ``name`` that associated with the owner class instance.
        """

        try:
            return self._get_plug(name).default
        except AttributeError as exception:
            raise AttributeError(f'"{type(self._instance)}" object has no attribute "{name}" of type "{Slot}".') from exception

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets the default value of the slot with the specified ``name`` that associated with the owner class instance.

        Args:
            name (str): The name of the slot to set the default value for.
            value (Any): The new default value to set.

        Raises:
            AttributeError: There is no slot with the specified ``name`` in the associated owner class instance.
            TypeError: The default value is not consistent with the ``dtype`` of the ``Plug``, i.e., it is neither `None`, nor of the type ``dtype``
                or one of the types in the tuple ``dtype``.
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
    """Optional Manager class for slots. Uses ``SlotDefaultAccess`` to access ``Plug`` default values. Also initializes ``Plug`` container object
    values during instantiation by keywords.

    See Also:
        :obj:`Slot`
        :obj:`SlotDefaultAccess`
        :obj:`Plug`
    """

    default = SlotDefaultAccess()
    """Contains a proxy object to access the default values of the owning class of a ``Plug``."""

    def __init__(self, **kwargs: Any) -> None:
        """Initializes a new ``Plugboard`` instance and initializes the slots via the keyword arguments passed in.

        Args:
            **kwargs (Any): The keyword arguments that are used to initialize slots. Only keyword arguments which correspond to the slot attribute
                names of the class are processed. All other keyword arguments are passed to the constructor of the next class in the inheritance
                hierarchy.
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

    def update_defaults(self, **kwargs: Any) -> None:
        """Updates the default values of all plugs of this instance using the keyword arguments.

        Args:
            **kwargs (Any): The keyword arguments that are used to update the default values of the plugs.
        """

        for attribute_name, new_attribute_default_value in kwargs.items():
            setattr(self.default, attribute_name, new_attribute_default_value)
