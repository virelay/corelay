"""Plugboards are classes which contain Slots that are filled using Plugs."""
from .tracker import Tracker


class EmptyInit:
    """Empty Init is a class intened to be inherited as a last step down the MRO, to catch any remaining positional
    and/or keyword arguments and thus raise proper Exceptions.
    """
    # Following is not useless, since this super delegation causes python to raise a more informative exception.
    # pylint: disable=useless-super-delegation
    def __init__(self):
        super().__init__()


class Slot(EmptyInit):
    """Slots are descriptors that contain objects in a container called `Plug`. `Slot`s have a dtype and a default
    value, which are enforced to be consistent. When a Slot instance is accessed in a class, it will return the
    contained object of its `Plug` container. When accessing or assigning Slot instances in a class, if it never has
    been accessed before, a Plug object is stored in the class' __dict__ under the same name the Slot was assigned to in
    the class. Slots may have their default value set to None, in which case setting Plugs belonging to it must have
    either a default, or an explicit object value on their own. Calling a Slot instance creates a corresponding Plug
    container instance.

    Note
    ----
    See `https://docs.python.org/3/howto/descriptor.html` for information on descriptors.

    Attributes
    ----------
    dtype : type or tuple of type
        Allowed type(s) of the parameter.
    default : :obj:`dtype`
        Default parameter value, should be an instance of (one of) :obj:`dtype`.
    optional : bool
        Non-mutuable. True, if Slot has a non-None default value, e.g. is optional.

    See Also
    --------
    :obj:`Plugboard`
    :obj:`Plug`

    """
    def __init__(self, dtype=object, default=None, **kwargs):
        """Configure type and default value of slot. A consistency check is performed.

        Parameters
        ----------
        dtype : type or tuple of type
            Allowed type(s) inside the slot.
        default : :obj:`dtype`
            Default plug value, should be an instance of (one of) :obj:`dtype`. May be `None` to indicate that no
            default value has been set.
        **kwargs :
            Keyword arguments passed down to potential entries below Slot in the MRO, for cooperativities' sake. In
            normal cases, this next class will be `EmptyInit`, and thus accept no more kwargs.

        """
        super().__init__(**kwargs)
        self._dtype = dtype if isinstance(dtype, tuple) else (dtype,)
        self._default = default
        self.__name__ = ''
        self._consistent()

    def _consistent(self):
        """Check whether self.dtype and self.default are consistent, e.g. self.default is either None, or of type
        self.dtype.

        Raises
        -------
        TypeError
            If self is not consistent in the sense written above.
        """
        if (
                not isinstance(self.dtype, type)
                and not (
                    isinstance(self.dtype, tuple)
                    and all(isinstance(element, type) for element in self.dtype)
                )
        ):
            raise TypeError(
                f"'{type(self).__name__}' object '{self.__name__}' default values '{self.dtype}' "
                "is neither a type, nor a tuple of types."
            )
        if self.default is not None and not isinstance(self.default, self.dtype):
            raise TypeError(
                f"'{type(self).__name__}' object '{self.__name__}' default value is not of type '{self.dtype}'."
            )

    def get_plug(self, instance, obj=None, default=None):
        """Get a corresponding Plug be accessing the Slot's instance's __dict__. For accessing a newly created and
        possibly mandatory Plug's, a obj and default value may be passed.

        Parameters
        ----------
        instance : object
            An instance of the class Slot was assigned in.
        obj : :obj:`dtype`
            Object value to write to newly created Plug instances.
        default : :obj:`dtype`
            Default value to write to newly created Plug instances.

        Returns
        -------
        :obj:`Plug`
            Either an existing Plug container from the instance's __dict__ if available, or a newly created Plug
            container instance, which is also appended to the instance's __dict__.

        """
        try:
            plug = instance.__dict__[self.__name__]
        except KeyError:
            plug = self(obj=obj, default=default)
            instance.__dict__[self.__name__] = plug
        return plug

    def __set_name__(self, owner, name):
        """Set the name of the Slot when assigned under a class. Necessary to write the correct __dict__ entry of the
        parent class.

        Parameters
        ----------
        owner : type
            Parent class the Slot was defined in.
        name : str
            Name under which the Slot was defined in the parent class.

        """
        self.__name__ = name

    def __get__(self, instance, owner):
        """If accessed under the class, return the Slot object. If accessed under an instance, return the corresponding
        Plug container's value.

        Parameters
        ----------
        instance : type
            Instance of parent class the Slot was defined in.
        owner : type
            Parent class the Slot was defined in.

        Returns
        -------
        object
            Value of the Plug container

        """
        if instance is None:
            return self
        return self.get_plug(instance).obj

    def __set__(self, instance, value):
        """Set the instance's Plug container object value

        Parameters
        ----------
        instance : type
            Instance of parent class the Slot was defined in.
        value : object
            Value to set the Plug container's object value to

        """
        self.get_plug(instance, obj=value).obj = value

    def __delete__(self, instance):
        """Delete the instance's Plug container object value if existing, enforcing the use of its default value.

        Parameters
        ----------
        instance : type
            Instance of parent class the Slot was defined in.

        Raises
        ------
        TypeError
            If Slot is mandatory and Plug does not have a default value

        """
        del self.get_plug(instance).obj

    @property
    def default(self):
        """Get the Slot's default value"""
        return self._default

    @default.setter
    def default(self, value):
        """Set the Slot's default value. Checks for consistency."""
        self._default = value
        self._consistent()

    @default.deleter
    def default(self):
        """Set the Slot's default value."""
        self._default = None

    @property
    def dtype(self):
        """Get the Slot's dtype."""
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        """Set the Slot's dtype. Checks for consistency."""
        self._dtype = value
        self._consistent()

    @property
    def optional(self):
        """Get whether the Slot is optional."""
        return self.default is not None

    def __call__(self, obj=None, default=None):
        """Create a new corresponding Plug container

        Parameters
        ----------
        obj : object
            Value to intialize the newly created Plug container's object value to.
        default : object
            Value to intialize the newly created Plug container's default value to.

        Returns
        -------
        :obj:`Plug`
            A newly created corresponding Plug container instance, obeying the type and optionality constraints.

        """
        return Plug(self, obj=obj, default=default)


class Plug(EmptyInit):
    """Container class to fill Slots associated with a certain instance. The instance is usually of type Plugboard, but
    may be of any kind of type.

    Attributes
    ----------
    obj : object
        Explicitly defined object held in the Plug container. If not set, self.default is returned.
    default : object
        Plug-dependent lower-priority object held in the container. If not set, self.fallback is returned.
    fallback : object
        Slot-dependent lowest-priority value to fall back to when no value has been assigned to the container. If not
        set, a TypeError is returned.
    slot : :obj:`Slot`
        Slot with which this Plug is associated.
    dtype : type
        dtype of the associated Slot.
    optional : bool
        Whether default or fallback is not None.

    See Also
    --------
    :obj:`Slot`
    :obj:`Plugboard`

    """
    def __init__(self, slot, obj=None, default=None, **kwargs):
        """Initialize container and check for consistency.

        Parameters
        ----------
        slot : :obj:`Slot`
            Slot with which this Plug is associated.
        obj : object
            Explicitly defined object held in the Plug container. If not set, self.default is returned.
        default : object
            Plug-dependent lower-priority object held in the container. If not set, self.fallback is returned.
        **kwargs :
            Keyword arguments passed down to potential entries below Plug in the MRO, for cooperativities' sake. In
            normal cases, this next class will be `EmptyInit`, and thus accept no more kwargs.

        """
        super().__init__(**kwargs)
        self._obj = obj
        self._slot = slot
        self._default = default
        self._consistent()

    def _consistent(self):
        """Checks whether all values are consistent, e.g. at least one of obj, default and fallback is set and of dtype
        slot.dtype.

        Raises
        ------
        TypeError
            If none of obj, default and fallback is set, or obj is not of type slot.dtype.

        """
        if self.obj is None:
            raise TypeError(
                f"'{type(self.slot).__name__}' object '{self.slot.__name__}' is mandatory, "
                "yet it has been accessed without being set."
            )
        if not isinstance(self.obj, self.slot.dtype):
            raise TypeError(
                f"'{type(self.slot).__name__}' object '{self.slot.__name__}' value '{self.obj}' "
                f"is not of type '{self.slot.dtype}'."
            )

    @property
    def slot(self):
        """Get associated Slot."""
        return self._slot

    @slot.setter
    def slot(self, value):
        """Set associated Slot. Checks for consistency"""
        self._slot = value
        self._consistent()

    @property
    def dtype(self):
        """Get associated Slot's dtype. Non-mutuable."""
        return self.slot.dtype

    @property
    def optional(self):
        """Get whether container has default values. Non-mutuable."""
        return self.default is not None

    @property
    def fallback(self):
        """Get associated Slot's default value. Non-mutuable."""
        return self.slot.default

    @property
    def obj(self):
        """Get contained object value. Get self.default instead if not set."""
        if self._obj is None:
            return self.default
        return self._obj

    @obj.setter
    def obj(self, value):
        """Set contained object value. Checks for consistency."""
        self._obj = value
        self._consistent()

    @obj.deleter
    def obj(self):
        """Delete contained object value by setting it to None."""
        self.obj = None

    @property
    def default(self):
        """Get low-priority contained default value. Get self.fallback instead if not set."""
        if self._default is None:
            return self.fallback
        return self._default

    @default.setter
    def default(self, value):
        """Set low-priority contained default value. Checks for consistency."""
        self._default = value
        self._consistent()

    @default.deleter
    def default(self):
        """Delete low-priority contained default value by setting it to None."""
        self.default = None


class SlotDefaultAccess:
    """A proxy-object descriptor class to access Plug default values of the owning class, since Plug-instances can not
    be returned except by accessing a classes' __dict__.

    See Also
    --------
    :obj:`Slot`
    :obj:`Plugboard`
    :obj:`Plug`

    """
    def __init__(self, instance=None):
        """Initialize associated instance reference.

        Parameters
        ----------
        instance : object
            Instance of the associated, Slot-owning owner class.

        """
        object.__setattr__(self, '_instance', instance)

    def _get_plug(self, name, default=None):
        """Return Plug of the instance of the associated Slot-owning owner class by name, by calling the Slot's
        `get_plug` method.

        Parameters
        ----------
        name : str
            Name of the Slot.
        default : object
            Default value to set if there does not yet exist a Plug instance associated with the Slot and instance.

        Returns
        -------
        :obj:`Plug`
            Plug container associated with the instance of the Slot-owning owner class and name.

        Raises
        ------
        AttributeError
            If there is no attribute in the associated owner class of this name of type :obj:`Slot`.

        """
        slot = getattr(type(self._instance), name)
        if not isinstance(slot, Slot):
            raise AttributeError(
                f"'{type(self._instance)}' object has no attribute '{name}' of type '{Slot}'"
            )
        return slot.get_plug(self._instance, default=default)

    def __get__(self, instance, owner):
        """Return a new instance of SlotDefaultAccess, initialized with the priovided instance value."""
        return type(self)(instance)

    def __set__(self, instance, value):
        """Set the default values of the associated owner class instance's Slots by assigning a dict."""
        new = type(self)(instance)
        if not isinstance(value, dict):
            raise TypeError("Can only directly set default values using a dict!")
        for key, val in value.items():
            setattr(new, key, val)

    def __getattr__(self, name):
        """Get the default values of the associated owner class instance's Slots 'name' Plug by attribute."""
        return self._get_plug(name).default

    def __setattr__(self, name, value):
        """Set the default values of the associated owner class instance's Slot's 'name' Plug by attribute."""
        self._get_plug(name, default=value).default = value

    def __delattr__(self, name):
        """Delete the default values of the associated owner class instance's Slot 'name' Plug by attribute."""
        del self._get_plug(name).default

    def __dir__(self):
        """Return a list of all Slots of the owner class."""
        return list(type(self._instance).collect(Slot))


class Plugboard(Tracker, EmptyInit):
    """Optional Manager class for Slots. Uses SlotDefaultAccess to access Plug default values. Also intializes Plug
    container object values during instatiation by keywords.

    Parameters
    ----------
    default : :obj:`SlotDefaultAccess`
        Proxy object to access instance's Plug's default values by attribute.

    See Also
    --------
    :obj:`Slot`
    :obj:`SlotDefaultAcces`
    :obj:`Plug`

    """
    default = SlotDefaultAccess()

    def __init__(self, **kwargs):
        """Initialize Plug container object values by keyword arguments.

        Parameters
        ----------
        **kwargs :
            Keyword arguements to initialize Slots. Only keyword arguments which correspond to class' Slot attribute
            names are processed. All other keyword arguements are passed to the next class' __init__ method in the MRO.

        """
        slots = self.collect(Slot)
        non_slot_kwargs = {key: val for key, val in kwargs.items() if key not in slots}
        # also pass responsibility to raise Exception on wrong kwargs to super-class
        super().__init__(**non_slot_kwargs)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def reset_defaults(self):
        """Delete default values of all Plugs of this instance."""
        for key in self.collect(Slot):
            delattr(self.default, key)

    def update_defaults(self, **kwargs):
        """Update default values of Plugs of this instance by providing their name and a new value as keyword
        arguments.

        """
        for key, val in kwargs.items():
            setattr(self.default, key, val)
