"""A module that contains the :py:class:`~corelay.tracker.Tracker`, which is used to track :py:class:`~corelay.plugboard.Slot` definitions in classes
that inherit from :py:class:`~corelay.plugboard.Plugboard`.
"""

import collections
import typing
from abc import ABCMeta
from warnings import warn


class MetaTracker(ABCMeta):
    """A meta class to track attributes of a type.

    Note:
        This is used to track the slots of a class. In CoRelAy, slots are declared as class attributes, and during instantiation, the class attributes
        are converted to respective instance attributes of the data type declared in the slot. For example, a :py:class:`~corelay.base.Param` slot is
        used to declare parameters of processors. When a processor has a class attribute that is a :py:class:`~corelay.base.Param` with a data type of
        :py:class:`int`, then when the processor is instantiated, an instance attribute of the same name is created with the data type of
        :py:class:`int`.

        There are two ways a slot can be declared:

        1) The old way of declaring a slot, which is to declare it as a class attribute and assign it a :py:class:`~corelay.plugboard.Slot` instance,
           e.g., `param = Param(int, 0)`. This is not the recommended way of declaring a slot anymore, because it causes problems with static type
           checkers, like MyPy. In Python, class attributes can be accessed using the class or using the instance. For example, if a class ``Test``
           has a class attribute `a = 5`, then `Test.a` and `Test().a` will both return `5`. Static type checkers, like MyPy, do not know that we are
           converting the class attribute ``a`` to an instance attribute of type :py:class:`int` during instantiation, so they will assume that when a
           slot is accessed using the instance, it will have the same type as the class attribute. This means that the static type checker will show
           an error when the slot is accessed using the instance, because the type of the class attribute is :py:class:`~corelay.plugboard.Slot` and
           not :py:class:`int`.
        2) The new way of declaring a slot, which is to declare it as a class attribute of type :py:class:`typing.Annotated`. The
           :py:class:`typing.Annotated` type is a special type that allows us to add metadata to a type hint. The first argument of
           :py:class:`typing.Annotated` is the actual type of the attribute, and then any number of additional arguments can be passed, which are used
           as metadata. So, for example, a parameter slot can be declared as `param: Annotated[float, Param(float, 0.0)]`. Since the static type
           checker knows that the actual type of the attribute is :py:class:`float`, it will not show an error when the slot is accessed using the
           instance. The metadata is used to store the instance of the slot, which contains the information about the data type, the default value,
           etc. Unfortunately, this is only a declaration, which has no effect on the runtime. This means, that no actual class attribute is created.
           But :py:class:`typing.Annotated` will add a special ``__annotations__`` attribute to the class, which contains a :py:class:`dict` with the
           names of the declared class attributes and their metadata.

        The :py:class:`MetaTracker` meta class is used to track the class attributes of a class that are not special "dunder" attributes like
        ``__class__``, as well as the declared class attributes from the ``__annotations__`` attribute. The tracked class attributes are stored in an
        :py:class:`collections.OrderedDict` called :py:attr:`Tracker.__tracked__`. The :py:class:`Tracker` class uses the :py:class:`MetaTracker` meta
        class and allows users to access the tracked class attributes. The class attributes are tracked in the order they were declared in the class,
        with the caveat that first, all class attributes come in the order of declaration, and then all declared class attributes come in the order of
        declaration. As long as only one method of declaring a slot is used, the order of declaration will be preserved.

        For more information on meta classes, please refer to `PEP 3115 <https://peps.python.org/pep-3115/>`_.

    Example:
        >>> class OrderedInts(metaclass=MetaTracker):
        ...     a = 14
        ...     b = 21
        ...     c = 42
        ... OrderedInts(a=0).__tracked__
        collections.OrderedDict([('a', 0), ('b', 21), ('c', 42)])
    """

    @classmethod
    def __prepare__(  # pylint: disable=unused-argument
        mcs,
        class_name: str,
        base_classes: tuple[type, ...],
        /,
        **kwargs: typing.Any
    ) -> collections.OrderedDict[str, typing.Any]:
        """Prepare the class dict to be an :py:class:`collections.OrderedDict`. This is done to preserve the order of declaration of the class
        attributes.

        Args:
            class_name (str): The name of the class.
            base_classes (tuple[type, ...]): The base classes of the class.
            **kwargs (typing.Any): Additional keyword arguments.

        Returns:
            collections.OrderedDict[str, typing.Any]: Returns a :py:class:`collections.OrderedDict`, which will be used as the dictionary for the
            class attributes.
        """

        return collections.OrderedDict()

    def __new__(mcs, class_name: str, base_classes: tuple[type, ...], class_attributes: collections.OrderedDict[str, typing.Any]) -> 'MetaTracker':
        """Is called when a new class is created with the :py:class:`MetaTracker` as its metaclass. Attaches a new :py:attr:`Tracker.__tracked__`
        attribute to the class, which is a :py:class:`dict` with all public attributes of the class, i.e., those not enclosed in double underscores.
        If the class that is being created already has a :py:attr:`Tracker.__tracked__` attribute, the new attributes are appended to it.

        Args:
            class_name (str): The name of the class.
            base_classes (tuple[type, ...]): The base classes of the class.
            class_attributes (collections.OrderedDict[str, typing.Any]): A :py:class:`dict` with the attributes of the class. In this case, with the
                addition of tracker attributes.

        Returns:
            MetaTracker: Returns the new class with the :py:attr:`Tracker.__tracked__` attribute.
        """

        # Retrieves the class attributes that are not special "dunder" attributes, like __class__, i.e., any class attributes that are not enclosed in
        # double underscores (this is the classical way in which slots can be declared)
        tracked_class_attributes: collections.OrderedDict[str, typing.Any] = collections.OrderedDict(
            (attribute_name, attribute_value)
            for attribute_name, attribute_value in class_attributes.items()
            if not (attribute_name[:2] + attribute_name[-2:]) == '____'
        )

        def is_slot(object_to_check: typing.Any) -> bool:
            """Checks if the given object is either a :py:class:`~corelay.plugboard.Slot` or of a type that directly or indirectly derives from it.
            This function is needed, because the module containing the :py:class:`~corelay.plugboard.Slot` class imports
            :py:class:`~corelay.tracker.Tracker`. Therefore, directly importing the :py:class:`~corelay.plugboard.Slot` class here would cause a
            circular import.

            Args:
                object_to_check (typing.Any): The object to check if it is a :py:class:`~corelay.plugboard.Slot`.

            Returns:
                bool: Returns :py:obj:`True` if the given object is either a :py:class:`~corelay.plugboard.Slot` or derives directly or indirectly
                from it. Otherwise :py:obj:`False` is returned.
            """

            if object_to_check is None:
                return False

            classes_to_check: list[type] = [object_to_check.__class__]
            classes_checked: list[type] = []
            while classes_to_check:
                current_class: type = classes_to_check.pop(0)

                if current_class.__name__ == 'Slot':
                    return True
                classes_checked.append(current_class)

                for base_class in current_class.__bases__:
                    if base_class not in classes_checked and base_class not in classes_to_check:
                        classes_to_check.append(base_class)

            return False

        # Since the old syntax of declaring slots is deprecated, but still supported for the time being, a deprecation warning is issued
        for attribute_name, attribute_value in tracked_class_attributes.items():
            if is_slot(attribute_value):
                warn(
                    f'The {attribute_value.__class__.__name__} "{attribute_name}" was declared using the old syntax of declaring slots, which is '
                    'deprecated. This syntax is currently still supported, but it will be removed in a future version of CoRelAy. Please refer to '
                    'the migration guide to find out why this syntax was deprecated and how to update your code: '
                    'https://corelay.readthedocs.io/en/latest/migration-guide/migrating-from-v0.2-to-v0.3.html.',
                    DeprecationWarning,
                    stacklevel=2
                )

        # Retrieves the declared class attributes, which were declared using Annotated and are are not special "dunder" attributes, like __class__,
        # i.e., any class attributes that are not enclosed in double underscores (this is the new way in which slots can be declared)
        tracked_declared_class_attributes: collections.OrderedDict[str, typing.Any] = collections.OrderedDict()
        if '__annotations__' in class_attributes:
            for attribute_name, attribute_value in class_attributes['__annotations__'].items():
                if (attribute_name[:2] + attribute_name[-2:]) == '____':
                    continue
                if not hasattr(attribute_value, '__metadata__'):
                    continue
                tracked_declared_class_attributes[attribute_name] = attribute_value.__metadata__[0]

        # The way that slots work is, that they override the __get__, __set__, and __delete__ methods of the class, which are invoked when the class
        # or instance attribute that the slot is assigned to is accessed (i.e., when the class or instance attribute is read, written, or deleted);
        # for this reason the slots that were declared using Annotated have to be added as real class attributes to the class, otherwise, the user
        # will not be able to access them
        class_attributes.update(tracked_declared_class_attributes)

        # Creates a new class with the given name, bases, and class attributes
        new_class: typing.Any = super().__new__(mcs, class_name, base_classes, dict(class_attributes))

        # Checks if the class or one of its base classes already has a __tracked__ attribute, if not, a new __tracked__ attribute is created,
        # otherwise, the __tracked__ attribute is copied
        tracked_attributes: collections.OrderedDict[str, typing.Any] = collections.OrderedDict()
        if hasattr(new_class, '__tracked__') and isinstance(new_class.__tracked__, collections.OrderedDict):
            tracked_attributes = new_class.__tracked__.copy()

        # Adds the class attributes and the declared class attributes that were retrieved above
        tracked_attributes.update(tracked_class_attributes)
        tracked_attributes.update(tracked_declared_class_attributes)

        # Adds the __tracked__ attribute to the new class and returns it
        new_class.__tracked__ = tracked_attributes
        tracked_new_class: MetaTracker = new_class
        return tracked_new_class


class Tracker(metaclass=MetaTracker):
    """Tracks all public class attributes, i.e., all class attributes not enclosed in double underscores. This makes them available in a class
    attribute :py:attr:`Tracker.__tracked__` using the meta class :py:class:`MetaTracker`.
    """

    __tracked__: collections.OrderedDict[str, typing.Any]
    """An :py:class:`collections.OrderedDict` with all public class attributes, i.e., all class attributes not enclosed with double underscores.

    :meta hide-value:
    """

    @classmethod
    def collect(cls, dtype: type | tuple[type, ...]) -> collections.OrderedDict[str, typing.Any]:
        """Retrieves all tracked class attributes of a certain type.

        Args:
            dtype (type | tuple[type, ...]): The type or types of the class attributes to retrieve.

        Returns:
            collections.OrderedDict[str, typing.Any]: Returns an :py:class:`collections.OrderedDict` that contains the public class attributes, i.e.,
            all class attributes not enclosed in double underscores, of the given type or types. The keys are the attribute names and the values are
            the attribute values.
        """

        return collections.OrderedDict(
            (attribute_name, attribute_value)
            for attribute_name, attribute_value in cls.__tracked__.items()
            if isinstance(attribute_value, dtype)
        )

    @classmethod
    def get(cls, attribute_name: str) -> typing.Any:
        """Retrieves a tracked class attribute by name.

        Args:
            attribute_name (str): The name of the class attribute to retrieve.

        Raises:
            AttributeError: The class attribute does not exist.

        Returns:
            typing.Any: Returns the value of the class attribute with the given name. If the class attribute does not exist :py:obj:`None` is
            returned.
        """

        if attribute_name not in cls.__tracked__:
            raise AttributeError(f"Class attribute '{attribute_name}' does not exist.")
        return cls.__tracked__.get(attribute_name)

    def collect_attr(self, dtype: type | tuple[type, ...]) -> collections.OrderedDict[str, typing.Any]:
        """Retrieves all instance attributes, corresponding to tracked class attributes of a certain type.

        Args:
            dtype (type | tuple[type, ...]): The type or types of the instance attributes to retrieve.

        Returns:
            collections.OrderedDict[str, typing.Any]: Returns an :py:class:`collections.OrderedDict` that contains the instance attributes,
            corresponding to tracked class attributes, of the given type or types. The keys are the attribute names and the values are the attribute
            values.
        """

        return collections.OrderedDict(
            (attribute_name, getattr(self, attribute_name, None))
            for attribute_name, attribute_value in self.__tracked__.items()
            if isinstance(attribute_value, dtype)
        )

    def get_attr(self, attribute_name: str) -> typing.Any:
        """Retrieves an instance attribute, corresponding to a tracked class attribute, by name.

        Args:
            attribute_name (str): The name of the class attribute to retrieve.

        Raises:
            AttributeError: The instance attribute does not exist.

        Returns:
            typing.Any: Returns the value of the instance attribute with the given name. If the instance attribute does not exist
            :py:obj:`None` is returned.
        """

        if attribute_name not in self.__tracked__:
            raise AttributeError(f"Instance attribute '{attribute_name}' does not exist.")
        return getattr(self, attribute_name)
