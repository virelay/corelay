"""Includes MetaTracker to track definition order of class attributes."""

from abc import ABCMeta
from collections import OrderedDict
from typing import Any


class MetaTracker(ABCMeta):
    """A meta class to track attributes of a type.

    Note:
        This is used to track the slots of a class. In CoRelAy, slots are declared as class attributes, and during instantiation, the class attributes
        are converted to respective instance attributes of the data type declared in the slot. For example, a ``Param`` slot is used to declare
        parameters of processors. When a processor has a class attribute that is a ``Param`` with a data type of ``int``, then when the processor is
        instantiated, an instance attribute of the same name is created with the data type of ``int``. There are two ways a slot can be declared:

        1) The old way of declaring a slot, which is to declare it as a class attribute and assign it a ``Slot`` instance, e.g.,
           ``param = Param(int, 0)``. This is not the recommended way of declaring a slot anymore, because it causes problems with static type
           checkers, like MyPy. In Python, class attributes can be accessed using the class or using the instance. For example, if a class ``Test``
           has a class attribute ``a = 5``, then ``Test.a`` and ``Test().a`` will both return ``5``. Static type checkers, like MyPy, do not know that
           we are converting the class attribute ``a`` to an instance attribute of type ``int`` during instantiation, so they will assume that when a
           slot is accessed using the instance, it will have the same type as the class attribute. This means that the static type checker will show
           an error when the slot is accessed using the instance, because the type of the class attribute is ``Slot`` and not ``int``.
        2) The new way of declaring a slot, which is to declare it as a class attribute of type ``Annotated``. The ``Annotated`` type is a special
           type that allows us to add metadata to a type hint. The first argument of ``Annotated`` is the actual type of the attribute, and then any
           number of additional arguments can be passed, which are used as metadata. So, for example, a parameter slot can be declared as
           ``param: Annotated[float, Param(float, 0.0)]``. Since the static type checker knows that the actual type of the attribute is ``float``, it
           will not show an error when the slot is accessed using the instance. The metadata is used to store the instance of the slot, which contains
           the information about the data type, the default value, etc. Unfortunately, this is only a declaration, which has no effect on the runtime.
           This means, that no actual class attribute is created. But ``Annotated`` will add a special ``__annotations__`` attribute to the class,
           which contains a dictionary with the names of the declared class attributes and their metadata.

        The ``MetaTracker`` meta class is used to track the class attributes of a class that are not special "dunder" attributes like ``__class__``,
        as well as the declared class attributes from the ``__annotations__`` attribute. The tracked class attributes are stored in an ``OrderedDict``
        called ``__tracked__``. The ``Tracker`` class uses the ``MetaTracker`` meta class and allows users to access the tracked class attributes. The
        class attributes are tracked in the order they were declared in the class, with the caveat that first, all class attributes come in the order
        of declaration, and then all declared class attributes come in the order of declaration. As long as only one method of declaring a slot is
        used, the order of declaration will be preserved.

        For more information on meta classes, please refer to `PEP 3115 <https://peps.python.org/pep-3115/>`_.

    Example:
        >>> class OrderedInts(metaclass=MetaTracker):
        ...     a = 14
        ...     b = 21
        ...     c = 42
        ... OrderedInts(a=0).__tracked__
        OrderedDict([('a', 0), ('b', 21), ('c', 42)])
    """

    @classmethod
    def __prepare__(  # pylint: disable=unused-argument
        mcs,
        class_name: str,
        base_classes: tuple[type, ...],
        /,
        **kwargs: Any
    ) -> OrderedDict[str, Any]:
        """Prepare the class dict to be an ``OrderedDict``. This is done to preserve the order of declaration of the class attributes.

        Args:
            class_name (str): The name of the class.
            base_classes (tuple[type, ...]): The base classes of the class.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            OrderedDict[str, Any]: Returns a ``OrderedDict``, which will be used as the dictionary for the class attributes.
        """

        return OrderedDict()

    def __new__(mcs, class_name: str, base_classes: tuple[type, ...], class_attributes: OrderedDict[str, Any]) -> 'MetaTracker':
        """Is called when a new class is created with the ``MetaTracker`` as its metaclass. Attaches a new ``__tracked__`` attribute to the class,
        which is a dictionary with all public attributes of the class, i.e., those not enclosed in double underscores. If the class that is being
        created already has a ``__tracked__`` attribute, the new attributes are appended to it.

        Args:
            class_name (str): The name of the class.
            base_classes (tuple[type, ...]): The base classes of the class.
            class_attributes (OrderedDict[str, Any]): A dictionary with the attributes of the class. In this case, with the addition of tracker
                attributes.

        Returns:
            MetaTracker: Returns the new class with the ``__tracked__`` attribute.
        """

        # Retrieves the class attributes that are not special "dunder" attributes, like __class__, i.e., any class attributes that is not enclosed in
        # double underscores (this is the classical way in which slots can be declared)
        tracked_class_attributes: OrderedDict[str, Any] = OrderedDict(
            (attribute_name, attribute_value)
            for attribute_name, attribute_value in class_attributes.items()
            if not (attribute_name[:2] + attribute_name[-2:]) == '____'
        )

        # Retrieves the declared class attributes, which were declared using Annotated (this is the new way in which slots can be declared)
        tracked_declared_class_attributes: OrderedDict[str, Any] = OrderedDict()
        if '__annotations__' in class_attributes:
            for attribute_name, attribute_value in class_attributes['__annotations__'].items():
                if (attribute_name[:2] + attribute_name[-2:]) == '____':
                    continue
                if not hasattr(attribute_value, '__metadata__'):
                    continue
                if not isinstance(attribute_value.__metadata__, tuple) or len(attribute_value.__metadata__) == 0:
                    continue
                tracked_declared_class_attributes[attribute_name] = attribute_value.__metadata__[0]

        # The way that slots work is, that they override the __get__, __set__, and __delete__ methods of the class, which are invoked when the class
        # or instance attribute that the slot is assigned to is accessed (i.e., when the class or instance attribute is read, written, or deleted);
        # for this reason the slots that were declared using Annotated have to be added as real class attributes to the class, otherwise, the user
        # will not be able to access them
        class_attributes.update(tracked_declared_class_attributes)

        # Creates a new class with the given name, bases, and class attributes
        new_class: Any = super().__new__(mcs, class_name, base_classes, dict(class_attributes))

        # Checks if the class or one of its base classes already has a __tracked__ attribute, if not, a new __tracked__ attribute is created,
        # otherwise, the __tracked__ attribute is copied
        tracked_attributes: OrderedDict[str, Any] = OrderedDict()
        if hasattr(new_class, '__tracked__') and isinstance(new_class.__tracked__, OrderedDict):
            tracked_attributes = new_class.__tracked__.copy()

        # Adds the class attributes and the declared class attributes that were retrieved above
        tracked_attributes.update(tracked_class_attributes)
        tracked_attributes.update(tracked_declared_class_attributes)

        # Adds the __tracked__ attribute to the new class and returns it
        new_class.__tracked__ = tracked_attributes
        tracked_new_class: MetaTracker = new_class
        return tracked_new_class


class Tracker(metaclass=MetaTracker):
    """Tracks all public class attributes, i.e., all class attributes not enclosed int double underscores. This makes them available in a class
    attribute ``__tracked__`` using the meta class ``MetaTracker``.
    """

    __tracked__: OrderedDict[str, Any]
    """An ``OrderedDict`` with all public class attributes, i.e., all class attributes not enclosed with double underscores."""

    @classmethod
    def collect(cls, dtype: type | tuple[type, ...]) -> OrderedDict[str, Any]:
        """Retrieves all tracked class attributes of a certain type.

        Args:
            dtype (type | tuple[type, ...]): The type or types of the class attributes to retrieve.

        Returns:
            OrderedDict[str, Any]: Returns an ``OrderedDict`` that contains the public class attributes, i.e., all class attributes not enclosed in
                double underscores, of the given type or types. The keys are the attribute names and the values are the attribute values.
        """

        return OrderedDict(
            (attribute_name, attribute_value)
            for attribute_name, attribute_value in cls.__tracked__.items()
            if isinstance(attribute_value, dtype)
        )

    @classmethod
    def get(cls, attribute_name: str) -> Any:
        """Retrieves a tracked class attribute by name.

        Args:
            attribute_name (str): The name of the class attribute to retrieve.

        Raises:
            AttributeError: The class attribute does not exist.

        Returns:
            Any: Returns the value of the class attribute with the given name. If the class attribute does not exist `None` is returned.
        """

        if attribute_name not in cls.__tracked__:
            raise AttributeError(f"Class attribute '{attribute_name}' does not exist.")
        return cls.__tracked__.get(attribute_name)

    def collect_attr(self, dtype: type | tuple[type, ...]) -> OrderedDict[str, Any]:
        """Retrieves all instance attributes, corresponding to tracked class attributes of a certain type.

        Args:
            dtype (type | tuple[type, ...]): The type or types of the instance attributes to retrieve.

        Returns:
            OrderedDict[str, Any]: Returns an ``OrderedDict`` that contains the instance attributes, corresponding to tracked class attributes, of the
                given type or types. The keys are the attribute names and the values are the attribute values.
        """

        return OrderedDict(
            (attribute_name, getattr(self, attribute_name, None))
            for attribute_name, attribute_value in self.__tracked__.items()
            if isinstance(attribute_value, dtype)
        )

    def get_attr(self, attribute_name: str) -> Any:
        """Retrieves an instance attribute, corresponding to a tracked class attribute, by name.

        Args:
            attribute_name (str): The name of the class attribute to retrieve.

        Raises:
            AttributeError: The instance attribute does not exist.

        Returns:
            Any: Returns the value of the instance attribute with the given name. If the instance attribute does not exist `None` is returned.
        """

        if attribute_name not in self.__tracked__:
            raise AttributeError(f"Instance attribute '{attribute_name}' does not exist.")
        return getattr(self, attribute_name)
