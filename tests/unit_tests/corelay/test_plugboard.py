"""A module that contains unit tests for the :py:mod:`corelay.plugboard` module."""

import pytest

from corelay.plugboard import Slot, Plug, Plugboard


class TestSlot:
    """Contains unit tests for the :py:class:`~corelay.plugboard.Slot` class."""

    @staticmethod
    def test_init() -> None:
        """Tests that a :py:class:`~corelay.plugboard.Slot` can be instantiated successfully."""

        Slot()

    @staticmethod
    def test_init_consistent_args() -> None:
        """Tests that a :py:class:`~corelay.plugboard.Slot` can be instantiated successfully if its arguments are consistent."""

        Slot(dtype=int, default=5)

    @staticmethod
    def test_init_inconsistent_args() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Slot` with inconsistent arguments raises an exception."""

        with pytest.raises(TypeError):
            Slot(dtype=str, default=5)

    @staticmethod
    def test_init_unknown_args() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Slot` with unknown arguments raises an exception."""

        with pytest.raises(TypeError):
            Slot(monkey='banana')

    @staticmethod
    def test_init_class_name() -> None:
        """Tests that when instantiating a class, the __name__ parameter of a contained :py:class:`~corelay.plugboard.Slot` is set accordingly."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot()
            """A test slot."""

        assert SlotHolder.my_slot.__name__ == 'my_slot'

    @staticmethod
    def test_init_instance_default() -> None:
        """Tests that when accessing a :py:class:`~corelay.plugboard.Slot` in an instance, where only the default value is set, the default is
        returned.
        """

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(default=42)
            """A test slot."""

        slot_holder = SlotHolder()
        assert slot_holder.my_slot == 42

    @staticmethod
    def test_instance_get() -> None:
        """Tests that getting a value after setting it succeeds."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int)
            """A test slot."""

        slot_holder = SlotHolder()
        slot_holder.my_slot = 15
        assert slot_holder.my_slot == 15

    @staticmethod
    def test_instance_get_no_default() -> None:
        """Tests that when accessing a :py:class:`~corelay.plugboard.Slot` in an instance, where nothing is set, an exception is raised."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot()
            """A test slot."""

        slot_holder = SlotHolder()

        with pytest.raises(TypeError):
            _ = slot_holder.my_slot

    @staticmethod
    def test_instance_set() -> None:
        """Tests that everything works alright when not setting a default value, but then setting the value of the object before accessing it."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int)
            """A test slot."""

        slot_holder = SlotHolder()
        slot_holder.my_slot = 15

        assert slot_holder.my_slot == 15

    @staticmethod
    def test_instance_set_wrong_dtype() -> None:
        """Tests that setting a value with the wrong data type raises an exception."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(str)
            """A test slot."""

        slot_holder = SlotHolder()

        with pytest.raises(TypeError):
            slot_holder.my_slot = 15

    @staticmethod
    def test_instance_delete_unchanged() -> None:
        """Tests that setting a value and deleting it afterwards with a set default value returns the default value"""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(int, 42)
            """A test slot."""

        slot_holder = SlotHolder()
        slot_holder.my_slot = 15

        del slot_holder.my_slot
        assert slot_holder.my_slot == 42

    @staticmethod
    def test_instance_delete_without_default() -> None:
        """Tests that setting a value and deleting it afterwards without a default value raises an exception."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(int)
            """A test slot."""

        slot_holder = SlotHolder()
        slot_holder.my_slot = 15

        with pytest.raises(TypeError):
            del slot_holder.my_slot

    @staticmethod
    def test_class_set_dtype() -> None:
        """Tests that setting a new data type that is consistent with the data type of the already existing default value succeeds."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=object, default=15)
            """A test slot."""

        SlotHolder.my_slot.dtype = int

    @staticmethod
    def test_class_set_dtype_inconsistent() -> None:
        """Tests that setting an new data type that is not consistent with the already existing default value fails."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=object, default=15)
            """A test slot."""

        with pytest.raises(TypeError):
            SlotHolder.my_slot.dtype = str

    @staticmethod
    def test_class_optional() -> None:
        """Tests that slots with default values are optional."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        assert SlotHolder.my_slot.optional

    @staticmethod
    def test_class_not_optional() -> None:
        """Tests that slots without default values are not optional."""

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int)
            """A test slot."""

        assert not SlotHolder.my_slot.optional

    @staticmethod
    def test_class_call() -> None:
        """Tests that calling a :py:class:`~corelay.plugboard.Slot` yields the :py:class:`~corelay.plugboard.Plug` associated with the
        :py:class:`~corelay.plugboard.Slot`.
        """

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        assert SlotHolder.my_slot is SlotHolder.my_slot().slot

    @staticmethod
    def test_class_call_obj() -> None:
        """Tests that calling a :py:class:`~corelay.plugboard.Slot` with an ``obj`` argument yields a :py:class:`~corelay.plugboard.Plug` with the
        :py:attr:`~corelay.plugboard.Plug.obj` property set to that value.
        """

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int)
            """A test slot."""

        assert SlotHolder.my_slot(obj=15).obj == 15

    @staticmethod
    def test_class_call_default() -> None:
        """Tests that calling a :py:class:`~corelay.plugboard.Slot` with ``default`` argument yields a :py:class:`~corelay.plugboard.Plug` with the
        :py:attr:`~corelay.plugboard.Plug.default` property set.
        """

        class SlotHolder:
            """A test class that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int)
            """A test slot."""

        assert SlotHolder.my_slot(default=15).default == 15


class TestPlug:
    """Contains unit tests for the :py:class:`~corelay.plugboard.Plug` class."""

    @staticmethod
    def test_init_with_slot_default() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plug` with the default value of a :py:class:`~corelay.plugboard.Slot` succeeds."""

        slot = Slot(dtype=int, default=10)
        Plug(slot)

    @staticmethod
    def test_init_no_slot_default() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plug` without the default value of a :py:class:`~corelay.plugboard.Slot` fails."""

        slot = Slot(dtype=int)

        with pytest.raises(TypeError):
            Plug(slot)

    @staticmethod
    def test_init_consistent() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plug` with the :py:attr:`~corelay.plugboard.Plug.obj` and
        :py:attr:`~corelay.plugboard.Plug.default` properties set to values that are consistent with the data type of the
        :py:class:`~corelay.plugboard.Slot` succeeds.
        """

        slot = Slot(dtype=int)
        Plug(slot, obj=15, default=16)

    @staticmethod
    def test_init_consistent_obj() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plug` with the :py:attr:`~corelay.plugboard.Plug.obj` property set to a value that
        is consistent with the data type of the :py:class:`~corelay.plugboard.Slot` succeeds.
        """

        slot = Slot(dtype=int)
        Plug(slot, obj=15)

    @staticmethod
    def test_init_consistent_default() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plug` with the :py:attr:`~corelay.plugboard.Plug.default` property set to a value
        that is consistent with the data type of the :py:class:`~corelay.plugboard.Slot` succeeds.
        """

        slot = Slot(dtype=int)
        Plug(slot, default=15)

    @staticmethod
    def test_init_inconsistent_obj() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plug` with the :py:attr:`~corelay.plugboard.Plug.obj` property set to a value that
        is not consistent with the data type of the :py:class:`~corelay.plugboard.Slot` fails.
        """

        slot = Slot(dtype=str)

        with pytest.raises(TypeError):
            Plug(slot, obj=15)

    @staticmethod
    def test_init_inconsistent_default() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plug` with the :py:attr:`~corelay.plugboard.Plug.default` property set to a value
        that is not consistent with the data type of the :py:class:`~corelay.plugboard.Slot` fails.
        """

        slot = Slot(dtype=str)

        with pytest.raises(TypeError):
            Plug(slot, default=15)

    @staticmethod
    def test_obj_hierarchy_obj() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.obj` property while the :py:class:`~corelay.plugboard.Slot` and the
        :py:class:`~corelay.plugboard.Plug` have a default values and the the :py:attr:`~corelay.plugboard.Plug.obj` property of the
        :py:class:`~corelay.plugboard.Plug` was initialized to a value, should return the value that :py:attr:`~corelay.plugboard.Plug.obj`
        initialized with.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, obj='obj', default='default')

        assert plug.obj == 'obj'

    @staticmethod
    def test_obj_hierarchy_default() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.obj` property while the :py:class:`~corelay.plugboard.Slot` and the
        :py:class:`~corelay.plugboard.Plug` have a default values, should return the :py:attr:`~corelay.plugboard.Plug.default` value of the
        :py:class:`~corelay.plugboard.Plug`.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, default='default')

        assert plug.obj == 'default'

    @staticmethod
    def test_obj_hierarchy_fallback() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.obj` property while the :py:class:`~corelay.plugboard.Slot` has a default value,
        should return the :py:attr:`~corelay.plugboard.Plug.default` value of the :py:class:`~corelay.plugboard.Slot`.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)

        assert plug.obj == 'fallback'

    @staticmethod
    def test_default_hierarchy_obj() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.default` property while the :py:class:`~corelay.plugboard.Slot` and the
        :py:class:`~corelay.plugboard.Plug` have a default values and the the :py:attr:`~corelay.plugboard.Plug.obj` property of the
        :py:class:`~corelay.plugboard.Plug` was initialized to a value, should return the :py:attr:`~corelay.plugboard.Plug.default` value of the
        :py:class:`~corelay.plugboard.Plug`.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, obj='obj', default='default')

        assert plug.default == 'default'

    @staticmethod
    def test_default_hierarchy_default() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.default` property while the :py:class:`~corelay.plugboard.Slot` and the
        :py:class:`~corelay.plugboard.Plug` have a default values, should return the :py:attr:`~corelay.plugboard.Plug.default` value of the
        :py:class:`~corelay.plugboard.Plug`.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, default='default')

        assert plug.default == 'default'

    @staticmethod
    def test_default_hierarchy_fallback() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.default` property while the :py:class:`~corelay.plugboard.Slot` has a default
        value, should return the :py:attr:`~corelay.plugboard.Plug.default` value of the :py:class:`~corelay.plugboard.Slot`.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)

        assert plug.default == 'fallback'

    @staticmethod
    def test_fallback_hierarchy_obj() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.fallback` property while the :py:class:`~corelay.plugboard.Slot` and the
        :py:class:`~corelay.plugboard.Plug` have a default values and the the :py:attr:`~corelay.plugboard.Plug.obj` property of the
        :py:class:`~corelay.plugboard.Plug` was initialized to a value, should return the :py:attr:`~corelay.plugboard.Plug.default` value of the
        :py:class:`~corelay.plugboard.Slot`.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, obj='obj', default='default')

        assert plug.fallback == 'fallback'

    @staticmethod
    def test_fallback_hierarchy_default() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.fallback` property while the :py:class:`~corelay.plugboard.Slot` and the
        :py:class:`~corelay.plugboard.Plug` have a default values, should return the :py:attr:`~corelay.plugboard.Plug.default` value of the
        :py:class:`~corelay.plugboard.Slot`.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, default='default')

        assert plug.fallback == 'fallback'

    @staticmethod
    def test_fallback_hierarchy_fallback() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.fallback` property while the :py:class:`~corelay.plugboard.Slot` has a default
        value, should return the :py:attr:`~corelay.plugboard.Plug.default` value of the :py:class:`~corelay.plugboard.Slot`.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)

        assert plug.fallback == 'fallback'

    @staticmethod
    def test_hierarchy_none() -> None:
        """Tests that accessing the :py:attr:`~corelay.plugboard.Plug.default` and :py:attr:`~corelay.plugboard.Plug.fallback` properties while the
        neither the :py:class:`~corelay.plugboard.Slot` nor the :py:class:`~corelay.plugboard.Plug` have a default value, but the
        :py:attr:`~corelay.plugboard.Plug.obj` property of the :py:class:`~corelay.plugboard.Plug` was initialized to a value returns :py:obj:`None`.
        """

        slot = Slot(dtype=str)
        plug = Plug(slot, obj='obj')

        assert plug.default is None
        assert plug.fallback is None

    @staticmethod
    def test_delete_hierarchy() -> None:
        """Tests that deleting the value of the :py:attr:`~corelay.plugboard.Plug.obj` property with the :py:class:`~corelay.plugboard.Plug` having a
        :py:attr:`~corelay.plugboard.Plug.default` set, returns the :py:attr:`~corelay.plugboard.Plug.default` value of the
        :py:class:`~corelay.plugboard.Plug`.
        """

        slot = Slot(dtype=str)
        plug = Plug(slot, default='default', obj='obj')
        del plug.obj

        assert plug.obj == 'default'

    @staticmethod
    def test_delete_hierarchy_last() -> None:
        """Tests that deleting the value of the :py:attr:`~corelay.plugboard.Plug.default` property of the :py:class:`~corelay.plugboard.Plug` fails,
        when the :py:class:`~corelay.plugboard.Slot` has no :py:attr:`~corelay.plugboard.Plug.default` value and the value of the
        :py:attr:`~corelay.plugboard.Plug.obj` property of the :py:class:`~corelay.plugboard.Plug` was previously deleted.
        """

        slot = Slot(dtype=str)
        plug = Plug(slot, default='default', obj='obj')
        del plug.obj

        with pytest.raises(TypeError):
            del plug.default

    @staticmethod
    def test_obj_set() -> None:
        """Tests that setting the :py:attr:`~corelay.plugboard.Plug.obj` property of a :py:class:`~corelay.plugboard.Plug` to a value that is
        consistent with the data type of the :py:class:`~corelay.plugboard.Slot` succeeds.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        plug.obj = 'obj'

    @staticmethod
    def test_obj_set_inconsistent() -> None:
        """Tests that setting the :py:attr:`~corelay.plugboard.Plug.obj` property of a :py:class:`~corelay.plugboard.Plug` to a value that is not
        consistent with the data type of the :py:class:`~corelay.plugboard.Slot` fails.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)

        with pytest.raises(TypeError):
            plug.obj = 15

    @staticmethod
    def test_obj_del() -> None:
        """Tests that deleting the value of the :py:attr:`~corelay.plugboard.Plug.obj` property of a :py:class:`~corelay.plugboard.Plug` that is
        associated with a :py:class:`~corelay.plugboard.Slot` that has a default value succeeds.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        plug.obj = 'obj'
        del plug.obj

    @staticmethod
    def test_default_set() -> None:
        """Tests that setting the :py:attr:`~corelay.plugboard.Plug.default` property of a :py:class:`~corelay.plugboard.Plug` to a value that is
        consistent with the data type of the :py:class:`~corelay.plugboard.Slot` succeeds.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        plug.default = 'default'

    @staticmethod
    def test_default_set_inconsistent() -> None:
        """Tests that setting the :py:attr:`~corelay.plugboard.Plug.default` property of a :py:class:`~corelay.plugboard.Plug` to a value that is not
        consistent with the data type of the :py:class:`~corelay.plugboard.Slot` fails.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)

        with pytest.raises(TypeError):
            plug.default = 15

    @staticmethod
    def test_default_del() -> None:
        """Tests that deleting the value of the :py:attr:`~corelay.plugboard.Plug.default` property of a :py:class:`~corelay.plugboard.Plug` that is
        associated with a :py:class:`~corelay.plugboard.Slot` that has a :py:attr:`~corelay.plugboard.Plug.default` value succeeds.
        """

        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        plug.default = 'default'

        del plug.default

    @staticmethod
    def test_not_optional() -> None:
        """Tests that if neither the :py:class:`~corelay.plugboard.Plug` nor the :py:class:`~corelay.plugboard.Slot` have a
        :py:attr:`~corelay.plugboard.Plug.default` value, the :py:attr:`~corelay.plugboard.Plug.obj` value is not optional.
        """

        slot = Slot(dtype=str)
        plug = Plug(slot, obj='obj')

        assert not plug.optional

    @staticmethod
    def test_optional() -> None:
        """Tests that if the :py:attr:`~corelay.plugboard.Plug.default` value of the :py:class:`~corelay.plugboard.Plug` is set, the
        :py:attr:`~corelay.plugboard.Plug.obj` property is optional.
        """

        slot = Slot(dtype=str)
        plug = Plug(slot, default='default', obj='obj')

        assert plug.optional

    @staticmethod
    def test_slot_set_consistent() -> None:
        """Tests that assigning a new :py:class:`~corelay.plugboard.Slot` to a :py:class:`~corelay.plugboard.Plug`, that has the correct data type
        succeeds.
        """

        slot = Slot(dtype=object)
        plug = Plug(slot, obj='default')
        plug.slot = Slot(dtype=str)

    @staticmethod
    def test_slot_set_inconsistent() -> None:
        """Tests that assigning a new :py:class:`~corelay.plugboard.Slot` to a :py:class:`~corelay.plugboard.Plug`, that does not have the correct
        data type fails.
        """

        slot = Slot(dtype=object)
        plug = Plug(slot, obj='default')

        with pytest.raises(TypeError):
            plug.slot = Slot(dtype=int)

    @staticmethod
    def test_slot_set_no_default() -> None:
        """Tests that assigning a new :py:class:`~corelay.plugboard.Slot` to a :py:class:`~corelay.plugboard.Plug`, that does not have a default
        value, and neither the original :py:class:`~corelay.plugboard.Slot` nor the :py:class:`~corelay.plugboard.Plug` have a
        :py:attr:`~corelay.plugboard.Plug.default` or :py:attr:`~corelay.plugboard.Plug.obj` value fails.
        """

        slot = Slot(dtype=object, default='fallback')
        plug = Plug(slot)

        with pytest.raises(TypeError):
            plug.slot = Slot(dtype=str)


class TestPlugboard:
    """Contains unit tests for the :py:class:`~corelay.plugboard.Plugboard` class."""

    @staticmethod
    def test_init() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plugboard` without anything set succeeds."""

        Plugboard()

    @staticmethod
    def test_init_unknown_kwargs() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plugboard` with unknown keyword arguments fails."""

        with pytest.raises(TypeError):
            Plugboard(stuff=19)

    @staticmethod
    def test_init_args() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plugboard` with any positional arguments fails."""

        with pytest.raises(TypeError):
            Plugboard(19)  # type: ignore[call-arg] # pylint: disable=too-many-function-args

    @staticmethod
    def test_init_assign() -> None:
        """Tests that instantiating a :py:class:`~corelay.plugboard.Plugboard` with keyword arguments identifying slots, sets the values of those
        slots.
        """

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard(my_slot=19)

        assert plugboard.my_slot == 19

    @staticmethod
    def test_default_get() -> None:
        """Tests that accessing the default value of a :py:class:`~corelay.plugboard.Slot` in a :py:class:`~corelay.plugboard.Plugboard` that has an
        explicit :py:attr:`~corelay.plugboard.Plug.obj` value succeeds.
        """

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard(my_slot=19)

        assert plugboard.default.my_slot == 15

    @staticmethod
    def test_default_set() -> None:
        """Tests that setting the default value of a :py:class:`~corelay.plugboard.Slot` in a :py:class:`~corelay.plugboard.Plugboard` that is
        consistent with the data type of the :py:class:`~corelay.plugboard.Slot` succeeds.
        """

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard()
        plugboard.default.my_slot = 17

        assert plugboard.my_slot == 17

    @staticmethod
    def test_default_set_dict() -> None:
        """Tests that setting the default value of a :py:class:`~corelay.plugboard.Slot` in a :py:class:`~corelay.plugboard.Plugboard` that is
        consistent with the data type of the :py:class:`~corelay.plugboard.Slot` using a :py:class:`dict` succeeds.
        """

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard()
        plugboard.default = {'my_slot': 17}

        assert plugboard.my_slot == 17

    @staticmethod
    def test_default_set_dict_wrong() -> None:
        """Tests that setting the default value of a :py:class:`~corelay.plugboard.Slot` in a :py:class:`~corelay.plugboard.Plugboard` using anything
        but a :py:class:`dict` or the attribute accessor of the :py:class:`~corelay.plugboard.Slot` fails.
        """

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard()

        with pytest.raises(TypeError):
            plugboard.default = 15  # type: ignore[assignment]

    @staticmethod
    def test_default_del() -> None:
        """Tests that deleting the default value of a :py:class:`~corelay.plugboard.Slot` in a :py:class:`~corelay.plugboard.Plugboard` succeeds and
        reverts back to the default value of the :py:class:`~corelay.plugboard.Slot`.
        """

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard()
        plugboard.default.my_slot = 17
        del plugboard.default.my_slot

        assert plugboard.my_slot == 15

    @staticmethod
    def test_default_set_influence_obj() -> None:
        """Tests that setting the default value of a :py:class:`~corelay.plugboard.Slot` in a :py:class:`~corelay.plugboard.Plugboard` does not
        influence the value of the :py:class:`~corelay.plugboard.Slot`.
        """

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard(my_slot=19)
        plugboard.default.my_slot = 17

        assert plugboard.my_slot == 19

    @staticmethod
    def test_default_dir() -> None:
        """Tests that the :py:attr:`~corelay.plugboard.Plug.default` property of the plugboard is a :py:class:`dict`, that contains entries for all of
        its slots."""

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard()

        assert set(plugboard.collect(Slot)) == set(dir(plugboard.default))

    @staticmethod
    def test_update_defaults() -> None:
        """Tests that setting the default value of a :py:class:`~corelay.plugboard.Slot` in a :py:class:`~corelay.plugboard.Plugboard` that is
        consistent with the data type of the :py:class:`~corelay.plugboard.Slot` using the
        :py:meth:`Plugboard.update_defaults <corelay.plugboard.Plugboard.update_defaults>` method succeeds.
        """

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard()
        plugboard.update_defaults(my_slot=17)

        assert plugboard.my_slot == 17

    @staticmethod
    def test_reset_defaults() -> None:
        """Tests that resetting the default value of a :py:class:`~corelay.plugboard.Slot` in a :py:class:`~corelay.plugboard.Plugboard` succeeds."""

        class MyPlugboard(Plugboard):
            """A test plugboard that holds a single :py:class:`~corelay.plugboard.Slot`."""

            my_slot = Slot(dtype=int, default=15)
            """A test slot."""

        plugboard = MyPlugboard()
        plugboard.default.my_slot = 17
        plugboard.reset_defaults()

        assert plugboard.my_slot == 15
