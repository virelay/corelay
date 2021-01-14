"""Tests for corelay/plugboard.py"""
import pytest

from corelay.plugboard import Slot, Plug, Plugboard


class TestSlot:
    """Test class for Slot"""
    @staticmethod
    def test_init():
        """Slot should successfully instatiate in any case"""
        Slot()

    @staticmethod
    def test_init_consistent_args():
        """If arguments are consistent, Slot should successfully instatiate"""
        Slot(dtype=int, default=5)

    @staticmethod
    def test_init_inconsistent_args():
        """If arguments are inconsistent, Slot should raise TypeError when instatiating"""
        with pytest.raises(TypeError):
            Slot(dtype=str, default=5)

    @staticmethod
    def test_init_unknown_args():
        """When unknown arguments are passed, Slot should raise TypeError when instatiating"""
        with pytest.raises(TypeError):
            Slot(monkey='banana')

    @staticmethod
    def test_init_class_name():
        """When instatiated in a class, the __name__ parameter of Slot should be set accordingly"""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot()
        assert SlotHolder.my_slot.__name__ == 'my_slot'

    @staticmethod
    def test_init_instance_default():
        """When instatiated in a class and accessed in an instance, with only default set, the default value should be
        returned.
        """
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(default=42)
        slot_holder = SlotHolder()
        assert slot_holder.my_slot == 42

    @staticmethod
    def test_instance_get():
        """Getting a value after setting it should succeed."""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(dtype=int)
        slot_holder = SlotHolder()
        slot_holder.my_slot = 15
        assert slot_holder.my_slot == 15

    @staticmethod
    def test_instance_get_no_default():
        """When instatiated in a class and accessed in an instance, without anything set, accessing the value should
        raise a TypeError.
        """
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot()
        slot_holder = SlotHolder()
        with pytest.raises(TypeError):
            # pylint: disable=pointless-statement
            slot_holder.my_slot

    @staticmethod
    def test_instance_set():
        """Setting a value without a default value afterwards should succeed."""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(dtype=int)
        slot_holder = SlotHolder()
        slot_holder.my_slot = 15

    @staticmethod
    def test_instance_set_wrong_dtype():
        """Setting a default value afterwards with a wrong dtype should not succeed."""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(str)
        slot_holder = SlotHolder()
        with pytest.raises(TypeError):
            slot_holder.my_slot = 15

    @staticmethod
    def test_instance_delete_unchanged():
        """Setting a value and deleting it afterwards with a set default value should return the default value"""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(int, 42)
        slot_holder = SlotHolder()
        slot_holder.my_slot = 15
        del slot_holder.my_slot
        assert slot_holder.my_slot == 42

    @staticmethod
    def test_instance_delete_without_default():
        """Setting a value and deleting it afterwards with a set default value should raise an Exception."""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(int)
        slot_holder = SlotHolder()
        slot_holder.my_slot = 15
        with pytest.raises(TypeError):
            del slot_holder.my_slot

    @staticmethod
    def test_class_set_dtype():
        """Setting a consistent dtype should succeed."""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(dtype=object, default=15)
        SlotHolder.dtype = int

    @staticmethod
    def test_class_set_dtype_inconsistent():
        """Setting an inconsistent dtype should fail."""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(dtype=object, default=15)
        with pytest.raises(TypeError):
            SlotHolder.my_slot.dtype = str

    @staticmethod
    def test_class_optional():
        """Slots with default values should be optional."""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(dtype=int, default=15)
        assert SlotHolder.my_slot.optional

    @staticmethod
    def test_class_not_optional():
        """Slots without default values should not be optional."""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(dtype=int)
        assert not SlotHolder.my_slot.optional

    @staticmethod
    def test_class_call():
        """Calling a Slot should yield a Plug associated with the Slot"""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(dtype=int, default=15)
        assert SlotHolder.my_slot is SlotHolder.my_slot().slot

    @staticmethod
    def test_class_call_obj():
        """Calling a Slot with obj-argument should yield a Plug with that obj set"""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(dtype=int)
        assert SlotHolder.my_slot(obj=15).obj == 15

    @staticmethod
    def test_class_call_default():
        """Calling a Slot with default-argument should yield a Plug with that default set"""
        class SlotHolder:
            """Holds a single Slot"""
            my_slot = Slot(dtype=int)
        assert SlotHolder.my_slot(default=15).default == 15


class TestPlug:
    """Test class for Plug"""
    @staticmethod
    def test_init_with_slot_default():
        """Instatiating a Plug with a slot's default value should succeed."""
        slot = Slot(dtype=int, default=10)
        Plug(slot)

    @staticmethod
    def test_init_no_slot_default():
        """Instatiating a Plug without a slot's default value should fail."""
        slot = Slot(dtype=int)
        with pytest.raises(TypeError):
            Plug(slot)

    @staticmethod
    def test_init_consistent():
        """Instatiating a Plug with obj and default set as the correct slot's dtype should succeed."""
        slot = Slot(dtype=int)
        Plug(slot, obj=15, default=16)

    @staticmethod
    def test_init_consistent_obj():
        """Instatiating a Plug with obj set as the correct slot's dtype should succeed."""
        slot = Slot(dtype=int)
        Plug(slot, obj=15)

    @staticmethod
    def test_init_consistent_default():
        """Instatiating a Plug with default set as the correct slot's dtype should succeed."""
        slot = Slot(dtype=int)
        Plug(slot, default=15)

    @staticmethod
    def test_init_inconsistent_obj():
        """Instatiating a Plug with obj not set as slot's dtype should fail."""
        slot = Slot(dtype=str)
        with pytest.raises(TypeError):
            Plug(slot, obj=15)

    @staticmethod
    def test_init_inconsistent_default():
        """Instatiating a Plug with obj not set as slot's dtype should fail."""
        slot = Slot(dtype=str)
        with pytest.raises(TypeError):
            Plug(slot, default=15)

    @staticmethod
    def test_obj_hierachy_obj():
        """Accessing obj with slot default, plug default and obj set should return obj."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, obj='obj', default='default')
        assert plug.obj == 'obj'

    @staticmethod
    def test_obj_hierachy_default():
        """Accessing obj with slot default and plug default set should return default."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, default='default')
        assert plug.obj == 'default'

    @staticmethod
    def test_obj_hierachy_fallback():
        """Accessing obj with only slot default set should return slot default."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        assert plug.obj == 'fallback'

    @staticmethod
    def test_default_hierachy_obj():
        """Accessing default with slot default, default and obj set should return default."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, default='default')
        assert plug.default == 'default'

    @staticmethod
    def test_default_hierachy_default():
        """Accessing default with slot default and plug default set should return default."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, default='default')
        assert plug.default == 'default'

    @staticmethod
    def test_default_hierachy_fallback():
        """Accessing default with only slot default set should return slot default."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        assert plug.default == 'fallback'

    @staticmethod
    def test_fallback_hierachy_obj():
        """Accessing fallback with slot default, default and obj set should return slot default."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, default='default')
        assert plug.fallback == 'fallback'

    @staticmethod
    def test_fallback_hierachy_default():
        """Accessing fallback with slot default and plug default set should return slot default."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot, default='default')
        assert plug.fallback == 'fallback'

    @staticmethod
    def test_fallback_hierachy_fallback():
        """Accessing fallback with only slot default set should return slot default."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        assert plug.fallback == 'fallback'

    @staticmethod
    def test_hierachy_none():
        """Accessing any of default and fallback with only obj set should return None."""
        slot = Slot(dtype=str)
        plug = Plug(slot, obj='obj')
        assert plug.default is None
        assert plug.fallback is None

    @staticmethod
    def test_delete_hierachy():
        """Deleting obj with default set should return default."""
        slot = Slot(dtype=str)
        plug = Plug(slot, default='default', obj='obj')
        del plug.obj
        assert plug.obj == 'default'

    @staticmethod
    def test_delete_hierachy_last():
        """Deleting such that all obj, default and fallback are None should fail."""
        slot = Slot(dtype=str)
        plug = Plug(slot, default='default', obj='obj')
        del plug.obj
        with pytest.raises(TypeError):
            del plug.default

    @staticmethod
    def test_obj_set():
        """Setting obj consistently should succeed."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        plug.obj = 'obj'

    @staticmethod
    def test_obj_set_inconsistent():
        """Setting obj inconsistently should fail."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        with pytest.raises(TypeError):
            plug.obj = 15

    @staticmethod
    def test_obj_del():
        """Deleting obj with slot default should succeed."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        plug.obj = 'obj'
        del plug.obj

    @staticmethod
    def test_default_set():
        """Setting default consistently should succeed."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        plug.default = 'default'

    @staticmethod
    def test_default_set_inconsistent():
        """Setting default inconsistently should fail."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        with pytest.raises(TypeError):
            plug.default = 15

    @staticmethod
    def test_default_del():
        """Deleting default with slot default should succeed."""
        slot = Slot(dtype=str, default='fallback')
        plug = Plug(slot)
        plug.default = 'default'
        del plug.default

    @staticmethod
    def test_not_optional():
        """If neither default, nor slot default are set, object should not be optional."""
        slot = Slot(dtype=str)
        plug = Plug(slot, obj='obj')
        assert not plug.optional

    @staticmethod
    def test_optional():
        """If default is set, object should be optional."""
        slot = Slot(dtype=str)
        plug = Plug(slot, default='default', obj='obj')
        assert plug.optional

    @staticmethod
    def test_slot_set_consistent():
        """Assigning a new slot with the correct dtype should succeed."""
        slot = Slot(dtype=object)
        plug = Plug(slot, obj='default')
        plug.slot = Slot(dtype=str)

    @staticmethod
    def test_slot_set_inconsistent():
        """Assigning a new slot without the correct dtype should fail."""
        slot = Slot(dtype=object)
        plug = Plug(slot, obj='default')
        with pytest.raises(TypeError):
            plug.slot = slot(dtype=int)

    @staticmethod
    def test_slot_set_no_default():
        """Assigning a new slot without any other but the slot default value set should fail."""
        slot = Slot(dtype=object, default='fallback')
        plug = Plug(slot)
        with pytest.raises(TypeError):
            plug.slot = slot(dtype=str)


class TestPlugboard:
    """Test class for Plugboard"""
    @staticmethod
    def test_init():
        """Instatiating a Plugboard without anything set should suceed."""
        Plugboard()

    @staticmethod
    def test_init_unknown_kwargs():
        """Instatiating a Plugboard with unknown kwargs should fail."""
        with pytest.raises(TypeError):
            Plugboard(stuff=19)

    @staticmethod
    def test_init_args():
        """Instatiating a Plugboard with any positional args should fail."""
        with pytest.raises(TypeError):
            # pylint: disable=too-many-function-args
            Plugboard(19)

    @staticmethod
    def test_init_assign():
        """Instatiating a Plugboard with kwargs identifying Slots should set those."""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard(my_slot=19)
        assert plugboard.my_slot == 19

    @staticmethod
    def test_default_get():
        """Accessing a Plugboard's Slot default values with an explicit obj value should succeed."""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard(my_slot=19)
        assert plugboard.default.my_slot == 15

    @staticmethod
    def test_default_set():
        """Setting a Plugboard's Slot default values should succeed."""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard()
        plugboard.default.my_slot = 17
        assert plugboard.my_slot == 17

    @staticmethod
    def test_default_set_dict():
        """Setting a Plugboard's Slot default values using a dict should succeed."""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard()
        plugboard.default = dict(my_slot=17)
        assert plugboard.my_slot == 17

    @staticmethod
    def test_default_set_dict_wrong():
        """Setting a Plugboard's Slot default values using anything but a dict should fail."""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard()
        with pytest.raises(TypeError):
            plugboard.default = 15

    @staticmethod
    def test_default_del():
        """Deleting a plugboard's slot default values should succeed."""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard()
        plugboard.default.my_slot = 17
        del plugboard.default.my_slot
        assert plugboard.my_slot == 15

    @staticmethod
    def test_default_set_influence_obj():
        """Setting a Plugboard's Slot default values should not influence the Plug obj value."""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard(my_slot=19)
        plugboard.default.my_slot = 17
        assert plugboard.my_slot == 19

    @staticmethod
    def test_default_dir():
        """The default __dir__ should contain all Slots"""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard()
        assert set(plugboard.collect(Slot)) == set(dir(plugboard.default))

    @staticmethod
    def test_update_defaults():
        """Setting a Plugboard's Slot default values using update_defaults should succeed."""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard()
        plugboard.update_defaults(my_slot=17)
        assert plugboard.my_slot == 17

    @staticmethod
    def test_reset_defaults():
        """Resetting a plugboard's slot default values should succeed."""
        class MyPlugboard(Plugboard):
            """Custom Plugboard"""
            my_slot = Slot(dtype=int, default=15)
        plugboard = MyPlugboard()
        plugboard.default.my_slot = 17
        plugboard.reset_defaults()
        assert plugboard.my_slot == 15
