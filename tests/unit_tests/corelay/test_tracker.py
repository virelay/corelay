"""A module that contains unit tests for the :py:mod:`corelay.tracker` module."""

import pytest

from corelay.base import Param
from corelay.plugboard import Slot
from corelay.tracker import Tracker


@pytest.fixture(name='tracker_type', scope='module')
def get_tracker_type_fixture() -> type[Tracker]:
    """Creates a sub-class of :py:class:`~corelay.tracker.Tracker` with some attributes.

    Returns:
        type[Tracker]: Returns a sub-class of :py:class:`~corelay.tracker.Tracker` with some attributes.
    """

    class SubTracked(Tracker):
        """A sub-class of :py:class:`~corelay.tracker.Tracker` with some attributes."""

        attr_1 = 42
        attr_2 = 'apple'
        attr_3 = object
        attr_4 = 15
        attr_5 = None
        attr_6 = str

    return SubTracked


@pytest.fixture(name='values', scope='module')
def get_values_fixture() -> dict[str, int | str | type | None]:
    """Generates a list of values, that can be used to test the collection of values.

    Returns:
        dict[str, int | str | type | None]: Returns a :py:class:`dict` with values that can be used to test the collection of values.
    """

    result: dict[str, int | str | type | None] = {
        'attr_1': 42,
        'attr_2': 'apple',
        'attr_3': object,
        'attr_4': 15,
        'attr_5': None,
        'attr_6': str
    }
    return result


@pytest.fixture(name='attributes', scope='module')
def get_attributes_fixture() -> dict[str, str]:
    """Generates some new values for tracked parameters.

    Returns:
        dict[str, str]: Returns a :py:class:`dict` with new values for tracked parameters.
    """

    result: dict[str, str] = {
        'attr_1': 'value_1',
        'attr_2': 'value_2',
        'attr_3': 'value_3',
        'attr_4': 'value_4',
        'attr_5': 'value_5',
        'attr_6': 'value_6'
    }
    return result


@pytest.fixture(name='instance', scope='module')
def get_instance_fixture(tracker_type: type[Tracker], attributes: dict[str, str]) -> Tracker:
    """Generates an instance of a tracker and adds new attributes to it to track.

    Args:
        tracker_type (type[Tracker]): The type of the tracker to be created.
        attributes (dict[str, str]): The attributes to be added to the tracker instance.

    Returns:
        Tracker: Returns an instance of the tracker with the new attributes.
    """

    result = tracker_type()
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class TestMetaTracker:
    """Contains unit tests for the :py:class:`~corelay.tracker.MetaTracker` meta class."""

    @staticmethod
    def test_warning_is_raised_when_old_slot_syntax_is_used() -> None:
        """Tests that a warning is raised when the old slot syntax is used."""

        with pytest.warns(DeprecationWarning, match='The Param "param" was declared using the old syntax of declaring slots, which is deprecated.'):

            class OldSlotSyntaxTracker(Tracker):
                """A test tracker that uses the old slot syntax."""

                param = Param(int, default=42)
                """A slot that is declared using the old syntax."""

            _ = OldSlotSyntaxTracker()

    @staticmethod
    def test_is_slot_check_with_complex_inheritance_hierarchy() -> None:
        """Tests that the :py:meth:`~corelay.tracker.MetaTracker.__new__.is_slot` method works correctly with a complex inheritance hierarchy."""

        class Test:
            """A base class that is not a :py:class:`~corelay.plugboard.Slot`."""

        class Baz(Slot):
            """A base class that is a :py:class:`~corelay.plugboard.Slot`."""

        class Bar(Baz, Test):
            """A class that inherits from :py:class:`Baz` and therefore is a :py:class:`~corelay.plugboard.Slot` indirectly."""

        class Foo(Test):
            """A class that only inherits from :py:class:`Test` and therefore is not a :py:class:`~corelay.plugboard.Slot`."""

        class FooBar(Foo, Bar):
            """A class that inherits from :py:class:`Foo` and :py:class:`Bar`, and therefore is a :py:class:`~corelay.plugboard.Slot` indirectly."""

        with pytest.warns(
            DeprecationWarning,
            match='The FooBar "foo_bar" was declared using the old syntax of declaring slots, which is deprecated.'
        ):

            class OldSlotSyntaxTracker(Tracker):
                """A test tracker that uses the old slot syntax."""

                foo_bar = FooBar(int, default=42)
                """A slot that is declared using the old syntax."""

            _ = OldSlotSyntaxTracker()

class TestTracker:
    """Contains unit tests for the :py:class:`~corelay.tracker.Tracker` class."""

    @staticmethod
    def test_collect(tracker_type: type[Tracker]) -> None:
        """Tests that parameters of different types are collected correctly.

        Args:
            tracker_type (type[Tracker]): The type of the tracker to be used for collecting the parameter values.
        """

        assert tracker_type.collect(int) == {'attr_1': 42, 'attr_4': 15}
        assert tracker_type.collect(str) == {'attr_2': 'apple'}
        assert tracker_type.collect(type) == {'attr_3': object, 'attr_6': str}
        assert tracker_type.collect(type(None)) == {'attr_5': None}

    @staticmethod
    def test_get(tracker_type: type[Tracker]) -> None:
        """Tests that parameters of different types are collected correctly.

        Args:
            tracker_type (type[Tracker]): The type of the tracker to be used for collecting the parameter values.
        """

        assert tracker_type.get('attr_1') == 42
        assert tracker_type.get('attr_2') == 'apple'
        assert tracker_type.get('attr_3') == object
        assert tracker_type.get('attr_4') == 15
        assert tracker_type.get('attr_5') is None
        assert tracker_type.get('attr_6') == str

        with pytest.raises(AttributeError):
            tracker_type.get('non_existent_attribute')

    @staticmethod
    def test_collect_multiple(tracker_type: type[Tracker], values: dict[str, int | str | type | None]) -> None:
        """Tests that collecting parameters of multiple data types succeeds.

        Args:
            tracker_type (type[Tracker]): The type of the tracker to be used for collecting the parameter values.
            values (dict[str, int | str | type | None]): The expected values to be collected.
        """

        assert tracker_type.collect((int, str, type, type(None))) == values

    @staticmethod
    def test_collect_attr(instance: Tracker, attributes: dict[str, str]) -> None:
        """Tests that collecting instance attribute values of owner attributes succeeds.

        Args:
            instance (Tracker): The instance of the tracker to be used for collecting the parameter values.
            attributes (dict[str, str]): The expected values to be collected.
        """

        assert instance.collect_attr((int, str, type, type(None))) == attributes

    @staticmethod
    def test_get_attr(instance: Tracker, attributes: dict[str, str]) -> None:
        """Tests that collecting instance attribute values of owner attributes succeeds.

        Args:
            instance (Tracker): The instance of the tracker to be used for collecting the parameter values.
            attributes (dict[str, str]): The expected values to be collected.
        """

        for attribute_name, attribute_value in attributes.items():
            assert instance.get_attr(attribute_name) == attribute_value

        with pytest.raises(AttributeError):
            instance.get_attr('non_existent_attribute')
