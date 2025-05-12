"""A module that contains unit tests for the :py:mod:`corelay.tracker` module."""

import pytest

from corelay.tracker import Tracker


@pytest.fixture(name='tracker_type', scope='module')
def get_tracked_fixture() -> type[Tracker]:
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
        attr_5 = 'pear'
        attr_6 = str

    return SubTracked


@pytest.fixture(name='values', scope='module')
def get_values_fixture() -> dict[str, int | str | type]:
    """Generates a list of values, that can be used to test the collection of values.

    Returns:
        dict[str, int | str | type]: Returns a :py:class:`dict` with values that can be used to test the collection of values.
    """

    result: dict[str, int | str | type] = {
        'attr_1': 42,
        'attr_2': 'apple',
        'attr_3': object,
        'attr_4': 15,
        'attr_5': 'pear',
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


class TestTracker:
    """Contains unit tests for the :py:class:`~corelay.tracker.Tracker` class."""

    @staticmethod
    def test_collect(tracker_type: type[Tracker]) -> None:
        """Tests that parameters of different types are collected correctly.

        Args:
            tracker_type (type[Tracker]): The type of the tracker to be used for collecting the parameter values.
        """

        assert tracker_type.collect(int) == {'attr_1': 42, 'attr_4': 15}
        assert tracker_type.collect(str) == {'attr_2': 'apple', 'attr_5': 'pear'}
        assert tracker_type.collect(type) == {'attr_3': object, 'attr_6': str}

    @staticmethod
    def test_collect_multiple(tracker_type: type[Tracker], values: dict[str, int | str | type]) -> None:
        """Tests that collecting parameters of multiple data types succeeds.

        Args:
            tracker_type (type[Tracker]): The type of the tracker to be used for collecting the parameter values.
            values (dict[str, int | str | type]): The expected values to be collected.
        """

        assert tracker_type.collect((int, str, type)) == values

    @staticmethod
    def test_collect_attr(instance: Tracker, attributes: dict[str, str]) -> None:
        """Tests that collecting instance attribute values of owner attributes succeeds.

        Args:
            instance (Tracker): The instance of the tracker to be used for collecting the parameter values.
            attributes (dict[str, str]): The expected values to be collected.
        """

        assert instance.collect_attr((int, str, type)) == attributes
