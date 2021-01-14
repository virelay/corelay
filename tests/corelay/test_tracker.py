"""Test tracker functionalities.

"""

import pytest

from corelay.tracker import Tracker


@pytest.fixture(scope='module')
def tracked():
    """Tracked class fixture"""
    class SubTracked(Tracker):
        """Tracker Subclass"""
        attr_1 = 42
        attr_2 = 'apple'
        attr_3 = object
        attr_4 = 15
        attr_5 = 'pear'
        attr_6 = str
    return SubTracked


@pytest.fixture(scope='module')
def values():
    """Fixute of list of values, how they were written in the tracked fixture class."""
    result = dict(
        attr_1=42,
        attr_2='apple',
        attr_3=object,
        attr_4=15,
        attr_5='pear',
        attr_6=str
    )
    return result


@pytest.fixture(scope='module')
# pylint: disable=unused-argument
def attributes(tracked):
    """Fixture for some new values for Tracked Parameters"""
    result = dict(
        attr_1='value_1',
        attr_2='value_2',
        attr_3='value_3',
        attr_4='value_4',
        attr_5='value_5',
        attr_6='value_6',
    )
    return result


@pytest.fixture(scope='module')
def instance(tracked, attributes):
    """Fixture instance with new attribute values"""
    result = tracked()
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class TestTracker:
    """Test class for Tracker"""
    @staticmethod
    def test_collect(tracked):
        """Types should be collected correctly and in order."""
        assert tracked.collect(int) == dict(attr_1=42, attr_4=15)
        assert tracked.collect(str) == dict(attr_2='apple', attr_5='pear')
        assert tracked.collect(type) == dict(attr_3=object, attr_6=str)

    @staticmethod
    def test_collect_multiple(tracked, values):
        """Collecting multiple dtypes should succeed"""
        assert tracked.collect((int, str, type)) == values

    @staticmethod
    def test_collect_attr(instance, attributes):
        """Collecting instance attribute values by owner attributes should succeed."""
        assert instance.collect_attr((int, str, type)) == attributes
