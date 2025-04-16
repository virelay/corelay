"""Test corelay.utils

"""

import pytest

from corelay.utils import import_or_stub, Iterable, zip_equal


def test_conditional_import():
    """Test conditional import which fails only when actually using the imported module."""
    non_existing_module = import_or_stub('non_existing_module')
    non_existing_function = import_or_stub('non_existing_module', 'non_existing_function')
    re = import_or_stub('re')
    findall = import_or_stub('re', 'findall')

    with pytest.raises(RuntimeError):
        non_existing_module.f()
    with pytest.raises(RuntimeError):
        non_existing_function()
    re.findall('aba', 'a')
    findall('aba', 'a')


def test_conditional_import_of_multiple_functions():
    """Test conditional importing of multiple function from the same module."""
    match, fullmatch = import_or_stub('non_existing_module', ('match', 'fullmatch'))
    with pytest.raises(RuntimeError):
        match('aba', 'a')
    with pytest.raises(RuntimeError):
        fullmatch('aba', 'a')

    match, fullmatch = import_or_stub('re', ('match', 'fullmatch'))
    match('aba', 'a')
    fullmatch('aba', 'a')

    with pytest.raises(ImportError):
        _, _ = import_or_stub('re', ('findall', 'non_existing'))


class TestIterable:
    """Test class for Iterable"""
    @staticmethod
    def test_instance_check_all_member_type():
        """Iterable without member type should be any Iterable"""
        assert isinstance([1, 'a', 3.5], Iterable)

    @staticmethod
    def test_instance_check_single_member_type_positive():
        """Iterable with single member type should succeed"""
        assert isinstance([1, 2, 3], Iterable[int])

    @staticmethod
    def test_instance_check_single_member_type_negative():
        """Iterable with single member type and wrong input should fail"""
        assert not isinstance([1, 2, 'a'], Iterable[int])

    @staticmethod
    def test_instance_check_multiple_member_type_positive():
        """Iterable with multiple member types should succeed"""
        assert isinstance([1, 2, 0.5], Iterable[int, float])

    @staticmethod
    def test_instance_check_multiple_member_type_negative():
        """Iterable with multiple member types and wrong input should fail"""
        assert not isinstance([1, 'a', 0.5], Iterable[int, float])


class TestZipEqual:
    """Test class for zip_equal"""
    @staticmethod
    def test_equal_length():
        """Zipping 2 iterables of equal length should succeed"""
        assert tuple(zip_equal(range(3), 'abc')) == ((0, 'a'), (1, 'b'), (2, 'c'))

    @staticmethod
    def test_many_equal_length():
        """Zipping more than 2 iterables of equal length should succeed"""
        assert tuple(zip_equal(*(range(3),) * 5)) == ((0,) * 5, (1,) * 5, (2,) * 5)

    @staticmethod
    def test_unequal_length():
        """Zipping 2 iterables of unequal length should fail"""
        with pytest.raises(TypeError):
            tuple(zip_equal(range(3), 'abcd'))
