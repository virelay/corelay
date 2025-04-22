"""A module that contains unit tests for the ``corelay.utils`` module."""

import pytest

from corelay.utils import import_or_stub, zip_equal


def test_conditional_import() -> None:
    """Tests that the conditional import only fails for not installed packages when the modules are first used."""

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


def test_conditional_import_of_multiple_functions() -> None:
    """Tests that the conditional importing can be used to import multiple functions from the same module."""

    match, fullmatch = import_or_stub('non_existing_module', ('match', 'fullmatch'))

    assert callable(match)
    assert callable(fullmatch)

    with pytest.raises(RuntimeError):
        match('aba', 'a')
    with pytest.raises(RuntimeError):
        fullmatch('aba', 'a')

    match, fullmatch = import_or_stub('re', ('match', 'fullmatch'))

    assert callable(match)
    assert callable(fullmatch)
    match('aba', 'a')
    fullmatch('aba', 'a')

    with pytest.raises(ImportError):
        _, _ = import_or_stub('re', ('findall', 'non_existing'))


class TestZipEqual:
    """Contains unit tests for the ``zip_equal`` function."""

    @staticmethod
    def test_equal_length() -> None:
        """Tests that zipping two iterables of equal length succeeds."""

        assert tuple(zip_equal(range(3), 'abc')) == ((0, 'a'), (1, 'b'), (2, 'c'))

    @staticmethod
    def test_many_equal_length() -> None:
        """Tests that zipping more than two iterables of equal length succeeds."""

        assert tuple(zip_equal(*(range(3),) * 5)) == ((0,) * 5, (1,) * 5, (2,) * 5)

    @staticmethod
    def test_unequal_length() -> None:
        """Tests that zipping two iterables of unequal length fails."""

        with pytest.raises(TypeError):
            tuple(zip_equal(range(3), 'abcd'))
