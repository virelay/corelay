"""A module that contains unit tests for the :py:mod:`corelay.utils` module."""

import re
import sys
from types import LambdaType
from typing import Any, Callable

import numpy
import numpy.testing
import pytest

from corelay.utils import get_fully_qualified_name, get_lambda_expression_source_code, get_object_representation, import_or_stub, zip_equal


def test_get_lambda_expression_source_code() -> None:
    """Tests that the :py:func:`~corelay.utils.get_lambda_expression_source_code` function returns the correct source code of a lambda expression."""

    # Test that the function returns the correct source code of a lambda expression that is passed as a parameter
    assert get_lambda_expression_source_code(lambda a, b: a + b) == 'lambda a, b: a + b'

    # Test that the function returns the correct source code of a lambda expression that is defined as a variable
    lambda_expression = lambda x, y: x * y
    assert get_lambda_expression_source_code(lambda_expression) == 'lambda x, y: x * y'

    # Test that the function returns the correct source code of a lambda expression that is defined as a default parameter or passed as a parameter to
    # a function
    def test_function_with_lambda_expression_as_parameter(lambda_expression_parameter: Callable[..., Any] = lambda x, y: x + y) -> str:
        """A test function that takes a lambda expression as a parameter and returns its string representation.

        Args:
            lambda_expression_parameter (Callable[..., Any]): A lambda expression to be represented as a string. Defaults to a simple addition lambda.

        Returns:
            str: Returns the string representation of the lambda expression."""

        assert isinstance(lambda_expression_parameter, LambdaType)
        return get_object_representation(lambda_expression_parameter)

    assert test_function_with_lambda_expression_as_parameter() == 'lambda x, y: x + y'
    assert test_function_with_lambda_expression_as_parameter(lambda a, b: a * b) == 'lambda a, b: a * b'

    # Tests that the function returns the object representation for a non-lambda object
    assert get_lambda_expression_source_code(123) == '123'  # type: ignore[arg-type]

    # Tests that the function returns the object representation if a function is passed instead of a lambda expression
    def test_function_used_instead_of_lambda(a: int, b: int) -> int:
        """A test function that adds two integers.

        Args:
            a (int): The first integer.
            b (int): The second integer.

        Returns:
            int: Returns the sum of the two integers.
        """

        return a + b

    assert re.fullmatch(
        '<function test_get_lambda_expression_source_code.<locals>.test_function_used_instead_of_lambda at 0x[0-9a-f]+>',
        get_lambda_expression_source_code(test_function_used_instead_of_lambda)
    ) is not None

    # Tests that the function returns the correct source code for a lambda expression if there are two lambda expressions in a line, but they differ
    # in their parameters
    lambda_expressions = (lambda x, y: x + y, lambda a, b: a * b)
    assert get_lambda_expression_source_code(lambda_expressions[0]) == 'lambda x, y: x + y'
    assert get_lambda_expression_source_code(lambda_expressions[1]) == 'lambda a, b: a * b'

    # Tests that the function returns the correct source code for a lambda expression if there are two lambda expressions in a line that do not differ
    # in their parameters
    lambda_expressions = (lambda x, y: x + y, lambda x, y: x * y)
    assert get_lambda_expression_source_code(lambda_expressions[0]) == 'lambda x, y: x + y'
    assert get_lambda_expression_source_code(lambda_expressions[1]) == 'lambda x, y: x * y'

    # Tests that the function returns the correct source code for a lambda expression if there are two lambda expressions in a line that do not differ
    # in their parameters, that use a global variable, and that have a different complexity (since the closure environment will be different when the
    # get_lambda_expression_source_code function compiles the lambda expressions, the byte code will be different, so the byte code cannot be used to
    # differentiate the two; since they have different complexity, the complexity of the byte code can be used to differentiate between them)
    global_var = 42
    lambda_expressions = (lambda x, y: x + y + global_var, lambda x, y: sum([x, y, global_var]))
    assert get_lambda_expression_source_code(lambda_expressions[0]) == 'lambda x, y: x + y + global_var'
    assert get_lambda_expression_source_code(lambda_expressions[1]) == 'lambda x, y: sum([x, y, global_var])'

    # Tests that the function returns the object representation for two lambda expressions that do not differ in their parameters, use a global
    # variable, differ in their implementation, but have the same complexity
    lambda_expressions = (lambda x, y: x + y + global_var, lambda x, y: x * y * global_var)
    assert re.fullmatch(
        '<function test_get_lambda_expression_source_code.<locals>.<lambda> at 0x[0-9a-f]+>',
        get_lambda_expression_source_code(lambda_expressions[0])
    ) is not None
    assert re.fullmatch(
        '<function test_get_lambda_expression_source_code.<locals>.<lambda> at 0x[0-9a-f]+>',
        get_lambda_expression_source_code(lambda_expressions[1])
    ) is not None


def test_get_object_representation() -> None:
    """Tests that the :py:func:`~corelay.utils.get_object_representation` function returns the correct string representation of various objects."""

    # Test lambda expressions
    lambda_expression = lambda a, b: a + b
    assert get_object_representation(lambda_expression) == 'lambda a, b: a + b'
    assert get_object_representation(lambda x: x) == 'lambda x: x'

    # Test functions, methods, classes, and modules
    assert get_object_representation(test_get_object_representation) == 'unit_tests.corelay.test_utils.test_get_object_representation'
    assert get_object_representation(object().__str__) == 'object.__str__'
    assert get_object_representation(sys) == 'sys'
    assert get_object_representation(sys.exit) == 'sys.exit'
    assert get_object_representation(len) == 'len'
    assert get_object_representation(None) == 'None'
    assert get_object_representation(int) == 'int'
    assert get_object_representation(str) == 'str'
    assert get_object_representation(list) == 'list'

    # Test values
    assert get_object_representation(123) == '123'
    assert get_object_representation('test') == "'test'"
    assert get_object_representation([]) == '[]'
    assert get_object_representation({}) == '{}'
    assert get_object_representation({'key': 'value'}) == "{'key': 'value'}"
    assert get_object_representation([1, 2, 3]) == '[1, 2, 3]'
    assert get_object_representation((1, 2, 3)) == '(1, 2, 3)'
    assert get_object_representation({1, 2, 3}) == '{1, 2, 3}'


def test_get_fully_qualified_name() -> None:
    """Tests that the :py:func:`~corelay.utils.get_fully_qualified_name` function correctly returns the fully-qualified name of values, types, and
    built-ins.
    """

    # Functions
    assert get_fully_qualified_name(test_get_fully_qualified_name) == 'unit_tests.corelay.test_utils.test_get_fully_qualified_name'

    # Types and Methods
    class TestClass:
        """A test class for testing the retrieval of fully qualified names."""

        def test_method(self) -> None:
            """A test method for testing the retrieval of fully qualified names of methods."""

    assert get_fully_qualified_name(TestClass) == 'unit_tests.corelay.test_utils.test_get_fully_qualified_name.<locals>.TestClass'
    assert (get_fully_qualified_name(TestClass().test_method) ==
        'unit_tests.corelay.test_utils.test_get_fully_qualified_name.<locals>.TestClass.test_method')
    TestClass.__module__ = '__main__'  # Simulate a class defined in the main module
    assert get_fully_qualified_name(TestClass().test_method) == 'test_get_fully_qualified_name.<locals>.TestClass.test_method'

    # Modules
    assert get_fully_qualified_name(sys) == 'sys'
    assert get_fully_qualified_name(numpy.testing) == 'numpy.testing'

    # Built-in functions
    assert get_fully_qualified_name(sys.exit) == 'sys.exit'
    assert get_fully_qualified_name(len) == 'len'

    # Lambda functions
    lambda_expression = lambda x: x
    assert get_fully_qualified_name(lambda_expression) == 'unit_tests.corelay.test_utils.test_get_fully_qualified_name.<locals>.<lambda>'
    assert get_fully_qualified_name(lambda x: x) == 'unit_tests.corelay.test_utils.test_get_fully_qualified_name.<locals>.<lambda>'

    # None is of type 'NoneType', but we want to return 'None' for it
    assert get_fully_qualified_name(None) == 'None'

    # Built-in method wrappers
    assert get_fully_qualified_name(object().__str__) == 'object.__str__'

    # Built-in method descriptors
    assert get_fully_qualified_name(str.join) == 'str.join'

    # Built-in class method descriptors
    assert get_fully_qualified_name(dict.__dict__['fromkeys']) == 'dict.fromkeys'

    # Built-in method wrapper type
    assert get_fully_qualified_name(object().__str__) == 'object.__str__'

    # Built-in wrapper descriptors
    assert get_fully_qualified_name(object.__init__) == 'object.__init__'

    # Built-in types
    assert get_fully_qualified_name(int) == 'int'
    assert get_fully_qualified_name(str) == 'str'
    assert get_fully_qualified_name(123) == 'int'
    assert get_fully_qualified_name('test') == 'str'
    assert get_fully_qualified_name([]) == 'list'

    # Built-in methods
    assert get_fully_qualified_name('test'.lower) == 'str.lower'

    # NumPy array functions and NunPy universal functions
    assert get_fully_qualified_name(numpy.min) == 'numpy.min'
    assert get_fully_qualified_name(numpy.sin) == 'numpy.sin'


def test_conditional_import() -> None:
    """Tests that the conditional import only fails for not installed packages when the modules are first used."""

    non_existing_module = import_or_stub('non_existing_module')
    non_existing_function = import_or_stub('non_existing_module', 'non_existing_function')
    datetime = import_or_stub('datetime')
    datetime_type = import_or_stub('datetime', 'datetime')

    with pytest.raises(RuntimeError):
        non_existing_module.f()
    with pytest.raises(RuntimeError):
        non_existing_function()
    datetime.datetime.now()
    datetime_type.now()  # type: ignore[union-attr]


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
    """Contains unit tests for the :py:func:`~corelay.utils.zip_equal` function."""

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
