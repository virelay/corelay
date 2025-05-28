"""A module containing utility classes and functions for CoRelAy. This includes as an enhanced :py:func:`zip` function, which ensures that the
sequences being zipped are of equal length. Furthermore, this module contains functions for getting the representations of runtime objects like
functions, methods, classes, and lambda expressions and conditional import functions, which return dummy functions, classes, or modules that raise an
error when called or accessed, indicating how to install the missing dependencies for the functionality to work.
"""

import ast
import inspect
import os
import types
import typing
from collections.abc import Callable, Iterable, Iterator
from importlib import import_module
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    ClassMethodDescriptorType,
    FunctionType,
    LambdaType,
    MethodDescriptorType,
    MethodType,
    MethodWrapperType,
    ModuleType,
    WrapperDescriptorType
)
from typing import NoReturn, overload

import numpy


def zip_equal(*args: Iterable[typing.Any]) -> Iterator[tuple[typing.Any, ...]]:
    """Zips the positional arguments, but only if they are of equal length.

    Args:
        *args (Iterable[typing.Any]): The iterables that are to be zipped. They must be of equal length.

    Raises:
        TypeError: The positional arguments have different lengths.

    Yields:
        tuple[typing.Any, ...]: Yields the zipped elements of the positional arguments.
    """

    iterators = [iter(iterable) for iterable in args]

    has_any_iterator_stopped = False
    while not has_any_iterator_stopped:
        zipped_element = []
        has_any_iterator_more_elements = False
        for iterator in iterators:
            try:
                value = next(iterator)
            except StopIteration:
                has_any_iterator_stopped = True
            else:
                has_any_iterator_more_elements = True
                zipped_element.append(value)

            if has_any_iterator_stopped and has_any_iterator_more_elements:
                raise TypeError('The iterables have different lengths.')

        if not has_any_iterator_stopped:
            yield tuple(zipped_element)


def get_lambda_expression_source_code(lambda_expression: Callable[..., typing.Any]) -> str:
    """Returns the source code of the specified lambda expression if possible. This is done by retrieving the source code of the lambda expression and
    parsing it into an abstract syntax tree (AST). The AST node that represents the lambda expression is then used to retrieve the source code of the
    lambda expression.

    Note:
        This function was adapted from the original implementation by Karol Kuczmarski, outline in their blog post
        `Source code of a Python lambda <http://xion.io/post/code/python-get-lambda-code.html>`_. The code was published as a
        `Gist on GitHub <https://gist.github.com/Xion/617c1496ff45f3673a5692c3b0e3f75a>`_ and was published under the
        `Creative Commons Attribution-ShareAlike 4.0 International License <https://creativecommons.org/licenses/by-sa/4.0/>`_. The original code
        was modified to fit the coding style of this project and to return the representation of the passed object instead of :py:obj:`None` if the
        source code cannot be retrieved.

    Args:
        lambda_expression (Callable[..., typing.Any]): The lambda expression for which the source code is to be returned.

    Returns:
        str: Returns the source code of the lambda expression if possible, otherwise the representation of the object is returned.
    """

    # Retrieves the line of the source code that contains the lambda expression, if the lambda expression is defined over multiple lines, then they
    # are joined together with new line characters
    try:
        source_lines, _ = inspect.getsourcelines(lambda_expression)
    except (IOError, TypeError):
        return repr(lambda_expression)
    source_code = os.linesep.join(source_lines).strip()

    # Parses the source code into an AST and retrieves the AST nodes that represent lambda expressions; if none were found, then "<lambda>" is
    # returned; if only a single lambda expression was found, then its code segment is returned
    source_ast = ast.parse(source_code)
    lambda_expression_source_segments: list[str] = [
        source_segment for source_segment in {
            ast.get_source_segment(source_code, node) for node in ast.walk(source_ast) if isinstance(node, ast.Lambda)
        } if source_segment is not None
    ]
    if not lambda_expression_source_segments:
        return repr(lambda_expression)
    if len(lambda_expression_source_segments) == 1:
        return lambda_expression_source_segments[0]

    # If there are multiple lambda expressions in the source code, then we need to compile the lambda functions to further compare them; if none of
    # them could be compiled, then "<lambda>" is returned; if only a single one could be compiled, then its source code is returned
    compiled_lambda_expressions: dict[str, LambdaType] = {}
    for source_segment in lambda_expression_source_segments:
        try:
            compiled_lambda_expressions[source_segment] = eval(compile(source_segment, '<filename>', 'eval'))  # pylint: disable=eval-used
        except SyntaxError:  # pragma: no cover
            continue
    if not compiled_lambda_expressions:  # pragma: no cover
        return repr(lambda_expression)
    if len(compiled_lambda_expressions) == 1:  # pragma: no cover
        return next(iter(compiled_lambda_expressions.keys()))

    # First we check if which of the lambda expressions that were compiled match the parameters of the specified lambda expression; if none of them
    # match, then we return "<lambda>"; if only one matches, then we return its source code
    lambda_expressions_with_matching_parameters: dict[str, LambdaType] = {}
    for lambda_expression_source_segment, compiled_lambda_expression in compiled_lambda_expressions.items():
        if compiled_lambda_expression.__code__.co_varnames == lambda_expression.__code__.co_varnames:
            lambda_expressions_with_matching_parameters[lambda_expression_source_segment] = compiled_lambda_expression
    if not lambda_expressions_with_matching_parameters:  # pragma: no cover
        return repr(lambda_expression)
    if len(lambda_expressions_with_matching_parameters) == 1:
        return next(iter(lambda_expressions_with_matching_parameters.keys()))

    # If there are multiple lambda expressions with matching parameters, then we need to compare their byte code; since we cannot replicate the exact
    # closure environment, there may be some divergence in the byte codes, but if they do not use any global variables, then the byte codes should be
    # identical and we can use the byte code to determine which lambda expression is the correct one
    for lambda_expression_source_segment, compiled_lambda_expression in lambda_expressions_with_matching_parameters.items():
        if compiled_lambda_expression.__code__.co_code == lambda_expression.__code__.co_code:
            return lambda_expression_source_segment

    # Finally, if none of the lambda expressions are an exact byte code match, we can compare their byte code lengths; maybe, the lambda expressions
    # differ in their complexity and the byte code of the correct lambda expression only differs in some minor way, for example, the LOAD_GLOBAL
    # instruction was used in the original code, but was compiled as a LOAD_FAST instruction, due to the closure environment being different; in such
    # a case, the byte code lengths should be similar, so we can use the difference in the byte code lengths to determine which lambda expression is
    # the correct one; since it is unlikely that the candidate lambda expressions will be smaller than the actual lambda expression, we can directly
    # dismiss candidates whose difference in byte code length is negative; in case the byte code lengths between the candidate lambda expressions does
    # not differ, i.e., there are multiple lambda expressions with the same byte code length, then we cannot determine which one is the correct one,
    # so we return the object representation instead
    lambda_expression_byte_code_length_differences: dict[str, int] = {}
    for lambda_expression_source_segment, compiled_lambda_expression in lambda_expressions_with_matching_parameters.items():
        byte_code_length_difference = len(compiled_lambda_expression.__code__.co_code) - len(lambda_expression.__code__.co_code)
        if byte_code_length_difference >= 0:
            lambda_expression_byte_code_length_differences[lambda_expression_source_segment] = byte_code_length_difference
    sorted_byte_code_length_differences = list(sorted(lambda_expression_byte_code_length_differences.values()))
    if len(sorted_byte_code_length_differences) > 1 and sorted_byte_code_length_differences[0] == sorted_byte_code_length_differences[1]:
        return repr(lambda_expression)
    return min(
        lambda_expression_byte_code_length_differences,
        key=lambda source_segment: lambda_expression_byte_code_length_differences[source_segment]
    )


def get_object_representation(obj: typing.Any) -> str:
    """Returns a :py:class:`str` representation of the object, which depends on the type of the object. If the object is a type, module, function or
    method, then its name is returned, if the object is a lambda expression, then its source code is returned, otherwise the :py:class:`str`
    representation of the object is returned.

    Args:
        obj (typing.Any): The object for which the :py:class:`str` representation is to be returned.

    Returns:
        str: Returns a :py:class:`str` representation of the object.
    """

    if isinstance(obj, LambdaType) and '<lambda>' in str(obj):
        return get_lambda_expression_source_code(obj)

    if isinstance(obj, (
        type,
        BuiltinFunctionType,
        BuiltinMethodType,
        ClassMethodDescriptorType,
        FunctionType,
        LambdaType,
        MethodDescriptorType,
        MethodType,
        MethodWrapperType,
        ModuleType,
        WrapperDescriptorType,
        numpy.ufunc,
        type(numpy.max))
    ):
        return get_fully_qualified_name(obj)

    return repr(obj)


def get_fully_qualified_name(obj: typing.Any) -> str:
    """Returns the fully qualified name of the object, which is the module name and the name of the object.

    Args:
        obj (typing.Any): The object for which the fully qualified name is to be returned. If the object is a :py:class:`type`, then the
            fully-qualified name of the type is returned, otherwise the fully-qualified name of the class of the object is returned.

    Returns:
        str: Returns the fully qualified name of the object.
    """

    if obj is None:
        return 'None'

    if isinstance(obj, (BuiltinFunctionType, BuiltinMethodType, FunctionType, LambdaType, numpy.ufunc, type(numpy.max))):
        if obj.__module__ is None or obj.__module__ in ('builtins', '__main__'):
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'

    if isinstance(obj, MethodType):
        if obj.__self__.__class__.__module__ == '__main__':
            return obj.__func__.__qualname__
        return f'{obj.__self__.__class__.__module__}.{obj.__qualname__}'

    if isinstance(obj, (ClassMethodDescriptorType, MethodDescriptorType, MethodWrapperType, WrapperDescriptorType)):
        return obj.__qualname__

    if isinstance(obj, ModuleType):
        return obj.__name__

    object_type = obj if isinstance(obj, type) else type(obj)
    if object_type.__module__ in ('builtins', '__main__'):
        return object_type.__qualname__
    return f'{object_type.__module__}.{object_type.__qualname__}'


def dummy_from_module_import(module_name: str) -> Callable[..., typing.Any]:
    """Creates a stub function that raises an error when called. This is used to replace an actual import of a type or function from a module, like
    `from module import type` or `from module import function`, of a package that has not been installed. It is useful for optional dependencies of a
    package, because the retrieved function will raise an exception that tells the user how to install the missing dependencies for the functionality
    to work. It does not matter if the user meant to import a function or a type, as types are also callable (i.e., if a type is called, then an
    instance of the type is created and the constructor ``__init__`` of that type is called) and invoking a function is syntactically
    indistinguishable from instantiating an instance of a type. If the user meant to import a function, then the exception will be raised when the
    function is called. If the user meant to import a type, then the exception will be raised when the user tries to instantiate an instance of the
    type.

    Args:
        module_name (str): The name of the module from which the type or function that is to be replaced was imported.

    Returns:
        Callable[..., typing.Any]: Returns a function that raises an error when called, which instructs the user on how to install the missing
        dependencies.
    """

    def function(*_args: typing.Any, **_kwargs: typing.Any) -> NoReturn:
        """A stub function that raises an error when called, which instructs the user on how to install the missing dependencies.

        Args:
            *_args (typing.Any): The positional arguments that were passed to the function.
            **_kwargs (typing.Any): The keyword arguments that were passed to the function.

        Raises:
            RuntimeError: Always raises a :py:class:`RuntimeError` with a message indicating how to install the missing dependencies.
        """

        # Retrieves the name of this package, which is the first part of the fully-qualified name of this module; this should always be "corelay",
        # unless the project will be renamed at some point in the future
        package_name = __name__.split('.', maxsplit=1)[0]

        raise RuntimeError(
            f'Support for "{module_name}" was not installed. The missing dependencies can be installed as follows: '
            f'pip install {package_name}[{module_name}].'
        )

    return function


def dummy_import_module(module_name: str) -> typing.Any:
    """Creates a stub class that raises an error when any of its attributes are accessed and returns an instance of it. This is used to replace an
    actual import of a module, like `import module`, of a package that has not been installed. It is useful for optional dependencies of a package,
    because the retrieved object will raise an exception that tells the user how to install the missing dependencies for the functionality to work. It
    does not matter that the user meant to import a module and not a class, as accessing a module's members is syntactically indistinguishable from
    accessing a class's attributes. So of the user imported a module, the exception will be raised when the user tries to access any of the module's
    members.

    Args:
        module_name (str): The name of the module that is to be replaced.

    Returns:
        typing.Any: Returns an instance of a class that raises an error when any of its attributes are accessed, which instructs the user on how to
        install the missing dependencies.
    """

    class Class:
        """A stub class that raises an error when any of its attributes are access, which instructs the user on how to install the missing
        dependencies.
        """

        def __getattr__(self, _item: str) -> NoReturn:
            """Is invoked when an attribute of the class is accessed. It raises a :py:class:`RuntimeError` with a message indicating how to install
            the missing dependencies.

            Args:
                _item (str): The name of the attribute that was accessed.

            Raises:
                RuntimeError: Always raises a :py:class:`RuntimeError` with a message indicating how to install the missing dependencies.
            """

            # Retrieves the name of this package, which is the first part of the fully-qualified name of this module; this should always be "corelay",
            # unless the project will be renamed at some point in the future
            package_name = __name__.split('.', maxsplit=1)[0]

            raise RuntimeError(
                f'Support for "{module_name}" was not installed. The missing dependencies can be installed as follows: '
                f'pip install {package_name}[{module_name}].'
            )

    return Class()


@overload
def import_or_stub(module_name: str) -> types.ModuleType:
    """Tries to import a module. If the import fails, the requested module is replaced with a dummy that will raise an exception when used. This is
    useful for optional dependencies of a package, because the retrieved module will raise an exception that tells the user how to install the missing
    dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported.

    Returns:
        types.ModuleType: Returns the imported module. If the import of the module fails, a dummy is returned that will raise an exception when used,
        telling the user how to install the missing dependencies.
    """


@overload
def import_or_stub(module_name: str, type_and_function_names: str) -> type[typing.Any] | Callable[..., typing.Any]:
    """Tries to import a type or a function from a module. If the import fails, the requested type or function is replaced with a dummy that will
    raise an exception when used. This is useful for optional dependencies of a package, because the retrieved type or function will raise an
    exception that tells the user how to install the missing dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported or from which the specified type or function is to be imported.
        type_and_function_names (str): The name of the type or function that is to be imported from the module.

    Returns:
        type[typing.Any] | Callable[..., typing.Any]: Returns the imported type or function. If the import of the module fails, a dummy is returned
        that will raise an exception when used, telling the user how to install the missing dependencies.
    """


@overload
def import_or_stub(module_name: str, type_and_function_names: tuple[str, ...]) -> list[type | Callable[..., typing.Any]]:
    """Tries to import types and/or functions from a module. If the import fails, the requested types and/or functions are replaced with dummies that
    will raise an exception when used. This is useful for optional dependencies of a package, because the retrieved types and/or functions will raise
    an exception that tells the user how to install the missing dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported or from which the specified types and/or functions are to be imported.
        type_and_function_names (tuple[str, ...]): The names of the types and/or functions that are to be imported from the module.

    Returns:
        list[type | Callable[..., typing.Any]]: Returns a list of the imported modules, types and/or functions. If the import of the module fails,
        dummies are returned that will raise an exception when used, telling the user how to install the missing dependencies.
    """


def import_or_stub(
    module_name: str,
    type_and_function_names: str | tuple[str, ...] | None = None
) -> types.ModuleType | type[typing.Any] | Callable[..., typing.Any] | list[type | Callable[..., typing.Any]]:
    """Tries to import a module, or types and/or functions from a module. If the import fails, the requested module, or types and/or functions are
    replaced with dummies that will raise an exception when used. This is useful for optional dependencies of a package, because the retrieved module,
    or types and/or functions will raise an exception that tells the user how to install the missing dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported or from which the specified types and/or functions are to be imported.
        type_and_function_names (str | tuple[str, ...] | None): The names of the types and/or functions that are to be imported from the
            module, e.g., `'function'` or `('function', 'type')`. If this argument is not provided, the entire module is imported. If the module, or
            the types and/or functions cannot be imported, dummies will be returned that will raise an exception when used. Defaults to
            :py:obj:`None`.

    Raises:
        ImportError: If a type or function cannot be imported from a module that is installed, an :py:class:`ImportError` is raised. This is done,
            instead of stubbing the imported type or function, because this always indicates a bug in the code and cannot be fixed by the user
            installing a missing dependency.

    Returns:
        types.ModuleType | type[typing.Any] | Callable[..., typing.Any] | list[type | Callable[..., typing.Any]]: Returns the imported module, or
        types and/or functions that were imported from the module. If the import of the module fails, dummies for the module, or types and/or
        functions are returned that will raise an exception when used, telling the user how to install the missing dependencies.
    """

    # If a list of types and functions to import was provided, then they are imported from the specified module(or replaced by dummies)
    if type_and_function_names is not None:
        type_and_function_names = (type_and_function_names, ) if isinstance(type_and_function_names, str) else type_and_function_names

        types_and_functions: list[type | Callable[..., typing.Any]] = []
        try:
            imported_module = import_module(module_name)
        except ImportError:
            types_and_functions = [dummy_from_module_import(module_name) for _ in type_and_function_names]
        else:
            for type_or_function_name in type_and_function_names:
                try:
                    type_or_function = getattr(imported_module, type_or_function_name)
                except AttributeError as exception:
                    raise ImportError(
                        f'Cannot import name "{type_or_function_name}" from "{module_name}" ({imported_module.__file__}).'
                    ) from exception

                types_and_functions.append(type_or_function)

        if len(types_and_functions) == 1:
            return types_and_functions[0]
        return types_and_functions

    # If no list of types and functions to import was provided, then the entire module is imported (or replaced by a stub) and returned
    try:
        return import_module(module_name)
    except ImportError:
        module: ModuleType = dummy_import_module(module_name)
        return module
