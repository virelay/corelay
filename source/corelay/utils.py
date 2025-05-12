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
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, LambdaType, MethodType, ModuleType
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


def get_lambda_expression_source_code(lambda_expression: LambdaType) -> str:
    """Returns the source code of the specified lambda expression if possible. This is done by retrieving the source code of the lambda expression and
    parsing it into an abstract syntax tree (AST). The AST node that represents the lambda expression is then used to retrieve the source code of the
    lambda expression by repeatedly removing the trailing junk from the source code until the source code can be compiled without errors, is at least
    as long as the shortest possible lambda expression, and the length of the compiled byte code is the same as the length as the byte code of the
    original lambda expression.

    Note:
        This function was adapted from the original implementation by Karol Kuczmarski, outline in their blog post
        `Source code of a Python lambda <http://xion.io/post/code/python-get-lambda-code.html>`_. The code was published as a
        `Gist on GitHub <https://gist.github.com/Xion/617c1496ff45f3673a5692c3b0e3f75a>`_ and was published under the
        `Creative Commons Attribution-ShareAlike 4.0 International License <https://creativecommons.org/licenses/by-sa/4.0/>`_. The original code
        was modified to fit the coding style of this project and to return "<lambda expression>" instead of :py:obj:`None` if the source
        code cannot be retrieved.

    Args:
        lambda_expression (LambdaType): The lambda expression for which the source code is to be returned.

    Returns:
        str: Returns the source code of the lambda expression if possible, otherwise "<lambda expression>" is returned.
    """

    # Retrieves the line of the source code that contains the lambda expression, if the lambda expression is defined over multiple lines, then they
    # are joined together with new line characters
    try:
        source_lines, _ = inspect.getsourcelines(lambda_expression)
    except (IOError, TypeError):
        return '<lambda expression>'
    if len(source_lines) != 1:
        return '<lambda expression>'
    source_code = os.linesep.join(source_lines).strip()

    # Parses the source code into an AST and retrieves the AST node that represents the lambda expression
    source_ast = ast.parse(source_code)
    lambda_ast_node = next((node for node in ast.walk(source_ast) if isinstance(node, ast.Lambda)), None)
    if lambda_ast_node is None:
        return '<lambda expression>'

    # Since we can (and most likely will) get source lines where the lambda expression is just a part of bigger expressions, they will have some
    # trailing junk after their definition; unfortunately, AST nodes only keep their _starting_ offsets from the original source, so we have to
    # determine the end ourselves; this is done by gradually shaving extra junk from after the definition while ensuring that the code is still valid,
    # is at least as long as the shortest possible lambda expression, and that the length of the compiled byte code is the same as the length of the
    # byte code of the original lambda expression
    lambda_source_code = source_code[lambda_ast_node.col_offset:]
    lambda_body_source_code = source_code[lambda_ast_node.body.col_offset:]
    length_of_smallest_valid_lambda_expression = len('lambda:_')
    while len(lambda_source_code) > length_of_smallest_valid_lambda_expression:
        try:

            # What's annoying is that sometimes the junk even parses, but results in a different lambda; You'd probably have to be deliberately
            # malicious to exploit it but here's one way:
            #
            #     lambda_expression_tuple = lambda x: False, lambda x: True
            #     get_lambda_expression_source_code(lambda_expression_tuple[0])
            #
            # Ideally, we'd just keep shaving until we get the same code, but that most likely won't happen because we can't replicate the exact
            # closure environment; thus the next best thing is to assume some divergence due to, e.g., LOAD_GLOBAL in original code being LOAD_FAST in
            # the one compiled above, or vice versa; But the resulting code should at least be the same length, if otherwise the same operations are
            # performed in it
            code = compile(lambda_body_source_code, '<unused filename>', 'eval')
            if len(code.co_code) == len(lambda_expression.__code__.co_code):
                return lambda_source_code

        # Syntax errors are expected, so we just ignore them; they are most likely caused by the trailing junk in the source code that has not, yet,
        # been removed
        except SyntaxError:
            pass

        # Shaves off the last character of the source code of the lambda expression and the body of the lambda expression
        lambda_source_code = lambda_source_code[:-1]
        lambda_body_source_code = lambda_body_source_code[:-1]

    # If the source code cannot be retrieved, then "<lambda expression>" is returned
    return '<lambda expression>'


def get_object_representation(obj: typing.Any) -> str:
    """Returns a :py:class:`str` representation of the object, which depends on the type of the object. If the object is a type, module, function or
    method, then its name is returned, if the object is a lambda expression, then its source code is returned, otherwise the :py:class:`str`
    representation of the object is returned.

    Args:
        obj (typing.Any): The object for which the :py:class:`str` representation is to be returned.

    Returns:
        str: Returns a :py:class:`str` representation of the object.
    """

    if isinstance(obj, LambdaType):
        return get_lambda_expression_source_code(obj)

    if isinstance(obj, (type, ModuleType, FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType, numpy.ufunc, type(numpy.max))):
        return get_fully_qualified_name(obj)

    return repr(obj)


def get_fully_qualified_name(obj: typing.Any) -> str:
    """Returns the fully qualified name of the object, which is the module name and the name of the object.

    Args:
        obj (typing.Any): The object for which the fully qualified name is to be returned.

    Returns:
        str: Returns the fully qualified name of the object.
    """

    object_class: type = obj.__class__
    object_module = object_class.__module__
    if object_module == 'builtins':
        return object_class.__qualname__
    return object_module + '.' + object_class.__qualname__


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
def import_or_stub(module_name: str, type_and_function_names: str) -> Callable[..., typing.Any]:
    """Tries to import a type or a function from a module. If the import fails, the requested type or function is replaced with a dummy that will
    raise an exception when used. This is useful for optional dependencies of a package, because the retrieved type or function will raise an
    exception that tells the user how to install the missing dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported or from which the specified type or function is to be imported.
        type_and_function_names (str): The name of the type or function that is to be imported from the module.

    Returns:
        Callable[..., typing.Any]: Returns the imported type or function. If the import of the module fails, a dummy is returned that will raise an
        exception when used, telling the user how to install the missing dependencies.
    """


@overload
def import_or_stub(module_name: str, type_and_function_names: tuple[str, ...]) -> list[types.ModuleType | Callable[..., typing.Any]]:
    """Tries to import types and/or functions from a module. If the import fails, the requested types and/or functions are replaced with dummies that
    will raise an exception when used. This is useful for optional dependencies of a package, because the retrieved types and/or functions will raise
    an exception that tells the user how to install the missing dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported or from which the specified types and/or functions are to be imported.
        type_and_function_names (tuple[str, ...]): The names of the types and/or functions that are to be imported from the module.

    Returns:
        list[types.ModuleType | Callable[..., typing.Any]]: Returns a list of the imported modules, types and/or functions. If the import of
        the module fails, dummies are returned that will raise an exception when used, telling the user how to install the missing dependencies.
    """


def import_or_stub(
    module_name: str,
    type_and_function_names: str | tuple[str, ...] | None = None
) -> list[types.ModuleType | Callable[..., typing.Any]] | types.ModuleType | Callable[..., typing.Any]:
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
        list[types.ModuleType | Callable[..., typing.Any]] | types.ModuleType | Callable[..., typing.Any]: Returns the imported
        module, or types and/or functions that were imported from the module. If the import of the module fails, dummies for the module, or types
        and/or functions are returned that will raise an exception when used, telling the user how to install the missing dependencies.
    """

    # Creates a list of the modules, types, and functions that were imported
    modules_types_and_functions: list[types.ModuleType | type | Callable[..., typing.Any]] = []

    # If a list of types and functions to import was provided, then they are imported (or replaced by dummies) and added to the list
    if type_and_function_names is not None:
        type_and_function_names = (type_and_function_names, ) if isinstance(type_and_function_names, str) else type_and_function_names

        try:
            imported_module = import_module(module_name)
        except ImportError:
            modules_types_and_functions = [dummy_from_module_import(module_name) for _ in type_and_function_names]
        else:
            for type_or_function_name in type_and_function_names:
                try:
                    type_or_function = getattr(imported_module, type_or_function_name)
                except AttributeError as exception:
                    raise ImportError(
                        f'Cannot import name "{type_or_function_name}" from "{module_name}" ({imported_module.__file__}).'
                    ) from exception

                modules_types_and_functions.append(type_or_function)

    # If, however, no list of types and functions to import was provided, then the entire module is imported (or replaced by a stub) and added to the
    # list
    else:
        try:
            modules_types_and_functions = [import_module(module_name)]
        except ImportError:
            modules_types_and_functions = [dummy_import_module(module_name)]

    # If only a single module/type/function was imported, then it is returned directly, otherwise a list of the imported modules/types/functions is
    # returned
    if len(modules_types_and_functions) == 1:
        return modules_types_and_functions[0]
    return modules_types_and_functions
