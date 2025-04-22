"""A module containing utility classes and functions for CoRelAy, such as an iterable with member type checking, an enhanced zip function, and
conditional import functions.
"""

from collections.abc import Callable, Iterable, Iterator
from importlib import import_module
from types import ModuleType
from typing import Any, NoReturn, overload


def zip_equal(*args: Iterable[Any]) -> Iterator[tuple[Any, ...]]:
    """Zips its positional arguments, but only if they are of equal length.

    Args:
        *args (Iterable[Any]): The iterables that are to be zipped. They must be of equal length.

    Raises:
        TypeError: The positional arguments have different lengths.

    Yields:
        tuple[Any, ...]: Yields the zipped elements of the positional arguments.
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


def dummy_from_module_import(module_name: str) -> Callable[..., Any]:
    """Creates a stub function that raises an error when called. This is used to replace an actual import of a type or function from a module, like
    `from module import type` or `from module import function`, of a package that has not been installed. It is useful for optional dependencies of a
    package, because the retrieved function will raise an exception that tells the user how to install the missing dependencies for the functionality
    to work. It does not matter if the user meant to import a function or a type, as types are also callable (i.e., if a type is called, then an
    instance of the type is created and the constructor ``__init__`` is called) and invoking a function is syntactically indistinguishable from
    instantiating an instance of a type. If the user meant to import a function, then the exception will be raised when the function is called. If the
    user meant to import a type, then the exception will be raised when the user tries to instantiate an instance of the type.

    Args:
        module_name (str): The name of the module from which the type or function that is to be replaced was imported.

    Returns:
        Callable[..., Any]: Returns a function that raises an error when called, which instructs the user on how to install the missing dependencies.
    """

    def function(*_args: Any, **_kwargs: Any) -> NoReturn:
        """A stub function that raises an error when called, which instructs the user on how to install the missing dependencies.

        Args:
            *_args (Any): The positional arguments that were passed to the function.
            **_kwargs (Any): The keyword arguments that were passed to the function.

        Raises:
            RuntimeError: Always raises a ``RuntimeError`` with a message indicating how to install the missing dependencies.
        """

        # Retrieves the name of this package, which is the first part of the fully-qualified name of this module; this should always be "corelay",
        # unless the project will be renamed at some point in the future
        package_name = __name__.split('.', maxsplit=1)[0]

        raise RuntimeError(
            f'Support for "{module_name}" was not installed. The missing dependencies can be installed as follows: '
            f'pip install {package_name}[{module_name}].'
        )

    return function


def dummy_import_module(module_name: str) -> Any:
    """Creates a stub class that raises an error when any of its attributes are accessed and returns an instance of it. This is used to replace an
    actual import of a module, like `import module`, of a package that has not been installed. It is useful for optional dependencies of a package,
    because the retrieved object will raise an exception that tells the user how to install the missing dependencies for the functionality to work. It
    does not matter that the user meant to import a module and not a class, as accessing a module's members is syntactically indistinguishable from
    accessing a class's attributes. So of the user imported a module, the exception will be raised when the user tries to access any of the module's
    members.

    Args:
        module_name (str): The name of the module that is to be replaced.

    Returns:
        Any: Returns an instance of a class that raises an error when any of its attributes are accessed, which instructs the user on how to install
            the missing dependencies.
    """

    class Class:
        """A stub class that raises an error when any of its attributes are access, which instructs the user on how to install the missing
        dependencies.
        """

        def __getattr__(self, _item: str) -> NoReturn:
            """Is invoked when an attribute of the class is accessed. It raises a ``RuntimeError`` with a message indicating how to install the
            missing dependencies.

            Args:
                _item (str): The name of the attribute that was accessed.

            Raises:
                RuntimeError: Always raises a ``RuntimeError`` with a message indicating how to install the missing dependencies.
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
def import_or_stub(module_name: str) -> ModuleType:
    """Tries to import a module. If the import fails, the requested module is replaced with a dummy that will raise an exception when used. This is
    useful for optional dependencies of a package, because the retrieved module will raise an exception that tells the user how to install the missing
    dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported.

    Returns:
        ModuleType: Returns the imported module. If the import of the module fails, a dummy is returned that will raise an exception when used,
            telling the user how to install the missing dependencies.
    """


@overload
def import_or_stub(module_name: str, type_and_function_names: str) -> Callable[..., Any]:
    """Tries to import a type or a function from a module. If the import fails, the requested type or function is replaced with a dummy that will
    raise an exception when used. This is useful for optional dependencies of a package, because the retrieved type or function will raise an
    exception that tells the user how to install the missing dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported or from which the specified type or function is to be imported.
        type_and_function_names (str): The name of the type or function that is to be imported from the module.

    Returns:
        Callable[..., Any]: Returns the imported type or function. If the import of the module fails, a dummy is returned that will raise an
            exception when used, telling the user how to install the missing dependencies.
    """


@overload
def import_or_stub(module_name: str, type_and_function_names: tuple[str, ...]) -> list[ModuleType | Callable[..., Any]]:
    """Tries to import types and/or functions from a module. If the import fails, the requested types and/or functions are replaced with dummies that
    will raise an exception when used. This is useful for optional dependencies of a package, because the retrieved types and/or functions will raise
    an exception that tells the user how to install the missing dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported or from which the specified types and/or functions are to be imported.
        type_and_function_names (tuple[str, ...]): The names of the types and/or functions that are to be imported from the module.

    Returns:
        list[ModuleType | Callable[..., Any]]: Returns a list of the imported modules, types and/or functions. If the import of the module fails,
            dummies are returned that will raise an exception when used, telling the user how to install the missing dependencies.
    """


def import_or_stub(
    module_name: str,
    type_and_function_names: str | tuple[str, ...] | None = None
) -> list[ModuleType | Callable[..., Any]] | ModuleType | Callable[..., Any]:
    """Tries to import a module, or types and/or functions from a module. If the import fails, the requested module, or types and/or functions are
    replaced with dummies that will raise an exception when used. This is useful for optional dependencies of a package, because the retrieved module,
    or types and/or functions will raise an exception that tells the user how to install the missing dependencies for the functionality to work.

    Args:
        module_name (str): The name of the module that is to be imported or from which the specified types and/or functions are to be imported.
        type_and_function_names (str | tuple[str, ...] | None, optional): The names of the types and/or functions that are to be imported from the
            module, e.g., `'function'` or `('function', 'type')`. If this argument is not provided, the entire module is imported. If the module, or
            the types and/or functions cannot be imported, dummies will be returned that will raise an exception when used. Defaults to `None`.

    Raises:
        ImportError: If a type or function cannot be imported from a module that is installed, an ``ImportError`` is raised. This is done, instead of
            stubbing the imported type or function, because this always indicates a bug in the code and cannot be fixed by the user installing a
            missing dependency.

    Returns:
        list[ModuleType | Callable[..., Any]] | ModuleType | Callable[..., Any]: Returns the imported module, or types and/or functions that were
            imported from the module. If the import of the module fails, dummies for the module, or types and/or functions are returned that will
            raise an exception when used, telling the user how to install the missing dependencies.
    """

    # Creates a list of the modules, types, and functions that were imported
    modules_types_and_functions: list[ModuleType | type | Callable[..., Any]] = []

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
