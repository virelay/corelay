"""CoRelAy utils contains conditional importing functionality.

"""

from importlib import import_module
from typing import Iterable as IterableBase


class IterableMeta(type):
    """Meta class to implement member instance checking for Iterables"""
    def __instancecheck__(cls, instance):
        """Is instance if iterable and all members are of provided types"""
        return isinstance(instance, IterableBase) and all(isinstance(obj, cls.__membertype__) for obj in instance)


class Iterable(metaclass=IterableMeta):
    """Iterables with strict member type checking"""
    __membertype__ = object

    def __class_getitem__(cls, params):
        """Dynamically creates a subclass with the provided member types"""
        if not isinstance(params, tuple):
            params = (params,)
        if not params:
            raise TypeError("At least one member type must be specified!")
        if not all(isinstance(obj, type) for obj in params):
            raise TypeError("Member types must be types!")
        # pylint: disable=no-member
        return type(f'{cls.__name__}[{params}]', (cls,), {'__membertype__': params})


def zip_equal(*args):
    """Zip positional arguments only if they are of equal length.

    Parameters
    ----------
    *args : Iterable
        Iterables of equal length to be zipped

    Yields
    ------
    tuple
        Zipped elements

    Raises
    ------
    TypeError
        If positional arguments are no Iterables, or have different length.

    """
    iterator_list = [iter(obj) for obj in args]
    stop = False
    while not stop:
        more = False
        result = []
        for iterator in iterator_list:
            try:
                value = next(iterator)
            except StopIteration:
                stop = True
            else:
                more = True
                result.append(value)
            if stop and more:
                raise TypeError("Unequal length!")
        if not stop:
            yield tuple(result)


def dummy_from_module_import(name):
    """Use to replace 'from lib import func'."""
    def func(*args, **kwargs):  # pylint: disable=unused-argument
        package_name = __name__.split('.', maxsplit=1)[0]
        raise RuntimeError(f"Support for {name} was not installed! Install with: pip install {package_name}[{name}]")
    return func


def dummy_import_module(name):
    """Use to replace 'import lib`."""
    class Class:
        """Dummy substitute class."""
        def __getattr__(self, item):
            package_name = __name__.split('.', maxsplit=1)[0]
            raise RuntimeError(
                f"Support for {name} was not installed! Install with: pip install {package_name}[{name}]"
            )
    return Class()


def import_or_stub(name, subname=None):
    """Use to conditionally import packages.

    Parameters
    ----------
    name: str
        Module name. Ie. 'module.lib'
    subname: tuple[str] or str or None
        Functions or classes to be imported from 'name'.
    """
    subnames = (subname, ) if isinstance(subname, str) else subname  # convert to tuple
    if subname is not None:
        try:
            tmp = import_module(name)
        except ImportError:
            module = [dummy_from_module_import(name) for _ in subnames]
        else:
            module = []
            for model_attribute in subnames:
                try:
                    attr = getattr(tmp, model_attribute)
                except AttributeError as err:
                    message = f"cannot import name '{model_attribute}' from '{name}' ({tmp.__file__})"
                    raise ImportError(message) from err
                else:
                    module.append(attr)
        if len(module) == 1:
            module = module[0]
    else:
        try:
            module = import_module(name)
        except ImportError:
            module = dummy_import_module(name)
    return module
