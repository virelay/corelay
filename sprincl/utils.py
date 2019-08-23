"""Sprincl utils contains conditional importing functionality.

"""

from importlib import import_module
from typing import Iterable as IterableBase


class IterableMeta(type):
    """Meta class to implement member instance checking for Iterables"""
    def __instancecheck__(self, instance):
        """Is instance if iterable and all members are of provided types"""
        return isinstance(instance, IterableBase) and all(isinstance(obj, self.__membertype__) for obj in instance)


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
        return type('{}[{}]'.format(cls.__name__, params), (cls,), {'__membertype__': params})


def zip_equal(*args):
    iters = [iter(obj) for obj in args]
    stop = False
    while not stop:
        more = False
        result = []
        for it in iters:
            try:
                value = next(it)
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
    def func(*args, **kwargs):
        raise RuntimeError("Support for {1} was not installed! Install with: pip install {0}[{1}]".format(
            __name__.split('.')[0], name
        ))
    return func


def dummy_import_module(name):
    """Use to replace 'import lib`."""
    class Class:
        """Dummy substitute class."""
        def __getattr__(self, item):
            raise RuntimeError("Support for {1} was not installed! Install with: pip install {0}[{1}]".format(
                __name__.split('.')[0], name))

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
                    message = "cannot import name '{}' from '{}' ({})".format(model_attribute, name, tmp.__file__)
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
