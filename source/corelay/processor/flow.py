"""A module that contains basic flow operation processors, such as :py:class:`~corelay.processor.flow.Shaper`,
:py:class:`~corelay.processor.flow.Sequential` and :py:class:`~corelay.processor.flow.Parallel`.
"""

import typing
from collections.abc import Iterable, Iterator, Sequence
from typing import Annotated

from corelay.base import Param
from corelay.processor.base import Processor
from corelay.utils import zip_equal


class Shaper(Processor):
    """Extracts and/or copies by indices.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Shaper` processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        indices (tuple[int | tuple[int], ...]): The indices to copy/extract. The resulting output will be a tuple with the same member shape. Each
            index may be passed an arbitrary amount of times. Outer tuples allow integers and tuples, inner tuples only allow integers.

    Examples:
        >>> Shaper(indices=(0, 1, (0, 1, 2)))(['a', 'b', 'c'])
        ('a', 'b', ('a', 'b', 'c'))
    """

    indices: Annotated[tuple[int | tuple[int], ...], Param(tuple, positional=True)]
    """The indices to copy/extract. The resulting output will be a tuple with the same member shape. Each index may be passed an arbitrary amount of
    times. Outer tuples allow integers and tuples, inner tuples only allow integers.
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Extracts and/or copies indices of data.

        Args:
            data (typing.Any): The data from which the elements, identified by the indices, are to be extracted. This can be any object, but if it is
                not an iterable, index 0 corresponds to the object itself.

        Raises:
            TypeError: An invalid index was accessed in the data.

        Returns:
            typing.Any: Returns the extracted/copied elements of the data, identified by the indices. The output is a tuple with the same member shape
            as the indices.
        """

        def extract_elements_from_indices(iterable: Sequence[typing.Any], indices: tuple[int | tuple[int], ...]) -> tuple[typing.Any, ...]:
            """Recursively extracts the elements of the specified ``iterable`` based on the specified ``indices``.

            Args:
                iterable (Sequence[typing.Any]): The iterable from which to extract the elements.
                indices (tuple[int | tuple[int], ...]): The indices to extract. This can be a tuple of integers or other tuples of integers.

            Raises:
                TypeError: An invalid index was accessed in the iterable.

            Returns:
                tuple[typing.Any, ...]: Returns the extracted elements as a tuple. The output is a tuple with the same member shape as the indices.
            """

            results = []
            for index in indices:
                if isinstance(index, Iterable):
                    extracted_element = extract_elements_from_indices(iterable, index)
                else:
                    try:
                        extracted_element = iterable[index]
                    except KeyError as err:
                        raise TypeError(f'The "{index}" is not a valid index for "{iterable}".') from err
                results.append(extracted_element)
            return tuple(results)

        if not isinstance(data, Sequence):
            data = (data,)
        try:
            return extract_elements_from_indices(data, self.indices)
        except TypeError as exception:
            raise TypeError('An invalid index was used to index the input data.') from exception


class GroupProcessor(Processor):
    """The abstract base class for groups of processors.

    Args:
        is_output (bool): A value indicating whether this :py:class:`GroupProcessor` processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        children (Iterable[Processor]): The children of the group. This is a list of processors that will be called in parallel or sequentially.
    """

    children: Annotated[Iterable[Processor], Param(Iterable, positional=True)]
    """The children of the group. This is a list of processors that will be called in parallel or sequentially."""


class Parallel(GroupProcessor):
    """A processor group that is invoking its children in parallel.

    Note:
        Please note, that the child processors are not executed in parallel in the sense of multiprocessing, but that the children all either receive
        the same input data or an element of the input data, in contrast to the :py:class:`Sequential` processor group, which first executes the first
        child and then feeds the output to the next child.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Parallel` group processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        children (Iterable[Processor]): The children of the group. This is a list of processors that will be called in parallel.
        broadcast (bool): A value indicating whether the input data should be broadcasted to all children. If :py:obj:`True`, the input data will
            be copied as many times as there are children. If :py:obj:`False` and the input is an iterable, the elements of the iterable will be
            passed to the children one by one. Defaults to :py:obj:`False`.

    Examples:
        >>> Parallel(children=[FunctionProcessor(processing_function=lambda x: x**n) for n in (1, 2, 3, 4)])((2, 2, 2, 2))
        [2, 4, 8, 16]
        >>> Parallel(children=[FunctionProcessor(processing_function=lambda x: x**n) for n in (1, 2, 3, 4)])(2)
        [2, 4, 8, 16]
    """

    broadcast: Annotated[bool, Param(bool, False)]
    """A value indicating whether the input data should be broadcasted to all children. If :py:obj:`True`, the input data will be copied as many times
    as there are children. If :py:obj:`False` and the input is an iterable, the elements of the iterable will be passed to the children one by one.
    Defaults to :py:obj:`False`.
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Invokes the children in parallel, passing the input data to each child. If :py:attr:`broadcast` is :py:obj:`True`, the input data will be
        copied as many times as there are children. If :py:attr:`broadcast` is :py:obj:`False` and the input is an iterable, the elements of the
        iterable will be passed to the children one by one.

        Args:
            data (typing.Any): The input data to pass to the children. If :py:attr:`broadcast` is :py:obj:`True`, this can be any object. If
                :py:class:`broadcast` is :py:obj:`False`, and the input is an iterable, the elements of the iterable will be passed to the children
                one by one.

        Raises:
            TypeError: The :py:attr:`broadcast` parameter is set to :py:obj:`True`, and the number of children and number of data elements mismatch.

        Returns:
            typing.Any: Returns a tuple that has the same number of elements as there are children and contains the outputs of the child processors.
        """

        def wrap_iterator_with_meaningful_exception(iterable: Iterable[typing.Any]) -> Iterator[typing.Any]:
            """An iterator that wraps the passed iterator and raises an error, if the number of elements that the iterator produces is not the same as
            the number of child processors. This is done so that the error message is more informative.

            Args:
                iterable (Iterable[typing.Any]): The iterator to wrap.

            Yields:
                typing.Any: The elements of the iterator.

            Raises:
                TypeError: The number of elements that the iterator produces is not the same as the number of child processors.
            """

            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except TypeError as exception:
                    raise TypeError('Number of data elements and children does not match.') from exception
                except StopIteration:
                    break

        if not isinstance(data, Iterable) or self.broadcast:
            data = tuple(data for _ in self.children)

        try:
            results = []
            for child, element in wrap_iterator_with_meaningful_exception(zip_equal(self.children, data)):
                output = child(element)
                results.append(output)
            return tuple(results)
        except TypeError as exception:
            raise TypeError('Number of data elements and children does not match.') from exception


class Sequential(GroupProcessor):
    """A processor group that invokes its children in sequence, feeding the input the first child, and then each output to the next child.

    Args:
        is_output (bool): A value indicating whether this :py:class:`Sequential` group processor is the output of a
            :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to :py:obj:`False`.
        is_checkpoint (bool | None): A value indicating whether check-pointed pipeline computations should start at this point, if there exists a
            previously computed checkpoint value. Defaults to :py:obj:`False`.
        io (Storable | None): An IO object that is used to cache intermediate results of the :py:class:`~corelay.pipeline.base.Pipeline`, which can
            then be re-used in this run or in subsequent runs of the :py:class:`~corelay.pipeline.base.Pipeline`. Defaults to an instance of
            :py:class:`~corelay.io.NoStorage`.
        children (Iterable[Processor]): The children of the group. This is a list of processors that will be called in sequentially.

    Examples:
        >>> Sequential(children=[FunctionProcessor(processing_function=lambda x: c + x) for c in 'abcd'])('=')
        'dcba='
    """

    def function(self, data: typing.Any) -> typing.Any:
        """Invokes the child processors of the processor group in sequence. The input data is fed to the first child, whose output is then fed into
        the second child, and so on.

        Args:
            data (typing.Any): The input data to pass to the first child.

        Returns:
            typing.Any: Returns the output of the last child processor.
        """

        for child in self.children:
            data = child(data)
        return data
