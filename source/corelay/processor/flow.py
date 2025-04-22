"""A module that contains basic flow operation processors, such as ``Shaper``, ``Sequential`` and ``Parallel``."""

from collections.abc import Iterable, Iterator, Sequence
from typing import Annotated, Any

from corelay.base import Param
from corelay.processor.base import Processor
from corelay.utils import zip_equal


RecursiveIndicesTuple = tuple['int | RecursiveIndicesTuple', ...]
"""A recursive tuple of integer indices, i.e., a tuple that contains integer indices or other tuples of integer indices, which themselves can contain
other tuples of integer indices, and so on. This is used to represent a nested structure of integer indices.
"""


class Shaper(Processor):
    """Extracts and/or copies by indices.

    Args:
        is_output (bool, optional): A value indicating whether this ``Shaper`` processor is the output of a ``Pipeline``. Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        indices (RecursiveIndicesTuple): The indices to copy/extract. The resulting output will be a tuple with the same member shape. Each index may
            be passed an arbitrary amount of times. Outer tuples allow integers and tuples, inner tuples only allow integers.

    Examples:
        >>> Shaper(indices=(0, 1, (0, 1, 2)))(['a', 'b', 'c'])
        ('a', 'b', ('a', 'b', 'c'))
    """

    indices: Annotated[RecursiveIndicesTuple, Param(tuple, positional=True)]
    """The indices to copy/extract. The resulting output will be a tuple with the same member shape. Each index may be passed an arbitrary amount of
    times. Outer tuples allow integers and tuples, inner tuples only allow integers.
    """

    def function(self, data: Any) -> Any:
        """Extracts and/or copies indices of data.

        Args:
            data (Any): The data from which the elements, identified by the indices, are to be extracted. This can be any object, but if it is not an
                iterable, index 0 corresponds to the object itself.

        Raises:
            TypeError: An invalid index was accessed in the data.

        Returns:
            Any: Returns the extracted/copied elements of the data, identified by the indices. The output is a tuple with the same member shape as the
                indices.
        """

        def extract_elements_from_indices(iterable: Sequence[Any], indices: RecursiveIndicesTuple) -> tuple[Any, ...]:
            """Recursively extracts the elements of the specified ``iterable`` based on the specified ``indices``.

            Args:
                iterable (Sequence[Any]): The iterable from which to extract the elements.
                indices (RecursiveIndicesTuple): The indices to extract. This can be a tuple of integers or other tuples of integers.

            Raises:
                TypeError: An invalid index was accessed in the iterable.

            Returns:
                tuple[Any, ...]: The extracted elements as a tuple. The output is a tuple with the same member shape as the indices.
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
        is_output (bool, optional): A value indicating whether this ``GroupProcessor`` processor is the output of a ``Pipeline``. Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        children (Iterable[Processor]): The children of the group. This is a list of processors that will be called in parallel or sequentially.
    """

    children: Annotated[Iterable[Processor], Param(Iterable, positional=True)]
    """The children of the group. This is a list of processors that will be called in parallel or sequentially."""


class Parallel(GroupProcessor):
    """A processor group that is invoking its children in parallel.

    Note:
        Please note, that the child processors are not executed in parallel in the sense of multiprocessing, but that the children all either receive
        the same input data or an element of the input data, in contrast to the ``Sequential`` processor group, which first executes the first child
        and then feeds the output to the next child.

    Args:
        is_output (bool, optional): A value indicating whether this ``Parallel`` group processor is the output of a ``Pipeline``. Defaults to `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        children (Iterable[Processor]): The children of the group. This is a list of processors that will be called in parallel.
        broadcast (bool, optional): A value indicating whether the input data should be broadcasted to all children. If ``True``, the input data will
            be copied as many times as there are children. If ``False`` and the input is an iterable, the elements of the iterable will be passed to
            the children one by one. Defaults to `False`.

    Examples:
        >>> Parallel(children=[FunctionProcessor(processing_function=lambda x: x**n) for n in (1, 2, 3, 4)])((2, 2, 2, 2))
        [2, 4, 8, 16]
        >>> Parallel(children=[FunctionProcessor(processing_function=lambda x: x**n) for n in (1, 2, 3, 4)])(2)
        [2, 4, 8, 16]
    """

    broadcast: Annotated[bool, Param(bool, False)]
    """A value indicating whether the input data should be broadcasted to all children. If ``True``, the input data will be copied as many times as
    there are children. If ``False`` and the input is an iterable, the elements of the iterable will be passed to the children one by one. Defaults to
    `False`.
    """

    def function(self, data: Any) -> Any:
        """Invokes the children in parallel, passing the input data to each child. If ``broadcast`` is ``True``, the input data will be copied as many
        times as there are children. If ``broadcast`` is ``False`` and the input is an iterable, the elements of the iterable will be passed to the
        children one by one.

        Args:
            data (Any): The input data to pass to the children. If ``broadcast`` is ``True``, this can be any object. If ``broadcast`` is ``False``,
                and the input is an iterable, the elements of the iterable will be passed to the children one by one.

        Raises:
            TypeError: The ``broadcast`` parameter is set to `True`, and the number of children and number of data elements mismatch.

        Returns:
            Any: Returns a tuple that has the same number of elements as there are children and contains the outputs of the child processors.
        """

        def wrap_iterator_with_meaningful_exception(iterable: Iterable[Any]) -> Iterator[Any]:
            """An iterator that wraps the passed iterator and raises an error, if the number of elements that the iterator produces is not the same as
            the number of child processors. This is done so that the error message is more informative.

            Args:
                iterable (Iterable[Any]): The iterator to wrap.

            Yields:
                Any: The elements of the iterator.

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
        is_output (bool, optional): A value indicating whether this ``Sequential`` group processor is the output of a ``Pipeline``. Defaults to
            `False`.
        is_checkpoint (bool | None, optional): A value indicating whether check-pointed pipeline computations should start at this point, if there
            exists a previously computed checkpoint value. Defaults to `False`.
        io (Storable | None, optional): An IO object that is used to cache intermediate results of the pipeline, which can then be re-used in this
            run or in subsequent runs of the ``Pipeline``. Defaults to an instance of ``NoStorage``.
        children (Iterable[Processor]): The children of the group. This is a list of processors that will be called in sequentially.

    Examples:
        >>> Sequential(children=[FunctionProcessor(processing_function=lambda x: c + x) for c in 'abcd'])('=')
        'dcba='
    """

    def function(self, data: Any) -> Any:
        """Invokes the child processors of the processor group in sequence. The input data is fed to the first child, whose output is then fed into
        the second child, and so on.

        Args:
            data (Any): The input data to pass to the first child.

        Returns:
            Any: Returns the output of the last child processor.
        """

        for child in self.children:
            data = child(data)
        return data
