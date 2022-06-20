"""Basic flow operation Processors, such as Shaper, Sequential and Parallel"""
from ..base import Param
from .base import Processor
from ..utils import zip_equal, Iterable


class Shaper(Processor):
    """Extracts and/ or copies by indices.

    Attributes
    ----------
    indices : iterable of (int or iterable of int)
        Iterable of indices to copy/ extract. The resuling output will be a tuple with the same member shape. Each index
        may be passed an arbitrary amount of times. Outer tuples allow ints and tuples, inner tuples only allow ints.

    Examples
    -------
    >>> Shaper(indices=(0, 1, (0, 1, 2)))(['a', 'b', 'c'])
    ('a', 'b', ('a', 'b', 'c'))

    """
    indices = Param(Iterable[int, Iterable[int]], positional=True)

    def function(self, data):
        """Extracts and/ or copies indices of data.

        Parameters
        ----------
        data : object
            Object from which to extract/ copy elements. If not an iterable, index 0 corresponds to the object itself.

        Returns
        -------
        tuple of object
            The extracted/ copied indices of data.

        Raises
        ------
        TypeError
            If an invalid index was accessed for data.

        """
        def extract_indices(iterable, indices):
            """Recursively extract indices"""
            result = []
            for index in indices:
                if isinstance(index, Iterable):
                    obj = extract_indices(iterable, index)
                else:
                    try:
                        obj = data[index]
                    except KeyError as err:
                        raise TypeError(f"'{index}' is not a valid index for '{data}'") from err
                result.append(obj)
            return tuple(result)

        if not isinstance(data, Iterable):
            data = (data,)
        result = extract_indices(data, self.indices)
        return result


class GroupProcessor(Processor):
    """Abstract class for groups of Processors.

    Attributes
    ----------
    children : iterable of :obj:`Processor`
        Child Processors for this group.

    """
    children = Param(Iterable[Processor], positional=True)


class Parallel(GroupProcessor):
    """Processor group calling its children in Parallel.

    Examples
    --------
    >>> Parallel(children=[FunctionProcessor(function=lambda x: x**n) for n in (1, 2, 3, 4)])((2, 2, 2, 2))
    [2, 4, 8, 16]
    >>> Parallel(children=[FunctionProcessor(function=lambda x: x**n) for n in (1, 2, 3, 4)])(2)
    [2, 4, 8, 16]

    """
    broadcast = Param(bool, False)

    def function(self, data):
        """Sequentially get one element from data per child, call all children with this element as input in parallel,
        and accumulate the outputs.

        Parameters
        ----------
        data : iterable or object
            Iterable from which to pass the elements to the children. If data is not an Iterable, it will be copied as
            many times as there are children.

        Raises
        ------
        TypeError
            If the number of children and number of data elements mismatch.

        """
        def wrap_err(iterable):
            """Wraps around iterator to raise correct error"""
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except TypeError as err:
                    raise TypeError("Number of data elements and children does not match!") from err
                except StopIteration:
                    break

        if not isinstance(data, Iterable) or self.broadcast:
            # pylint: disable=not-an-iterable
            data = tuple(data for _ in self.children)

        result = []
        for child, element in wrap_err(zip_equal(self.children, data)):
            out = child(element)
            result.append(out)
        return tuple(result)


class Sequential(GroupProcessor):
    """Processor group calling its children in Sequence, feeding the input the first child, and then each output to the
    next child.

    Examples
    --------
    >>> Sequential(children=[FunctionProcessor(function=lambda x: c + x) for c in 'abcd'])('=')
    'dcba='

    """
    def function(self, data):
        """Feed data forward sequentially, passing each child's output to the next child.

        Parameters
        ----------
        data : object
            Input data to pass to the first child.

        """
        # pylint: disable=not-an-iterable
        for child in self.children:
            data = child(data)
        return data
