===============================
Migrating from v0.2.* to v1.0.*
===============================

Between v0.2.* and v1.0.*, CoRelAy has made some significant changes, including, but not limited to, the following:

* The project was converted from a ``setup.py`` project to a ``uv`` project.
* The Python dependencies were updated to the latest versions.
* The linting of the project was improved.
* The docstrings of the code were improved.
* Type hints were added to the code.
* The code was refactored to improve readability and maintainability.
* The test coverage was improved.
* The documentation was improved.

For more information on the changes made in this version, please refer to the :repo:`CHANGELOG.md` file in the root directory of the project.

The most important change, however, is how slots (:py:class:`~corelay.plugboard.Slot`) are defined. Slots are the underlying mechanism for parameters (:py:class:`~corelay.base.Param`) of processors (:py:class:`~corelay.processor.base.Processor`) and tasks (:py:class:`~corelay.pipeline.base.Task`) of pipelines (:py:class:`~corelay.pipeline.base.Pipeline`). In the new version, instead of assigning an instance of a slot (:py:class:`~corelay.base.Param` or :py:class:`~corelay.pipeline.base.Task`) to a class attribute (of a :py:class:`~corelay.processor.base.Processor` or :py:class:`~corelay.pipeline.base.Pipeline`), the slot is now defined by declaring a class attribute of type :py:class:`~typing.Annotated` with the data type of the slot as the first argument and the instance of the slot as the second argument.

This change was made necessary because the project now uses type hints and MyPy to statically type check the code. In Python, class attributes can be accessed using the class or the instance. For example, if a class ``Test`` has a class attribute `a = 5`, then `Test.a` and `Test().a` will both return `5`. Static type checkers, like MyPy, do not know that we are dynamically converting class attributes that are slots to an instance attribute of the specified data type at runtime, so they will assume that when a slot is accessed using the instance, it will have the same type as the class attribute. This means that the static type checker will assume, that a slot, defined as `parameter: Param(int, 5)`, will be of type :py:class:`~corelay.base.Param` and not :py:class:`int`, and will produce a type error when the slot is accessed using the instance.

:py:class:`~typing.Annotated` is a special type that allows us to annotate variables, parameters, and attributes with metadata. It signals to static type checkers that the variable, parameter, or attribute is of the type specified in the first argument. So a slot, defined as `parameter: Annotated[int, Param(int, 5)]`, will present to MyPy as an :py:class:`int` and will not produce a type error when accessed using the instance. The second argument of :py:class:`~typing.Annotated` is the instance of the slot, which is used to define the slot at runtime. This change allows us to use type hints and MyPy to statically type check the code, while still being able to define slots dynamically at runtime.

To update your code, you will need to change all slot definitions, i.e., all uses of :py:class:`~corelay.base.Param` and :py:class:`~corelay.pipeline.base.Task`, to use the new syntax. For example, if you have a processor that looks like this:

.. code-block:: python

    from typing import Annotated, Any

    from corelay.base import Param
    from corelay.processor.base import Processor

    class MyProcessor(Processor):
        """An example processor."""

        parameter = Param(int, 5)
        """A parameter that was defined using the old syntax."""

        def function(self, data: Any) -> Any:
            """Applies the processor to the input data.

            Args:
                data (Any): The input data that is to be processed.

            Returns:
                Any: Returns the processed data.
            """

            return data + self.parameter

You will need to change it to look like this:

.. code-block:: python

    from typing import Annotated, Any

    from corelay.base import Param
    from corelay.processor.base import Processor

    class MyProcessor(Processor):
        """An example processor."""

        parameter: Annotated[int, Param(int, 5)]
        """A parameter that was defined using the new syntax."""

        def function(self, data: Any) -> Any:
            """Applies the processor to the input data.

            Args:
                data (Any): The input data that is to be processed.

            Returns:
                Any: Returns the processed data.
            """

            return data + self.parameter

For the time being, the old syntax is still supported for backward compatibility. It is, however, highly recommended to update your code to use the new syntax, as the old syntax will be removed in a future version of CoRelAy. Also, this will cause an error when you are using a static type checker like MyPy.
