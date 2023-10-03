
r"""
Modified from Habitat codebase

Import the global registry object using

.. code:: py

    from lctgen.core.registry import registry

Various decorators for registry different kind of classes with unique keys

-   Register a model: ``@registry.register_model``
-   Register a metric: ``@registry.register_metric``
-   Register a dataset: ``@registry.register_dataset``
-   Register a LLM model: ``@registry.register_llm``

"""

import collections
from typing import Any, Callable, DefaultDict, Optional, Type

from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from torchmetrics import Metric
from lctgen.core.basic import BasicLLM

class Registry():
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_dataset(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a dataset to registry with key :p:`name`

        :param name: Key with which the metric will be registered.
            If :py:`None` will use the name of the class
        """

        return cls._register_impl(
            "dataset", to_register, name, assert_type=Dataset
        )

    @classmethod
    def register_metric(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a metric to registry with key :p:`name`

        :param name: Key with which the metric will be registered.
            If :py:`None` will use the name of the class
        """

        return cls._register_impl(
            "metric", to_register, name, assert_type=Metric
        )

    @classmethod
    def register_model(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a model to registry with key :p:`name`

        :param name: Key with which the metric will be registered.
            If :py:`None` will use the name of the class
        """

        return cls._register_impl(
            "model", to_register, name, assert_type=LightningModule
        )

    @classmethod
    def register_llm(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a LLM model to registry with key :p:`name`

        :param name: Key with which the metric will be registered.
            If :py:`None` will use the name of the class
        """
        return cls._register_impl(
            "llm", to_register, name, assert_type=BasicLLM
        )

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_dataset(cls, name: str) -> Type[Dataset]:
        return cls._get_impl("dataset", name)

    @classmethod
    def get_metric(cls, name: str) -> Type[Metric]:
        return cls._get_impl("metric", name)

    @classmethod
    def get_model(cls, name: str) -> Type[LightningModule]:
        return cls._get_impl("model", name)

    @classmethod
    def get_llm(cls, name: str) -> Type[BasicLLM]:
        return cls._get_impl("llm", name)

registry = Registry()
