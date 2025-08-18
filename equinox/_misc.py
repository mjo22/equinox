from typing import Any, Generic, TypeVar

import dataclasses
import jax
import jax.core
import jax.numpy as jnp
from jaxtyping import Array

from ._filters import is_array


T = TypeVar("T")


@dataclasses.dataclass(frozen=True)  # not a pytree
class if_array(Generic[T]):
    """Returns a callable that returns the specified argument if evaluated on an array.
    Otherwise, it returns `None`.

    !!! Example

        ```python
        fn = if_array(1)
        # Evaluate on an array, return the integer.
        fn(jax.numpy.array([0, 1, 2]))  # 1
        # Evaluate on not-an-array, return None.
        fn(True)  # None
        ```
    """

    value: T

    def __call__(self, x: Any) -> T | None:
        return self.value if is_array(x) else None


def left_broadcast_to(arr: Array, shape: tuple[int, ...]) -> Array:
    arr = arr.reshape(arr.shape + (1,) * (len(shape) - arr.ndim))
    return jnp.broadcast_to(arr, shape)


def currently_jitting():
    return isinstance(jnp.array(1) + 1, jax.core.Tracer)


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32
