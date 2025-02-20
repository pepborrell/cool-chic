from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


class ConstantIterable(Iterable[T]):
    def __init__(self, value: T):
        self.value = value

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        return self.value
