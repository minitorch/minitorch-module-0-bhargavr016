"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable, List, TypeVar

T = TypeVar('T')
U = TypeVar('U')

# Implementation of a prelude of elementary functions.

def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    return (1.0 / (1.0 + math.pow(math.e, -x))) if x >= 0 else (math.pow(math.e, x) / (1.0 + math.pow(math.e, x)))

def relu(x: float) -> float:
    return max(x, 0.0)

EPS = 1e-6

def log(x: float) -> float:
    return math.log(x + EPS)

def exp(x: float) -> float:
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    return d * ((x * log(x)) - x)

def inv(x: float) -> float:
    return 1 / x

def inv_back(x: float, d: float) -> float:
    return d * (-(1.0 / (x ** 2)))

def relu_back(x: float, d: float) -> float:
    return d * (1.0 if x > 0.0 else 0.0)

# Task 0.3 - Small practice library of elementary higher-order functions.

def map(fn: Callable[[T], U]) -> Callable[[Iterable[T]], List[U]]:
    return lambda iterable_obj: [fn(item) for item in iterable_obj]

def negList(ls: List[float]) -> List[float]:
    return map(neg)(ls)

def zipWith(fn: Callable[[T, U], U]) -> Callable[[Iterable[T], Iterable[T]], List[U]]:
    return lambda iterable1, iterable2: [fn(x, y) for x, y in zip(iterable1, iterable2)]

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)

def reduce(fn: Callable[[U, T], U], start: U) -> Callable[[Iterable[T]], U]:
    def reducer(iterable: Iterable[T]) -> U:
        reduce_out = start
        for item in iterable:
            reduce_out = fn(reduce_out, item)
        return reduce_out
    return reducer

def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)

def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)
