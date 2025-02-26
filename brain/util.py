from typing import Callable
from inspect import signature

def only_kwargs_from_fn(fn: Callable, kwargs: dict) -> dict:
    """
        get a dictionary of only the kwargs that the function has as parameters;
        so, take a set of kwargs and return the kwargs that the function accepts.
    """
    return {
        name: kwargs[name] 
        for name, _ in signature(fn).parameters.items() 
        if name in kwargs
    }