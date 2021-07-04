import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from multipledispatch import dispatch
import Eir.exceptions as e

from typing import List

# sources:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5348083/


# this is an abstract class that should never be instantiated
class CompartmentalModel:
    """ Base Class for all Deterministic Compartmental Models. Should never be instantiated."""

    def __init__(self, labels: List[str], S0: List[int], I0: List[int]):
        for s in S0:
            assert s >= 0
        for i in I0:
            assert i >= 0
        self.labels = labels
        self.S0 = np.array(S0)
        self.I0 = np.array(I0)

    @dispatch()
    def _deriv(self):
        pass

    @dispatch(float)
    # runs the Euler's Method
    def _update(self, dt: float):
        pass

    # creates the arrays & starting items, then calls _update() to run Euler's Method
    # then returns the completed arrays
    def _simulate(self, days: int, dt: float):
        pass

    @dispatch(int, float, bool)
    def run(self, days: int, dt: float, plot=True):
        pass

    def intCheck(self, vals: list):
        for val in vals:
            for v in val:
                if type(v) != int:
                    raise e.NotIntException(v)

    def floatCheck(self, vals: list):
        for val in vals:
            for v in val:
                if type(v) != int and type(v) != float:
                    raise e.NotFloatException(v)

    def negValCheck(self, vals: list):
        for val in vals:
            for v in val:
                if v < 0:
                    raise e.NegativeValException(v)

    def probCheck(self, vals: list):
        for val in vals:
            for v in val:
                if not 0 <= v <= 1:
                    raise e.ProbabilityException(v)

    def lengthCheck(self, vals: list):
        lengths = np.array([len(val) for val in vals])
        if len(set(lengths)) != 1:
            raise e.LengthException(lengths)
