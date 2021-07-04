from Eir.Deterministic.CompartmentalModel import CompartmentalModel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from multipledispatch import dispatch

from typing import List


class AgeStructuredSIR(CompartmentalModel):
    # beta is the transmission rate, gamma is the recovery rate
    # S0, I0, R0 are the starting Susceptible, infected, and removed people respectively
    """SIR Deterministic Model

    Parameters
    ----------

    labels: List[str]
        Names for each group.

    beta: List[float]
        Effective transmission rate of infected people, on average.

    gamma: List[float]
        Proportion of I that goes to R.

    S0: List[int]
        Initial susceptibles.

    I0: List[int]
        Initial infecteds.

    R0: List[int]
        Initial removed.

    """

    def __init__(
        self,
        labels: List[str],
        beta: List[float],
        gamma: List[float],
        S0: List[int],
        I0: List[int],
        R0: List[int],
    ):
        # run error checks
        self.intCheck([S0, I0, R0])
        self.floatCheck([beta, gamma, S0, I0, R0])
        self.negValCheck([beta, gamma, S0, I0, R0])
        self.lengthCheck([labels, beta, gamma, S0, I0, R0])
        self.probCheck([gamma])
        super().__init__(labels, S0, I0)
        self.R0 = np.array(R0)
        self.beta = np.array(beta)
        self.gamma = np.array(gamma)
        self.N = self.S0 + self.I0 + self.R0

    # meant to change the starting value of S, S0, of SIR object
    def changeS0(self, x: List[int]):
        self.S0 = x
        # after modifying S0, change N accordingly
        self.N = sum(self.S0) + sum(self.I0) + sum(self.R0)

    # meant to change the starting value of I, I0, of SIR object
    def changeI0(self, x: List[int]):
        self.I0 = x
        # after modifying I0, change N accordingly
        self.N = sum(self.S0) + sum(self.I0) + sum(self.R0)

    # meant to change the starting value of R, R0, of SIR object
    def changeR0(self, x: List[int]):
        self.R0 = x
        # after modifying R0, change N accordingly
        self.N = sum(self.S0) + sum(self.I0) + sum(self.R0)

    # meant to change the value of beta of SIR object
    def changeBeta(self, x: List[float]):
        self.beta = x

    # meant to change the value of gamma of SIR object
    def changeGamma(self, x: List[float]):
        self.gamma = x

    # computes the derivatives at a particular point, given the beta/gamma of object and current s and i values
    # @dispatch(float, float)
    def _deriv(self, s: np.ndarray, i: np.ndarray):
        # amount leaving I -> R
        y = self.gamma * i
        # amount leaving S -> I
        x = np.zeros_like(y)
        for n in range(len(x)):
            x[n] = (self.beta[n] * i * s[n] / self.N).sum()

        # returns in the order S, I, R
        return -x, x - y, y

    # runs Euler's Method
    # @dispatch(float, np.ndarray, np.ndarray, np.ndarray)
    def _update(self, dt: float, S: np.ndarray, I: np.ndarray, R: np.ndarray):
        # for all the days that ODE will be solved
        for i in range(1, len(S[0])):
            # get the derivatives at the point before for the Euler's method
            f = self._deriv(S[:, i - 1], I[:, i - 1])
            # computer the Euler's approximation f(x+h) = f(x) + h * (df/dx)
            S[:, i] = S[:, i - 1] + dt * f[0]
            I[:, i] = I[:, i - 1] + dt * f[1]
            R[:, i] = R[:, i - 1] + dt * f[2]
        return S, I, R

    # combines the Euler's Method with all initialization and stuff and runs full simulation
    # days is the number of days being simulated, dt is the step size for Euler's method
    def _simulate(self, days: int, dt: float):
        # total number of iterations that will be run + the starting value at time 0
        size = int(days / dt + 1)
        # create the arrays to store the different values
        l = len(self.S0)
        S, I, R = np.zeros((l, size)), np.zeros((l, size)), np.zeros((l, size))
        # initialize the arrays
        S[:, 0], I[:, 0], R[:, 0] = self.S0, self.I0, self.R0
        # run the Euler's Method
        S, I, R = self._update(dt, S, I, R)
        return S, I, R

    def run(self, days: int, dt: float, plot=True):
        self.floatCheck([[days], [dt]])
        self.negValCheck([[days], [dt]])

        t = np.linspace(0, days, int(days / dt) + 1)
        S, I, R = self._simulate(days, dt)
        data1 = {
            "Days": t,
            "Susceptible": S.sum(axis=0),
            "Infected": I.sum(axis=0),
            "Removed": R.sum(axis=0),
        }
        df = pd.DataFrame.from_dict(data=data1)
        for i, group in enumerate(self.labels):
            data_group = {
                "Days": t,
                f"Susceptible_{group}": S[i],
                f"Infected_{group}": I[i],
                f"Removed_{group}": R[i],
            }
            df_group = pd.DataFrame.from_dict(data_group)
            df = pd.merge(df, df_group, on="Days")

        if plot:
            fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=False)
            ax = ax.flatten()
            for i, c in enumerate(df.columns):
                if i < 4:
                    ax[0].plot(df["Days"], df[c], label=c)
                else:
                    ax[(i - 3) % 3 + 1].plot(df["Days"], df[c], label=c)

            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            ax[3].legend()
            ax[2].set_xlabel("Number of Days")
            ax[3].set_xlabel("Number of Days")
            ax[0].set_ylabel("Number of people")
            ax[2].set_ylabel("Number of People")
            plt.close()

            return df, fig

        return df

    # plot an accumulation function of total cases
    def accumulate(self, days: int, dt: float, plot=True):
        self.floatCheck([[days], [dt]])
        self.negValCheck([[days], [dt]])
        t = np.linspace(0, days, int(days / dt) + 1)
        S, I, R = self._simulate(days, dt)
        # create a numpy array that will hold all of the values
        cases = np.zeros((len(I), len(I[0])))
        # add up the total infected and removed at given time to account for everyone with the virus
        for i in range(len(I)):
            for j in range(len(I[0])):
                cases[i][j] = I[i][j] + R[i][j]
        # create a dictionary that holds the data for easy conversion to dataframe
        data1 = {
            "Days": t,
            "Susceptible": S.sum(axis=0),
            "Infected": I.sum(axis=0),
            "Removed": R.sum(axis=0),
            "Total Cases": cases.sum(axis=0),
        }
        df = pd.DataFrame.from_dict(data=data1)
        if plot:
            # do some plotting
            df.plot(x="Days", y=["Total Cases"])
            plt.xlabel("Days")
            plt.ylabel("Total Cases")
            plt.show()
        # return dataframe
        return df

    # create everything as a percentage of the total population, given a dataframe
    def normalizeDataFrame(self, df: pd.DataFrame):
        """
        Divide all of the columns by the total population in order to get numbers as a proportion of population.

        Parameters
        ----------

        df: pd.DataFrame
            The dataframe to be normalized

        Returns
        -------

        pd.DataFrame
            Normalized dataframe.
        """
        colnames = list(df.columns)
        colnames.pop(0)
        for i in colnames:
            df[i] = df[i].div(self.N)
        return df

    def normalizeRun(self, days: int, dt: float, accumulate=True):
        """
        Does a normalized run of the simulation.

        Parameters
        ---------

        days: int
            Number of days being simulated.

        dt: float
            The differential used for Euler's method.

        """
        df: pd.DataFrame
        if accumulate:
            df = self.accumulate(days, dt, plot=False)
            colnames = list(df.columns)
            # get rid of the days column in the list
            colnames.pop(0)
            for i in colnames:
                df[i] = df[i].div(self.N)
        else:
            df = self.run(days, dt, plot=False)
            colnames = list(df.columns)
            colnames.pop(0)
            for i in colnames:
                df[i] = df[i].div(self.N)
        return df
