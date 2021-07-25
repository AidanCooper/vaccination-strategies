from .AgeStructuredSIR import AgeStructuredSIR
import numpy as np
from matplotlib import pyplot as plt
from multipledispatch import dispatch
import pandas as pd

from typing import List


class AgeStructuredSIRD(AgeStructuredSIR):
    """
    SIRD deterministic model.

    Parameters
    ----------

    labels: List[str]
        Names for each group.

    beta: np.array
        Matrix of r values between pairs of groups.

    gamma: float
        Proportion of people who go from I to R

    omega: float
        Proportion of people who go from I to D.

    S0: int
        Initial number of susceptibles

    I0: int
        Initial number of infecteds

    R0: int
        Initial number of removeds.
    """

    # omega is the amount of people that go from I to D
    def __init__(
        self,
        labels: List[str],
        beta: List[float],
        gamma: List[float],
        omega: List[float],
        S0: List[int],
        I0: List[int],
        R0: List[int],
    ):
        self.intCheck([S0, I0, R0])
        self.floatCheck([gamma, omega, S0, I0, R0])
        self.negValCheck([gamma, omega, S0, I0, R0])
        self.probCheck([gamma, omega])
        super().__init__(labels, beta, gamma, S0, I0, R0)
        self.omega = np.array(omega)
        for g, o in zip(self.gamma, self.omega):
            assert g + o <= 1

    # change the variable omega
    def changeOmega(self, x: List[float]):
        self.omega = x

    # @dispatch(float, float, int)
    # calculate the derivatives; because open pop, feed in the current alive population
    def _deriv(self, s: np.ndarray, i: np.ndarray, n: np.ndarray):
        # amount leaving I -> R
        y = self.gamma * i
        # amount leaving I -> D
        z = self.omega * i
        # amount leaving S -> I
        x = np.zeros_like(y)
        for j in range(len(x)):
            x[j] = (self.beta[j] * i * s[j] / n.sum()).sum()

        # returns in the order S, I, R, D
        return -x, x - y - z, y, z

    # run Euler's method
    # @dispatch(float, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def _update(
        self, dt: float, S: np.ndarray, I: np.ndarray, R: np.ndarray, D: np.ndarray
    ) -> tuple:
        # run Euler's method
        for i in range(1, len(S[0])):
            n = S[:, i - 1] + I[:, i - 1] + R[:, i - 1]  # living population

            # get the derivatives at the point before for the Euler's method
            f = self._deriv(S[:, i - 1], I[:, i - 1], n)
            # computer the Euler's approximation f(x+h) = f(x) + h * (df/dx)
            S[:, i] = [max(v, 0) for v in S[:, i - 1] + dt * f[0]]
            I[:, i] = I[:, i - 1] + dt * f[1]
            R[:, i] = R[:, i - 1] + dt * f[2]
            D[:, i] = D[:, i - 1] + dt * f[3]

        return S, I, R, D

    def _simulate(self, days: int, dt: float):
        # total number of iterations that will be run + the starting value at time 0
        size = int(days / dt + 1)
        # create the arrays to store the different values
        l = len(self.S0)
        S, I, R, D = (
            np.zeros((l, size)),
            np.zeros((l, size)),
            np.zeros((l, size)),
            np.zeros((l, size)),
        )
        # initialize the arrays
        S[:, 0], I[:, 0], R[:, 0], D[:, 0] = (
            self.S0,
            self.I0,
            self.R0,
            np.zeros_like(self.S0),
        )
        # run the Euler's Method
        S, I, R, D = self._update(dt, S, I, R, D)
        return S, I, R, D

    # @dispatch(int, float, plot=bool, Sbool=bool, Ibool=bool, Rbool=bool, Dbool=bool)
    def run(
        self,
        days: int,
        dt: float,
        plot=True,
    ):
        self.floatCheck([[days], [dt]])
        self.negValCheck([[days], [dt]])

        t = np.linspace(0, days, int(days / dt) + 1)
        S, I, R, D = self._simulate(days, dt)
        data1 = {
            "Days": t,
            "Susceptible": S.sum(axis=0),
            "Infected": I.sum(axis=0),
            "Removed": R.sum(axis=0),
            "Deaths": D.sum(axis=0),
        }
        df = pd.DataFrame.from_dict(data=data1)
        for i, group in enumerate(self.labels):
            data_group = {
                "Days": t,
                f"Susceptible_{group}": S[i],
                f"Infected_{group}": I[i],
                f"Removed_{group}": R[i],
                f"Deaths_{group}": D[i],
            }
            df_group = pd.DataFrame.from_dict(data_group)
            df = pd.merge(df, df_group, on="Days")

        if plot:
            fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=False)
            ax = ax.flatten()
            for i, c in enumerate(df.columns):
                if i != 0:
                    if i < 5:
                        ax[0].plot(df["Days"], df[c], label=c)
                    else:
                        ax[(i - 4) % 4 + 1].plot(df["Days"], df[c], label=c)

            for i, a in enumerate(ax[:-1]):
                a.legend()
                if i % 3 == 0:
                    a.set_ylabel("Number of People")
                if i in [3, 4]:
                    a.set_xlabel("Number of Days")
            ax[5].remove()
            plt.close()

            self.df_results_ = df

            return df, fig

        self.df_results_ = df

        return df

    # plot an accumulation function of total cases
    def accumulate(self, days: int, dt: float, plot=True):
        self.floatCheck([[days], [dt]])
        self.negValCheck([[days], [dt]])
        t = np.linspace(0, days, int(days / dt) + 1)
        S, I, R, D = self._simulate(days, dt)
        # create a numpy array that will hold all of the values
        cases = np.zeros((len(I), len(I[0])))
        # add up the total infected and removed at given time to account for everyone with the virus
        for i in range(len(I)):
            for j in range(len(I[0])):
                cases[i][j] = I[i][j] + R[i][j] + D[i][j]
        # create a dictionary that holds the data for easy conversion to dataframe
        data1 = {
            "Days": t,
            "Susceptible": S.sum(axis=0),
            "Infected": I.sum(axis=0),
            "Removed": R.sum(axis=0),
            "Deaths": D.sum(axis=0),
            "Total Cases": cases.sum(axis=0),
        }
        df = pd.DataFrame.from_dict(data1)
        if plot:
            # do some plotting
            df.plot(x="Days", y=["Total Cases"])
            plt.xlabel("Days")
            plt.ylabel("Total Cases")
            plt.show()
        # return dataframe
        return df

    def end_state(self) -> pd.DataFrame:
        df_end = pd.DataFrame(index=[["Combined"] + self.labels])
        df_end["Start_Count"] = np.array([sum(self.N)] + list(self.N))

        df_end["End_Susceptible"] = (
            self.df_results_.iloc[-1][
                ["Susceptible"] + [f"Susceptible_{l}" for l in self.labels]
            ]
            .astype(int)
            .values
        )
        df_end["Infected_Count"] = 0  # placeholder - populated later
        df_end["Removed_Count"] = (
            self.df_results_.iloc[-1][
                ["Removed"] + [f"Removed_{l}" for l in self.labels]
            ]
            .astype(int)
            .values
        )
        df_end["Infected_Count"] = (
            df_end["Start_Count"]
            - df_end["End_Susceptible"]
            - np.array([sum(self.R0)] + list(self.R0))
        )
        df_end["Deaths_Count"] = (
            self.df_results_.iloc[-1][["Deaths"] + [f"Deaths_{l}" for l in self.labels]]
            .astype(int)
            .values
        )
        df_end["Fatality_Rate%"] = (
            df_end["Deaths_Count"]
            / (
                df_end["Removed_Count"]
                + df_end["Deaths_Count"]
                - np.array([sum(self.R0)] + list(self.R0))
            )
            * 100
        )

        return df_end