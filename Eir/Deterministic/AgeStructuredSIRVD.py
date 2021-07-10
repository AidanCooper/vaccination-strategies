from .AgeStructuredSIRD import AgeStructuredSIRD
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from multipledispatch import dispatch

from typing import List


# Flow of the Compartmental Model:
# S -> I -> R, S -> I -> D, S -> V
class AgeStructuredSIRVD(AgeStructuredSIRD):
    """
    SIRV deterministic model.

    Parameters
    ----------

    beta: float
        Effective transmission rate of an infectious person, on average.

    gamma: float
        Proportion of people that go from I to R.

    omega: float
        Proportion of people who go from I to D.

    Vd: int
        Count of people that get vaccinated each day.

    S0: int
        Initial susceptibles at the start of the simulation.

    I0: int
        Initial infecteds at the start of the simulation.

    R0: int
        Initial recovereds at the start of the simulation.

    V0: int
        Initial vaccinated at the start of the simulation.
    """

    def __init__(
        self,
        labels: List[str],
        beta: List[float],
        gamma: List[float],
        omega: List[float],
        Vd: int,
        S0: List[int],
        I0: List[int],
        R0: List[int],
        V0: List[int],
    ):
        self.intCheck([S0, I0, R0, V0, [Vd]])
        self.floatCheck([gamma, omega])
        self.negValCheck([gamma, omega, S0, I0, R0, V0, [Vd]])
        self.lengthCheck([labels, beta[0], gamma, omega, S0, I0, R0, V0])
        self.probCheck([gamma, omega])
        super().__init__(labels, beta, gamma, omega, S0, I0, R0)
        self.V0 = np.array(V0)
        self.Vd = Vd
        for g, o in zip(self.gamma, self.omega):
            assert g + o <= 1
        self.N = S0 + I0 + R0 + V0

    def changeS0(self, x: List[int]):
        self.S0 = x
        # after modifying S0, change N accordingly
        self.N = self.S0 + self.I0 + self.R0 + self.V0

    def changeI0(self, x: List[int]):
        self.I0 = x
        # after modifying I0, change N accordingly
        self.N = self.S0 + self.I0 + self.R0 + self.V0

    def changeR0(self, x: List[int]):
        self.R0 = x
        # after modifying R0, change N accordingly
        self.N = self.S0 + self.I0 + self.R0 + self.V0

    def changeV0(self, x: int):
        self.V0 = x
        self.N = self.S0 + self.I0 + self.R0 + self.V0

    def changeEta(self, x: int):
        self.eta = x

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
            x[j] = (self.beta[j] * i * s[j] / n[j]).sum()

        # returns in the order S, I, R, D
        return -x, x - y - z, y, z

    def _vaccinate(self, S, V, dt, reverse: bool):
        if reverse:
            s = S[::-1].copy()
            v = V[::-1].copy()
            for i, Si in enumerate(S[::-1]):
                if Si > 0:
                    if Si > dt * self.Vd:
                        s[i] -= dt * self.Vd
                        v[i] += dt * self.Vd
                        break
                    else:
                        s[i] = 0
                        v[i] += Si
                        break
            s = s[::-1]
            v = v[::-1]
        else:
            s = S.copy()
            v = V.copy()
            for i, Si in enumerate(S):
                if Si > 0:
                    if Si > dt * self.Vd:
                        s[i] -= dt * self.Vd
                        v[i] += dt * self.Vd
                        break
                    else:
                        s[i] = 0
                        v[i] += Si
                        break
        return s, v

    # run Euler's method
    # @dispatch(float, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def _update(self, dt: float, S, I, R, V, D, reverse: bool):
        # run Euler's method
        for i in range(1, len(S[0])):
            # vaccinate
            s, V[:, i] = self._vaccinate(S[:, i - 1], V[:, i - 1], dt, reverse)

            n = np.zeros(S.shape[0])
            for j in range(len(n)):
                n[j] = int(s[j] + I[j][i - 1] + R[j][i - 1])

            # get the derivatives at a point b`efore
            f = self._deriv(s, I[:, i - 1], n)
            # compute the euler's method: f(x+h) = f(x) + h * (df/dx)
            S[:, i] = s + dt * f[0]  # susceptible pop after vaccination and infection
            I[:, i] = I[:, i - 1] + dt * f[1]
            R[:, i] = R[:, i - 1] + dt * f[2]
            D[:, i] = D[:, i - 1] + dt * f[3]

        return S, I, R, V, D

    def _simulate(self, days: int, dt: float, reverse: bool):
        # total number of iterations that will be run + the starting value at time 0
        size = int(days / dt + 1)
        # create the arrays to store the different values
        l = len(self.S0)
        S, I, R, D, V = (
            np.zeros((l, size)),
            np.zeros((l, size)),
            np.zeros((l, size)),
            np.zeros((l, size)),
            np.zeros((l, size)),
        )
        # initialize the arrays
        S[:, 0], I[:, 0], R[:, 0], V[:, 0], D[:, 0] = (
            self.S0,
            self.I0,
            self.R0,
            np.zeros_like(self.S0),
            np.zeros_like(self.S0),
        )
        # run the Euler's Method
        S, I, R, V, D = self._update(dt, S, I, R, V, D, reverse)
        return S, I, R, V, D

    # @dispatch(int, float, plot=bool, Sbool=bool, Ibool=bool, Rbool=bool, Vbool=bool)
    def run(
        self,
        days: int,
        dt: float,
        reverse: bool = False,
        plot=True,
    ):
        self.floatCheck([[days], [dt]])
        self.negValCheck([[days], [dt]])

        t = np.linspace(0, days, int(days / dt) + 1)
        S, I, R, V, D = self._simulate(days, dt, reverse)
        data1 = {
            "Days": t,
            "Susceptible": S.sum(axis=0),
            "Infected": I.sum(axis=0),
            "Removed": R.sum(axis=0),
            "Vaccinated": V.sum(axis=0),
            "Deaths": D.sum(axis=0),
        }
        df = pd.DataFrame.from_dict(data=data1)
        for i, group in enumerate(self.labels):
            data_group = {
                "Days": t,
                f"Susceptible_{group}": S[i],
                f"Infected_{group}": I[i],
                f"Removed_{group}": R[i],
                f"Vaccinated_{group}": V[i],
                f"Deaths_{group}": D[i],
            }
            df_group = pd.DataFrame.from_dict(data_group)
            df = pd.merge(df, df_group, on="Days")

        if plot:
            fig, ax = plt.subplots(3, 2, figsize=(12, 16), sharex=True, sharey=False)
            ax = ax.flatten()
            for i, c in enumerate(df.columns):
                if i != 0:
                    if i < 6:
                        ax[0].plot(df["Days"], df[c], label=c)
                    else:
                        ax[(i - 5) % 5 + 1].plot(df["Days"], df[c], label=c)

            for i, a in enumerate(ax):
                a.legend()
                if i % 2 == 0:
                    a.set_ylabel("Number of People")
                if i in [4, 5]:
                    a.set_xlabel("Number of Days")
            plt.close()

            return df, fig
        return df

    # plot an accumulation function of total cases
    def accumulate(self, days: int, dt: float, plot=True):
        self.floatCheck([[days], [dt]])
        self.negValCheck([[days], [dt]])
        t = np.linspace(0, days, int(days / dt) + 1)
        S, I, R, D = self._simulate(days, dt)
        # create a numpy array that will hold all of the values
        cases = np.zeros((len(I), len(I[0])))
        # add up the total infected/removed/deceased at given time
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
