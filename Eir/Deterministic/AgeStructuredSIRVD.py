from .AgeStructuredSIRD import AgeStructuredSIRD
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from multipledispatch import dispatch

from typing import List, Optional


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
        self.N = self.S0 + self.I0 + self.R0 + self.V0

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
            x[j] = (self.beta[j] * i * s[j] / n.sum()).sum()

        # returns in the order S, I, R, D
        return -x, x - y - z, y, z

    def _vaccinate(self, S, V, dt, reverse: bool, delay: int, i):
        if i * dt < delay:
            return S, V
        elif reverse:
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
    def _update(self, dt: float, S, I, R, V, D, reverse: bool, delay: int):
        # run Euler's method
        for i in range(1, len(S[0])):
            # vaccinate
            s, V[:, i] = self._vaccinate(
                S[:, i - 1], V[:, i - 1], dt, reverse, delay, i
            )

            n = s + I[:, i - 1] + R[:, i - 1] + V[:, i - 1]  # living population

            # get the derivatives at a point before
            f = self._deriv(s, I[:, i - 1], n)
            # compute the euler's method: f(x+h) = f(x) + h * (df/dx)
            S[:, i] = [
                max(v, 0) for v in s + dt * f[0]
            ]  # susceptible pop after vaccination and infection
            I[:, i] = I[:, i - 1] + dt * f[1]
            R[:, i] = R[:, i - 1] + dt * f[2]
            D[:, i] = D[:, i - 1] + dt * f[3]

        return S, I, R, V, D

    def _simulate(self, days: int, dt: float, reverse: bool, delay: int):
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
        S, I, R, V, D = self._update(dt, S, I, R, V, D, reverse, delay)
        return S, I, R, V, D

    # @dispatch(int, float, plot=bool, Sbool=bool, Ibool=bool, Rbool=bool, Vbool=bool)
    def run(
        self,
        days: int,
        dt: float,
        reverse_vaccination: bool = False,
        delay_vaccination: int = 0,
        plot=True,
    ) -> (pd.DataFrame, Optional[Figure]):
        """
        Parameters
        ----------
        days : int
            Number of days for simulation.
        dt : float
            Increment of simulation as a fraction of `days`.
        reverse_vaccination : bool
            If False, vaccinate the groups in the order they appear in `self.labels`. If
            True, vaccinate the groups in reverse order.
        delay_vaccination : int
            Optionally delay vaccination rollout by this many days.
        plot : bool
            Optionally plot model results.

        Returns
        -------
        (pd.DataFrame, Optional[Figure])
            pandas DataFrame of results at each simulation step, and optional plot.
        """
        self.floatCheck([[days], [dt]])
        self.negValCheck([[days], [dt]])

        t = np.linspace(0, days, int(days / dt) + 1)
        S, I, R, V, D = self._simulate(days, dt, reverse_vaccination, delay_vaccination)
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
        df_end["Vaccinated_Count"] = (
            self.df_results_.iloc[-1][
                ["Vaccinated"] + [f"Vaccinated_{l}" for l in self.labels]
            ]
            .astype(int)
            .values
        )
        df_end["Infected_Count"] = (
            df_end["Start_Count"]
            - df_end["End_Susceptible"]
            - df_end["Vaccinated_Count"]
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