import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.DTMC.spatialModel.Hub.HubSIRS import HubSIRS
from src.utility import dist, randEvent, Person

class StrongInfSIRS(HubSIRS):
    def __init__(self, pss: float, rstart: float, alpha: int, side: float, S0: int, I0: int, R0: int, days:int, gamma: float, kappa: float, w0=.7):
        # error checking
        self.intCheck([S0, I0, R0,days])
        self.floatCheck([pss, gamma, kappa, side, rstart, w0, alpha])
        self.negValCheck([S0, I0, R0, pss, gamma, kappa, side, rstart, days, w0, alpha])
        self.probValCheck([pss, gamma, kappa, w0])
        super(StrongInfSIRS, self).__init__(pss=pss, rstart=rstart, alpha=alpha, side=side, S0=S0, I0=I0, R0=R0, days=days, gamma=gamma, kappa=kappa, w0=w0, hubConstant=1)

    def _infect(self, inf: Person, sus: Person):
        # compute the distance between two Person objects
        r = dist(inf, sus)
        # make variable that can potentially be changed if someone is a super spreader
        r0 = self.rstart
        # if the susceptible is too far away from the infectious person
        if r > r0:
            return 0
        # in range of the infected person
        if inf.ss:
            return self.w0
        # return using the normal probability function if not a super spreader
        return self.w0 * (1 - r / r0) ** self.alpha
    
    def plot(self):
        t = np.linspace(0, self.days, self.days + 1)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex='all')
        ax1.plot(t, self.S, label="Susceptible", color='r')
        ax1.set_ylabel("Number of Susceptible People")
        ax1.set_title("Strong Infectious SIR Simulation")
        ax2.plot(t, self.I, label="Active Cases", color='b')
        ax2.set_ylabel("Active Cases")
        ax3.plot(t, self.R, label="Recovered", color='m')
        ax3.set_xlabel("Days")
        ax3.set_ylabel('Number of Recovered')
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.show()