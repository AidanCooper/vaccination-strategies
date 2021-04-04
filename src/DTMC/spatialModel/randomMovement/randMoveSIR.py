import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

import src.utility as u
from src.DTMC.spatialModel.simul_details import Simul_Details
# not to be confused with the person object that is used in the Hub/Strong Infectious Model
from src.utility import Person1 as Person
from src.DTMC.spatialModel.randomMovement.randMoveSIS import RandMoveSIS

class RandMoveSIR(RandMoveSIS):
    """
    An SIR model that follows the Random Movement Model. When the individuals in the simulation move, 
    they move according to a randomly generated angle and a randomly generated radius.

    Parameters:
    ----------

    S0: int
        The starting number of susceptible individuals in the simulation.
    
    I0: int
        The starting number of infectious individuals in the simulation. 
    
    R0: int
        The starting number of recovered individuals in the simulation.

    gamma: float
        The recovery probability of an individual going from I -> R.
    
    planeSize : float
        The length of each side of the square plane in which the individuals are confined to. For example,
        if planeSize=50, then the region which people in the simulation are confined to is the square with
        vertices (0,0), (50,0), (50,50), and (0,50).
    
    move_r: float
        The mean of the movement radius of each person in the simulation. Will be used as mean along with 
        sigma_R as the standard deviation to pull from a normal distribution movement radii each time 
        _move(day) function is called.
    
    sigma_R: float
        The standard deviation of the movement radius of each person in the simulation. Will be used along with 
        move_R as the mean to pull from a normal distribution movement radii each time _move(day) function is 
        called.

    spread_r: float
        The mean of the spreading radius of each person in the simulation. Will be used along with sigma_r 
        as the standard deviation to pull from an normal distribution spreading radii for each individaul person
        when the RandMoveSIS object is initialized. 
    
    sigma_r: float
        The standard deviation of the spreading radius of each person in the simulation. 
        Will be used along with spread_r as the mean to pull from an normal distribution spreading radii 
        for each individaul person when the RandMoveSIS object is initialized. 
    
    days: int
        The number of days that was simulated.
    
    w0: float optional
        The probability of infection if the distance between an infectious person and susceptible person is 0. Default is 1.0.
    
    alpha: float optional
        A constant used in the _infect() method. The greater the constant, the greater the infection probability. Default is 2.0.

    Attributes
    ----------

    S: ndarray
        A numpy array that stores the number of people in the susceptible state on each given day of the simulation.
    
    I: ndarray
        A numpy array that stores the number of people in the infected state on each given day of the simulation.
    
    R: ndarray
        A numpy array that stores the number of people in the recovered state on each given day of the simulation.
    
    popsize: int
        The total size of the population in the simulation. Given by S0 + I0 + R0.
        
    Scollect: list
        Used to keep track of the states each Person object is in. If the copy of a Person object has 
        isIncluded == True, then the person is SUSCEPTIBLE. Has a total of popsize Person objects,
        with numbers [0, popsize). 
    
    Icollect: list
         Used to keep track of the states each Person object is in. If the copy of a Person object has 
        isIncluded == True, then the person is INFECTED. Has a total of popsize Person objects,
        with numbers [0, popsize).
    
    Rcollect: list
        Used to keep track of the states each Person object is in. If the copy of a Person object has 
        isIncluded == True, then the person is RECOVERED. Has a total of popsize Person objects,
        with numbers [0, popsize).


    details: Simul_Details 
        An object that can be returned to give a more in-depth look into the simulation. With this object,
        one can see transmission chains, state changes, the movement history of each individaul, the state
        history of each person, and more.
    
     """

    def __init__(self, S0:int, I0:int, R0:int, gamma:float, planeSize:float, move_r:float, sigma_R:float, spread_r:float, sigma_r: float,
    days:int, w0=1.0, alpha=2.0):
        self.intCheck([S0, I0, R0, days])
        self.floatCheck(gamma, planeSize, move_r, sigma_R, spread_r, sigma_r, w0, alpha)
        self.negValCheck(S0, I0, R0, gamma, planeSize, move_r, sigma_R, spread_r, sigma_r, days, w0, alpha)
        self.probValCheck([gamma, w0])
        super(RandMoveSIR, self).__init__(S0=S0, I0=I0, gamma=gamma, planeSize=planeSize, move_r=move_r, sigma_R=sigma_R, spread_r=spread_r, sigma_r=sigma_r,
        days=days, w0=w0, alpha=alpha)
        self.R0 = R0
        self.R = np.zeros(days+1)
        self.R[0] = R0
        self.popsize += self.R0
        # redirect details object to point to another object to delete copies in inheritance
        self.details = Simul_Details(days, self.popsize)
        self.Scollect, self.Icollect, self.Rcollect = [], [], []
        spreading_r = np.random.normal(spread_r, sigma_r, S0+I0+R0)
        # generate the random x, y locations with every position within the plane being equally likely
        loc_x = np.random.random(S0+I0+R0) * planeSize
        loc_y = np.random.random(S0+I0+R0) * planeSize
        # create the special objects:
        for i in range(self.popsize):
            # create the person object
            # for this model, the people will move with random radius R each timestep
            # therefore, the R component can be made 0, as that is only relevant for the 
            # periodic mobility model
            p1 = Person(loc_x[i], loc_y[i], 0, spreading_r[i])
            p2 = Person(loc_x[i], loc_y[i], 0, spreading_r[i]) 
            p3 = Person(loc_x[i], loc_y[i], 0, spreading_r[i])
            self.details.addLocation(0, (loc_x[i], loc_y[i]))       
            # if the person is in the susceptible objects created
            if i < S0:
                p1.isIncluded = True
                self.details.addStateChange(i, "S", 0)
            elif S0 <= i < S0+I0:
                p2.isIncluded = True
                self.details.addStateChange(i, "I", 0)
            else:
                p3.isIncluded=True
                self.details.addStateChange(i, "R", 0)
            # append them to the data structure
            self.Scollect.append(p1)
            self.Icollect.append(p2)
            self.Rcollect.append(p3)
            self.details.addLocation(0, (p1.x, p1.y))
    

    def _ItoR(self):
        """
        Takes care of running state changes from I compartment to S compartment 
        
        Return
        ------

        set:
            contains the indices of people who should get transferred from I to R compartment.
        """
        # set that contains the indices for transfering from I to S
        return self._changeHelp(self.Icollect, self.gamma)

    def run(self, getDetails=True):
        """
        Run the actual simulation. 

        Parameters
        ----------

        getDetails: bool optional
            If getDetails=True, then run will return a Simul_Details object which will allow the user to 
            examine details of the simulation that aren't immediately obvious.
        
        Returns
        -------

        Simul_Details:
            Allows the user to take a deeper look into the dynamics of the simulation by examining transmission
            chains. User can also examine transmission history and state changes of individuals in the object
            by utilizing the Simul_Details object. 
        """

        # for all the days in the simulation
        for i in range(1, self.days+1):
            print("Day ", i)
            #print("Location: (", self.Scollect[0].x, ",", self.Scollect[0].y, ").")
            # run the state changes
            StoI = self._StoI(i)
            ItoR = self._ItoR()
            # change the indices of the transfers
            self._stateChanger(StoI, self.Icollect, "I", i)
            self._stateChanger(ItoR, self.Rcollect, "R", i)
            
            # make everyone move randomly
            self._move(i, [self.Scollect, self.Icollect, self.Rcollect])
            # change the values in the arrays
            self.S[i] = self.S[i-1] - len(StoI)
            self.I[i] = self.I[i-1] + len(StoI) - len(ItoR)
            self.R[i] = self.R[i-1] + len(ItoR)
        if getDetails:
            return self.details
    
    def toDataFrame(self):
        """
        Gives user access to pandas dataframe with amount of people in each state on each day.

        Returns
        -------

        pd.DataFrame
            DataFrame object containing the number of susceptibles and number of infecteds on each day. 

        """
        # create the linspaced numpy array
        t = np.linspace(0, self.days, self.days + 1)
        # create a 2D array with the days and susceptible and infected arrays
        # do it over axis one so that it creates columns days, susceptible, infected
        arr = np.stack([t, self.S, self.I, self.R], axis=1)
        df = pd.DataFrame(arr, columns=["Days", "Susceptible", "Infected", "Removed"])
        return df
    
    # maybe add picking what to plot later
    def plot(self):
        
        "Plots the number of susceptible and infected individuals on the y-axis and the number of days on the x-axis."

        t = np.linspace(0, self.days, self.days + 1)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex='all')
        ax1.plot(t, self.S, label="Susceptible", color='r')
        ax1.set_ylabel("# Susceptibles")
        ax1.set_title("Random Movement SIR Simulation")
        ax2.plot(t, self.I, label="Active Cases", color='b')
        ax2.set_ylabel("# Active Infections")
        ax3.set_xlabel("Days")
        ax3.set_ylabel("# Recovered")
        ax3.plot(t, self.R, label="Removed")
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.show()