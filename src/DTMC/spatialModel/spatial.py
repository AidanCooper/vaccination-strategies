from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.utility import Person
import src.utility as u
import multiprocessing as mp


class Spatial:
    """
        Abstract Class used to for static spatial models. 

        Attributes
        ----------

        popsize : int
            the number of people in the population 
        
        pss: float
            0 <= pss <= 1. Represents the probability that a person is a super spreader.
        
        rstart: float
            the spreading radius of a normal person
        
        alpha: int
            constant in the exponent of the w(r) formula, which is used to calculate the probability of 
            an infectious person infecting a susceptible person.
        
        side: float
            the length of the side of the continuous plane in which the people are in. For example, if 
            side=50, then the plane is a square with corners at (0,0), (50,0), (50,50), (0,50).

        S0: int
            the starting number of susceptible people in the simulation.

        I0: int
            the starting number of infectious people in the simulation. 
        
        days: int
            the number of days the simulation lasts for. 
        
        w0: float, optional
            the starting probability if the distance between an infectious person and susceptible person is 0.
            The probability is initialized to 1.0 if no value is passed. 
        
        locx: ndarray
            a numpy array with length of popsize that contains randomly generated points for the x-coordinates
            of every person.
        
        locy: ndarray
            a numpy array with length of popsize that contains randomly generated points for the y-coordinates
            of every person.
    """
    def __init__(self, popsize: int, pss: float, rstart: float, alpha: int, side: float, S0: int, I0: int, days: int, w0=1.0):
        
        # the total size of the closed population
        self.popsize = popsize
        # the probability that a person is a super spreader
        self.pss = pss
        # the starting spreading radius of an infectious perosn
        self.rstart = rstart
        # the normal probability constant that is used in Fujie & Odagaki paper infect function
        self.alpha = alpha
        # one side of the plane of the simulation
        self.side = side
        # initial susceptibles, initial infecteds
        self.S0, self.I0 = S0, I0
        # make the data structure that makes it easy to look up different things rather than using sets
        self.Scollect, self.Icollect = [], []
        # initialize the starting probability when 0 units from infectious person
        self.w0 = w0
        # number of days
        self.days = days
        # create the arrays storing the number of people in each state on each day
        self.S, self.I = np.zeros(days+1), np.zeros(days+1)
        # initialize for day 0
        self.S[0], self.I[0] = S0, I0
        # create an array of correspond x and y coordinates for popsize number of people
        # the infected person will be at locx[0] and locy[0]
        self.locx = np.random.rand(popsize) * side
        self.locy = np.random.rand(popsize) * side



    # will be inherited from Hub and Strong Infectious classes and implemented there
    def _infect(self, inf: Person, sus: Person):
        """
        Method that determines how the probability of infection is calculated. Formulas are different
        for Hub model and Strong Infectious model. 

        Parameters
        ----------
        inf: Person
            represents the infectious person. His spreading radius(normal or super spreader) will be used
            when calculating the probability of infection.
        
        sus: Person
            represents the susceptible person.

        """
        pass

    def _statechange(self):
        pass