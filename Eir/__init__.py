# init file for Eir package
import sys
import os

path = os.path.dirname(__file__)
sys.path.insert(0, path)

# Deterministic
from Eir.Deterministic.AgeStructuredSIR import *
from Eir.Deterministic.AgeStructuredSIRD import *
from Eir.Deterministic.SEIR import *
from Eir.Deterministic.SIR import *
from Eir.Deterministic.SIRD import *
from Eir.Deterministic.SIRS import *
from Eir.Deterministic.SIRV import *
from Eir.Deterministic.SIS import *