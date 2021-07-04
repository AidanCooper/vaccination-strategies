# init file for Eir package
import sys
import os

path = os.path.dirname(__file__)
sys.path.insert(0, path)

# Deterministic
from Eir.Deterministic.AgeStructuredSIR import *