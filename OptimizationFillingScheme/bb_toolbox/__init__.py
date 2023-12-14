'''
A simple package to convert NX CERN Logging information into a pandas dataframe.
'''
#%%
__version__ = "v0.2"
import os
print(os.getcwd())
from .synthesis import *
from .analysis import *

# %%
