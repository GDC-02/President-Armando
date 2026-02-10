#imports for data analysis and preparation
import pandas as pd
import matplotlib.pyplot as plt

#imports for data analysis and preparation and modeling
import numpy as np

#imports for modeling
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import random


#Import the dataset as a df
df=pd.read_csv(r"dataset_path",delimiter=';')
df
