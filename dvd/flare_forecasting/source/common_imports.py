################################################################################
# Filename: common_imports.py
# Description: This file contains all imports common to each experiment.
################################################################################

# Python Imports
import datetime as dt
import os
import random
import string
import sys
import warnings

from datetime import datetime
from datetime import timedelta
from typing import List, Tuple, Dict, Any

# Third-Party Imports
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from pandas.core.common import SettingWithCopyWarning
from pandas.plotting import parallel_coordinates


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import chi2, f_classif
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from sklearn.model_selection import train_test_split, LeaveOneOut, LeavePOut, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle





# Custom Imports
from source.constants import *
