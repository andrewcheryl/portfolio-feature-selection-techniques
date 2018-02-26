#standard tools
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from IPython.display import display
from collections import defaultdict
import random
from tqdm import tqdm
from time import sleep
#models
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.linear_model import RidgeClassifier , Ridge , RidgeCV , Lasso , LassoCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
#processing and selection
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE,RFECV, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
#postgresdb access
import psycopg2 as pg2
from psycopg2.extras import RealDictCursor
#local libraries
from lib.datasets import Madelon_data
from lib.datasets import Postgres_data
from lib.datasets import Pickle_files
