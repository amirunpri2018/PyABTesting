from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np
import backtrader as bt
import backtrader.indicators as btind
import backtrader.analyzers as btanal
import datetime as dt
import pandas as pd
import pandas_datareader as web
from pandas import Series, DataFrame
import random
import pyfolio as pf
from copy import deepcopy
 
from rpy2.robjects.packages import importr
 
pa = importr("PerformanceAnalytics")    # The R package PerformanceAnalytics, containing the R function VaR
 
from rpy2.robjects import numpy2ri, pandas2ri
numpy2ri.activate()
pandas2ri.activate()