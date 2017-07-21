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

## import rpy2's package module
#import rpy2.robjects.packages as rpackages
## import R's utility package
#utils = rpackages.importr('utils')
## select a mirror for R packages
#utils.chooseCRANmirror(ind=1) # select the first mirror in the list
## R package names
#packnames = ('PerformanceAnalytics')
## R vector of strings
#from rpy2.robjects.vectors import StrVector
## Selectively install what needs to be install.
#utils.install_packages(packnames)

from rpy2.robjects.packages import importr
pa = importr("PerformanceAnalytics")    # The R package PerformanceAnalytics, containing the R function VaR
 
from rpy2.robjects import numpy2ri, pandas2ri
numpy2ri.activate()
pandas2ri.activate()

class SMAC(bt.Strategy):
    """A simple moving average crossover strategy; crossing of a fast and slow moving average generates buy/sell
       signals"""
    params = {"fast": 20, "slow": 50,                  # The windows for both fast and slow moving averages
              "optim": False, "optim_fs": (20, 50)}    # Used for optimization; equivalent of fast and slow, but a tuple
                                                       # The first number in the tuple is the fast MA's window, the
                                                       # second the slow MA's window
 
    def __init__(self):
        """Initialize the strategy"""
 
        self.fastma = dict()
        self.slowma = dict()
        self.cross  = dict()
 
        if self.params.optim:    # Use a tuple during optimization
            self.params.fast, self.params.slow = self.params.optim_fs    # fast and slow replaced by tuple's contents
 
        if self.params.fast > self.params.slow:
            raise ValueError(
                "A SMAC strategy cannot have the fast moving average's window be " + \
                 "greater than the slow moving average window.")
 
        for d in self.getdatanames():
 
            # The moving averages
            self.fastma[d] = btind.SimpleMovingAverage(self.getdatabyname(d),      # The symbol for the moving average
                                                       period=self.params.fast,    # Fast moving average
                                                       plotname="FastMA: " + d)
            self.slowma[d] = btind.SimpleMovingAverage(self.getdatabyname(d),      # The symbol for the moving average
                                                       period=self.params.slow,    # Slow moving average
                                                       plotname="SlowMA: " + d)
 
            # This is different; this is 1 when fast crosses above slow, -1 when fast crosses below slow, 0 o.w.
            self.cross[d] = btind.CrossOver(self.fastma[d], self.slowma[d], plot=False)
 
    def next(self):
        """Define what will be done in a single step, including creating and closing trades"""
        for d in self.getdatanames():    # Looping through all symbols
            pos = self.getpositionbyname(d).size or 0
            if pos == 0:    # Are we out of the market?
                # Consider the possibility of entrance
                if self.cross[d][0] > 0:    # A buy signal
                    self.buy(data=self.getdatabyname(d))
 
            else:    # We have an open position
                if self.cross[d][0] < 0:    # A sell signal
                    self.sell(data=self.getdatabyname(d))
 
 
class PropSizer(bt.Sizer):
    """A position sizer that will buy as many stocks as necessary for a certain proportion of the portfolio
       to be committed to the position, while allowing stocks to be bought in batches (say, 100)"""
    params = {"prop": 0.1, "batch": 100}
 
    def _getsizing(self, comminfo, cash, data, isbuy):
        """Returns the proper sizing"""
 
        if isbuy:    # Buying
            target = self.broker.getvalue() * self.params.prop    # Ideal total value of the position
            price = data.close[0]
            shares_ideal = target / price    # How many shares are needed to get target
            batches = int(shares_ideal / self.params.batch)    # How many batches is this trade?
            shares = batches * self.params.batch    # The actual number of shares bought
 
            if shares * price > cash:
                return 0    # Not enough money for this trade
            else:
                return shares
 
        else:    # Selling
            return self.broker.getposition(data).size    # Clear the position
 
 
class AcctValue(bt.Observer):
    alias = ('Value',)
    lines = ('value',)
 
    plotinfo = {"plot": True, "subplot": True}
 
    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()    # Get today's account value (cash + stocks)
 
 
class AcctStats(bt.Analyzer):
    """A simple analyzer that gets the gain in the value of the account; should be self-explanatory"""
 
    def __init__(self):
        self.start_val = self.strategy.broker.get_value()
        self.end_val = None
 
    def stop(self):
        self.end_val = self.strategy.broker.get_value()
 
    def get_analysis(self):
        return {"start": self.start_val, "end": self.end_val,
                "growth": self.end_val - self.start_val, "return": self.end_val / self.start_val}
 
 
class TimeSeriesSplitImproved(TimeSeriesSplit):
    """Time Series cross-validator
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.
    Examples
    --------
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> tscv = TimeSeriesSplit(n_splits=3)
    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(n_splits=3)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [1] TEST: [2]
    TRAIN: [2] TEST: [3]
    >>> for train_index, test_index in tscv.split(X, fixed_length=True,
    ...     train_splits=2):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [1 2] TEST: [3]
 
    Notes
    -----
    When ``fixed_length`` is ``False``, the training set has size
    ``i * train_splits * n_samples // (n_splits + 1) + n_samples %
    (n_splits + 1)`` in the ``i``th split, with a test set of size
    ``n_samples//(n_splits + 1) * test_splits``, where ``n_samples``
    is the number of samples. If fixed_length is True, replace ``i``
    in the above formulation with 1, and ignore ``n_samples %
    (n_splits + 1)`` except for the first training set. The number
    of test sets is ``n_splits + 2 - train_splits - test_splits``.
    """
 
    def split(self, X, y=None, groups=None, fixed_length=False,
              train_splits=1, test_splits=1):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.
        fixed_length : bool, hether training sets should always have
            common length
        train_splits : positive int, for the minimum number of
            splits to include in training sets
        test_splits : positive int, for the number of splits to
            include in the test set
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        train_splits, test_splits = int(train_splits), int(test_splits)
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (n_folds - train_splits - test_splits) < 0:
            raise ValueError(
                ("Cannot have more training and testing splits than "
                 "there are splits in total."))
        if not (train_splits > 0 and test_splits > 0):
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))
        indices = np.arange(n_samples)
        split_size = (n_samples // n_folds)
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        if fixed_length:
            for i, test_start in zip(range(len(test_starts)),
                                     test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (indices[(test_start - train_size - rem):test_start],
                       indices[test_start:test_start + test_size])
        else:
            for test_start in test_starts:
                yield (indices[:test_start],
                    indices[test_start:test_start + test_size])
                

                
start = dt.datetime(2010, 1, 1)
end = dt.datetime(2016, 10, 31)
symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "YHOO", "SNY", "VZ", "IBM", "HPQ", "QCOM", "NVDA"]
datafeeds = {s: web.DataReader(s, "google", start, end) for s in symbols}
for df in datafeeds.values():
    df["OpenInterest"] = 0    # PandasData reader expects an OpenInterest column;
                              # not provided by Google and we don't use it so set to 0
 
cerebro = bt.Cerebro(stdstats=False)
 
plot_symbols = ["AAPL", "GOOG", "NVDA"]
is_first = True
#plot_symbols = []
for s, df in datafeeds.items():
    data = bt.feeds.PandasData(dataname=df, name=s)
    if s in plot_symbols:
        if is_first:
            data_main_plot = data
            is_first = False
        else:
            data.plotinfo.plotmaster = data_main_plot
    else:
        data.plotinfo.plot = False
    cerebro.adddata(data)
 
cerebro.broker.setcash(1000000)
cerebro.broker.setcommission(0.02)
cerebro.addstrategy(SMAC)
cerebro.addobserver(AcctValue)
cerebro.addobservermulti(bt.observers.BuySell)
cerebro.addsizer(PropSizer)



# Using R’s PerformanceAnalytics Package in Python
class VaR(bt.Analyzer):
    """
    Computes the value at risk metric for the whole account using the strategy, based on the R package
    PerformanceAnalytics VaR function
    """
    params = {
        "p": 0.95,                # Confidence level of calculation
        "method": "historical"    # Method used; can be "historical", "gaussian", "modified", "kernel"
    }
 
    def __init__(self):
        self.acct_return = dict()
        self.acct_last = self.strategy.broker.get_value()
        self.vardict = dict()
 
    def next(self):
        if len(self.data) > 1:
            curdate = self.strategy.datetime.date(0)
            # I use log returns
            self.acct_return[curdate] = np.log(self.strategy.broker.get_value()) - np.log(self.acct_last)
            self.acct_last = self.strategy.broker.get_value()
 
    def stop(self):
        srs = Series(self.acct_return)    # Need to pass a time-series-like object to VaR
        srs.sort_index(inplace=True)
        self.vardict["VaR"] = pa.VaR(srs, p=self.params.p, method=self.params.method)[0]    # Get VaR
        del self.acct_return              # This dict is of no use to us anymore
 
    def get_analysis(self):
        return self.vardict
 
 
class SortinoRatio(bt.Analyzer):
    """
    Computes the Sortino ratio metric for the whole account using the strategy, based on the R package
    PerformanceAnalytics SortinoRatio function
    """
    params = {"MAR": 0}    # Minimum Acceptable Return (perhaps the risk-free rate?); must be in same periodicity
                           # as data
 
    def __init__(self):
        self.acct_return = dict()
        self.acct_last = self.strategy.broker.get_value()
        self.sortinodict = dict()
 
    def next(self):
        if len(self.data) > 1:
            # I use log returns
            curdate = self.strategy.datetime.date(0)
            self.acct_return[curdate] = np.log(self.strategy.broker.get_value()) - np.log(self.acct_last)
            self.acct_last = self.strategy.broker.get_value()
 
    def stop(self):
        srs = Series(self.acct_return)    # Need to pass a time-series-like object to SortinoRatio
        srs.sort_index(inplace=True)
        self.sortinodict['sortinoratio'] = pa.SortinoRatio(srs, MAR = self.params.MAR)[0]    # Get Sortino Ratio
        del self.acct_return              # This dict is of no use to us anymore
 
    def get_analysis(self):
        return self.sortinodict

cerebro.addanalyzer(btanal.PyFolio)                # Needed to use PyFolio
cerebro.addanalyzer(btanal.TradeAnalyzer)          # Analyzes individual trades
cerebro.addanalyzer(btanal.SharpeRatio_A)          # Gets the annualized Sharpe ratio
#cerebro.addanalyzer(btanal.AnnualReturn)          # Annualized returns (does not work?)
cerebro.addanalyzer(btanal.Returns)                # Returns
cerebro.addanalyzer(btanal.DrawDown)               # Drawdown statistics
cerebro.addanalyzer(VaR)                           # Value at risk
cerebro.addanalyzer(SortinoRatio, MAR=0.00004)     # Sortino ratio with risk-free rate of 0.004% daily (~1% annually)

res = cerebro.run()
res[0].analyzers.tradeanalyzer.get_analysis()
res[0].analyzers.sharperatio_a.get_analysis()
res[0].analyzers.returns.get_analysis()
res[0].analyzers.drawdown.get_analysis()
res[0].analyzers.var.get_analysis()
res[0].analyzers.sortinoratio.get_analysis()



# PyFolio and backtrader
returns, positions, transactions, gross_lev = res[0].analyzers.pyfolio.get_pf_items()
pf.create_round_trip_tear_sheet(returns, positions, transactions)

len(returns.index)
benchmark_rets = pd.Series([0.00004] * len(returns.index), index=returns.index)    # Risk-free rate, we need to
                                                                                   # set this ourselves otherwise
                                                                                   # PyFolio will try to fetch from
                                                                                   # Yahoo! Finance which is now garbage
 
# NOTE: Thanks to Yahoo! Finance giving the finger to their users (thanks Verizon, we love you too),
# PyFolio is unstable and will be until updates are made to pandas-datareader, so there may be issues
# using it
pf.create_full_tear_sheet(returns, positions, transactions, benchmark_rets=benchmark_rets)




# Optimizing with Different Metrics
def wfa(cerebro, strategy, opt_param, split, datafeeds, analyzer_max, var_maximize,
        opt_p_vals, opt_p_vals_args={}, params={}, minimize=False):
    """Perform a walk-forward analysis
 
    args:
        cerebro: A Cerebro object, with everything added except the strategy and data feeds; that is, all
            analyzers, sizers, starting balance, etc. needed have been added
        strategy: A Strategy object for optimizing
        params: A dict that contains all parameters to be set for the Strategy except the parameter to be
            optimized
        split: Defines the splitting of the data into training and test sets, perhaps created by
            TimeSeriesSplit.split()
        datafeeds: A dict containing pandas DataFrames that are the data feeds to be added to Cerebro
            objects and used by the strategy; keys are the names of the strategies
        analyzer_max: A string denoting the name of the analyzer with the statistic to maximize
        var_maximize: A string denoting the variable to maximize (perhaps the key of a dict)
        opt_param: A string: the parameter of the strategy being optimized
        opt_p_vals: This can be either a list or function; if a function, it must return a list, and the
            list will contain possible values that opt_param will take in optimization; if a list, it already
            contains these values
        opt_p_vals_args: A dict of parameters to pass to opt_p_vals; ignored if opt_p_vals is not a function
        minimize: A boolean that if True will minimize the parameter rather than maximize
 
    return:
        A list of dicts that contain data for the walk-forward analysis, including the relevant
        analyzer's data, the value of the optimized statistic on the training set, start and end dates for
        the test set, and the walk forward efficiency ratio (WFER)
    """
 
    usr_opt_p_vals = opt_p_vals
    walk_forward_results = list()
    for train, test in split:
        trainer, tester = deepcopy(cerebro), deepcopy(cerebro)
 
        if callable(usr_opt_p_vals):
            opt_p_vals = usr_opt_p_vals(**opt_p_vals_args)
 
        # TRAINING
        trainer.optstrategy(strategy, **params, **{opt_param: opt_p_vals})
        for s, df in datafeeds.items():
            data = bt.feeds.PandasData(dataname=df.iloc[train], name=s)
            trainer.adddata(data)
        res = trainer.run()
        res_df = DataFrame({getattr(r[0].params, opt_param): dict(getattr(r[0].analyzers,
                                                                          analyzer_max).get_analysis()) for r in res}
                       ).T.loc[:, var_maximize].sort_values(ascending=minimize)
        opt_res, opt_val = res_df.index[0], res_df[0]
 
        # TESTING
        tester.addstrategy(strategy, **params, **{opt_param: opt_res})
        for s, df in datafeeds.items():
            data = bt.feeds.PandasData(dataname=df.iloc[test], name=s)
            tester.adddata(data)
        res = tester.run()
        res_dict = dict(getattr(res[0].analyzers, analyzer_max).get_analysis())
        res_dict["train_" + var_maximize] = opt_val
        res_dict[opt_param] = opt_res
        s0 = [*datafeeds.keys()][0]
        res_dict["start_date"] = datafeeds[s0].iloc[test[0]].name
        res_dict["end_date"] = datafeeds[s0].iloc[test[-1]].name
        test_val = getattr(res[0].analyzers, analyzer_max).get_analysis()[var_maximize]
        try:
            res_dict["WFER"] = test_val / opt_val
        except:
            res_dict["WFER"] = np.nan
        for anlz in res[0].analyzers:
            res_dict.update(dict(anlz.get_analysis()))
 
        walk_forward_results.append(res_dict)
 
    return walk_forward_results


def random_fs_list():
    """Generate random combinations of fast and slow window lengths to test"""
    windowset = set()    # Use a set to avoid duplicates
    while len(windowset) < 40:
        f = random.randint(1, 10) * 5
        s = random.randint(1, 10) * 10
        if f > s:    # Cannot have the fast moving average have a longer window than the slow, so swap
            f, s = s, f
        elif f == s:    # Cannot be equal, so do nothing, discarding results
            continue
        windowset.add((f, s))
 
    return list(windowset)

walkorebro = bt.Cerebro(stdstats=False, maxcpus=1)
 
walkorebro.broker.setcash(1000000)
walkorebro.broker.setcommission(0.02)
walkorebro.addanalyzer(btanal.SharpeRatio_A)
walkorebro.addanalyzer(btanal.Returns)
walkorebro.addanalyzer(AcctStats)
walkorebro.addsizer(PropSizer)
 
tscv = TimeSeriesSplitImproved(10)
split = tscv.split(datafeeds["AAPL"], fixed_length=True, train_splits=2)

wfa_sharpe_res = wfa(walkorebro, SMAC, opt_param="optim_fs", params={"optim": True}, 
                     split=split, datafeeds=datafeeds, analyzer_max="sharperatio_a",
                     var_maximize="sharperatio", opt_p_vals=random_fs_list)
DataFrame(wfa_sharpe_res)

walkorebro.addanalyzer(SortinoRatio)
split = tscv.split(datafeeds["AAPL"], fixed_length=True, train_splits=2)
wfa_sortino_res = wfa(walkorebro, SMAC, opt_param="optim_fs", params={"optim": True},
                      split=split, datafeeds=datafeeds, analyzer_max="sortinoratio",
                      var_maximize="sortinoratio", opt_p_vals=random_fs_list)
DataFrame(wfa_sortino_res)