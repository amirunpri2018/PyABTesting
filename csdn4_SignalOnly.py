# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 23:56:24 2017

@author: sky_x
"""
  
from __future__ import (absolute_import, division, print_function,  
                        unicode_literals)  
  
import datetime  # For datetime objects  
import pandas as pd  
import backtrader as bt  
  
class MySignal(bt.Indicator):  
    lines = ('signal',)  
    params = (('period', 30),)  
  
    def __init__(self):  
        self.lines.signal = self.data - bt.indicators.SMA(period=self.p.period)  
  
if __name__ == '__main__':  
    cerebro = bt.Cerebro()  
    dataframe = pd.read_csv('./datas/yhoo-1996-2015.txt', index_col=0, parse_dates=True)  
    dataframe['openinterest'] = 0  
    data = bt.feeds.PandasData(dataname=dataframe,  
                            fromdate = datetime.datetime(2013, 1, 1),  
                            todate = datetime.datetime(2015, 12, 31)  
                            )  
    # Add the Data Feed to Cerebro  
    cerebro.adddata(data)  
  
    cerebro.add_signal(bt.SIGNAL_LONGSHORT, MySignal, subplot=False)  
    # 这句话很有用，画图看效果  
    # cerebro.signal_accumulate(True)  
    cerebro.broker.setcash(10000.0)  
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)  
    cerebro.broker.setcommission(commission=0.0)  
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())  
    cerebro.run()  
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())  
    cerebro.plot() 