# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 00:03:38 2017

@author: Ivan Liu
"""
#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015,2016 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime

# The above could be sent to an independent module
import backtrader as bt
from backtrader.utils import flushfile  # win32 quick stdout flushing

StoreCls = bt.stores.OandaStore
DataCls = bt.feeds.OandaData
#BrokerCls = bt.brokers.OandaBroker


class TestStrategy(bt.Strategy):
    params = dict(
        smaperiod=5,
        trade=False,
        stake=10,
        exectype=bt.Order.Market,
        stopafter=0,
        valid=None,
        cancel=0,
        donotcounter=False,
        sell=False,
        usebracket=False,
    )

    def __init__(self):
        # To control operation entries
        self.orderid = list()
        self.order = None

        self.counttostop = 0
        self.datastatus = 0

        # Create SMA on 2nd data
        self.sma = bt.indicators.MovAv.SMA(self.data, period=self.p.smaperiod)

        print('--------------------------------------------------')
        print('Strategy Created')
        print('--------------------------------------------------')

    def notify_data(self, data, status, *args, **kwargs):
        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status), *args)
        if status == data.LIVE:
            self.counttostop = self.p.stopafter
            self.datastatus = 1

    def notify_store(self, msg, *args, **kwargs):
        print('*' * 5, 'STORE NOTIF:', msg)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Cancelled, order.Rejected]:
            self.order = None

        print('-' * 50, 'ORDER BEGIN', datetime.datetime.now())
        print(order)
        print('-' * 50, 'ORDER END')

    def notify_trade(self, trade):
        print('-' * 50, 'TRADE BEGIN', datetime.datetime.now())
        print(trade)
        print('-' * 50, 'TRADE END')

    def prenext(self):
        print("pre periods")
        self.next(frompre=True)

    def next(self, frompre=False):
        txt = list()
        txt.append('Data0')
        txt.append('%04d' % len(self.data0))
        dtfmt = '%Y-%m-%dT%H:%M:%S.%f'
        txt.append('{:f}'.format(self.data.datetime[0]))
        txt.append('%s' % self.data.datetime.datetime(0).strftime(dtfmt))
        txt.append('{:f}'.format(self.data.open[0]))
        txt.append('{:f}'.format(self.data.high[0]))
        txt.append('{:f}'.format(self.data.low[0]))
        txt.append('{:f}'.format(self.data.close[0]))
        txt.append('{:6d}'.format(int(self.data.volume[0])))
        txt.append('{:d}'.format(int(self.data.openinterest[0])))
        txt.append('{:f}'.format(self.sma[0]))
        print(', '.join(txt))

        if len(self.datas) > 1 and len(self.data1):
            txt = list()
            txt.append('Data1')
            txt.append('%04d' % len(self.data1))
            dtfmt = '%Y-%m-%dT%H:%M:%S.%f'
            txt.append('{}'.format(self.data1.datetime[0]))
            txt.append('%s' % self.data1.datetime.datetime(0).strftime(dtfmt))
            txt.append('{}'.format(self.data1.open[0]))
            txt.append('{}'.format(self.data1.high[0]))
            txt.append('{}'.format(self.data1.low[0]))
            txt.append('{}'.format(self.data1.close[0]))
            txt.append('{}'.format(self.data1.volume[0]))
            txt.append('{}'.format(self.data1.openinterest[0]))
            txt.append('{}'.format(float('NaN')))
            print(', '.join(txt))

        if self.counttostop:  # stop after x live lines
            self.counttostop -= 1
            if not self.counttostop:
                self.env.runstop()
                return

        if not self.p.trade:
            return

        if self.datastatus and not self.position and len(self.orderid) < 1:
            if not self.p.usebracket:
                if not self.p.sell:
                    # price = round(self.data0.close[0] * 0.90, 2)
                    price = self.data0.close[0] - 0.005
                    self.order = self.buy(size=self.p.stake,
                                          exectype=self.p.exectype,
                                          price=price,
                                          valid=self.p.valid)
                else:
                    # price = round(self.data0.close[0] * 1.10, 4)
                    price = self.data0.close[0] - 0.05
                    self.order = self.sell(size=self.p.stake,
                                           exectype=self.p.exectype,
                                           price=price,
                                           valid=self.p.valid)

            else:
                print('USING BRACKET')
                price = self.data0.close[0] - 0.05
                self.order, _, _ = self.buy_bracket(size=self.p.stake,
                                                    exectype=bt.Order.Market,
                                                    price=price,
                                                    stopprice=price - 0.10,
                                                    limitprice=price + 0.10,
                                                    valid=self.p.valid)

            self.orderid.append(self.order)
        elif self.position and not self.p.donotcounter:
            if self.order is None:
                if not self.p.sell:
                    self.order = self.sell(size=self.p.stake // 2,
                                           exectype=bt.Order.Market,
                                           price=self.data0.close[0])
                else:
                    self.order = self.buy(size=self.p.stake // 2,
                                          exectype=bt.Order.Market,
                                          price=self.data0.close[0])

            self.orderid.append(self.order)

        elif self.order is not None and self.p.cancel:
            if self.datastatus > self.p.cancel:
                self.cancel(self.order)

        if self.datastatus:
            self.datastatus += 1

    def start(self):
        if self.data0.contractdetails is not None:
            print('-- Contract Details:')
            print(self.data0.contractdetails)

        header = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
                  'OpenInterest', 'SMA']
        print(', '.join(header))

        self.done = False



if __name__ == '__main__':
    # Create a cerebro
    cerebro = bt.Cerebro()
    
    storekwargs = dict(
        token="8153764443276ed6230c2d8a95dac609-e9e68019e7c1c51e6f99a755007914f7",
        account="101-011-6029361-001",
        practice="practice"
    )
    
    store = StoreCls(**storekwargs)
    broker = store.getbroker()
    # broker = BrokerCls(**storekwargs)
    cerebro.setbroker(broker)
    
    timeframe = bt.TimeFrame.TFrame(bt.TimeFrame.Names[4])
    compression = 15
    # Manage data1 parameters
    tf1 = timeframe
    cp1 = compression
    resample = False
    replay = False
    if resample or replay:
        datatf = datatf1 = bt.TimeFrame.Ticks
        datacomp = datacomp1 = 1
    else:
        datatf = timeframe
        datacomp = compression
        datatf1 = tf1
        datacomp1 = cp1
    
    fromdate = "2010-01-01"
    if fromdate:
        dtformat = '%Y-%m-%d' + ('T%H:%M:%S' * ('T' in fromdate))
        fromdate = datetime.datetime.strptime(fromdate, dtformat)
    
    no_store = False
    DataFactory = DataCls if no_store else store.getdata
    
    qcheck = 0.5
    historical = False
    bidask = True
    useask = True
    no_backfill_start = False
    no_backfill = False
    timezone = None
    datakwargs = dict(
        timeframe=datatf, compression=datacomp,
        qcheck=qcheck,
        historical=historical,
        fromdate=fromdate,
        bidask=bidask,
        useask=useask,
        backfill_start=not no_backfill_start,
        backfill=not no_backfill,
        tz=timezone
    )
    
    if no_store and not broker:   # neither store nor broker
        datakwargs.update(storekwargs)  # pass the store args over the data
    
    data0 = None
    data0 = DataFactory(dataname=data0, **datakwargs)
    
    data1 = None
    if data1 is not None:
        if data1 != data0:
            datakwargs['timeframe'] = datatf1
            datakwargs['compression'] = datacomp1
            data1 = DataFactory(dataname=data1, **datakwargs)
        else:
            data1 = data0
    
    no_bar2edge = None
    no_adjbartime = None
    no_rightedge = None
    no_takelate = None
    rekwargs = dict(
        timeframe=timeframe, compression=compression,
        bar2edge=not no_bar2edge,
        adjbartime=not no_adjbartime,
        rightedge=not no_rightedge,
        takelate=not no_takelate,
    )
    
    if replay:
        cerebro.replaydata(data0, **rekwargs)
    
        if data1 is not None:
            rekwargs['timeframe'] = tf1
            rekwargs['compression'] = cp1
            cerebro.replaydata(data1, **rekwargs)
    
    elif resample:
        cerebro.resampledata(data0, **rekwargs)
    
        if data1 is not None:
            rekwargs['timeframe'] = tf1
            rekwargs['compression'] = cp1
            cerebro.resampledata(data1, **rekwargs)
    
    else:
        cerebro.adddata(data0)
        if data1 is not None:
            cerebro.adddata(data1)
    
    valid = None
    if valid is None:
        valid = None
    else:
        valid = datetime.timedelta(seconds=valid)
    # Add the strategy
    smaperiod = 20
    trade = None
    exectype = bt.Order.ExecTypes[0]
    stopafter = 0
    donotcounter = None
    cancel = 0
    sell = None
    usebracket = None
    stake = None
    cerebro.addstrategy(TestStrategy,
                        smaperiod=smaperiod,
                        trade=trade,
                        exectype=bt.Order.ExecType(exectype),
                        stake=stake,
                        stopafter=stopafter,
                        valid=valid,
                        cancel=cancel,
                        donotcounter=donotcounter,
                        sell=sell,
                        usebracket=usebracket)
    
    # Live data ... avoid long data accumulation by switching to "exactbars"
    exactbars = 1
    cerebro.run(exactbars=exactbars)
    plot = None
    if exactbars < 1:  # plotting is possible
        if plot:
            pkwargs = dict(style='line')
            if plot is not True:  # evals to True but is not True
                npkwargs = eval('dict(' + plot + ')')  # args were passed
                pkwargs.update(npkwargs)
    
            cerebro.plot(**pkwargs)