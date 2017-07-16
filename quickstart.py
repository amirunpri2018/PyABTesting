# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 06:06:37 2017

@author: Ivan Liu
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt








if __name__ == '__main__':
    
    # Instantiate the Cerebro engine
    cerebro = bt.Cerebro()

    # Setting the Cash
    cerebro.broker.setcash(100000.0)
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    
    
    
    
    
    
    
