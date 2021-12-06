from QuantConnect.Data.UniverseSelection import *
import math
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
from sklearn.decomposition import PCA
class MuscularBrownAlbatross(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2021, 1, 1)  # Set Start Date
        self.SetEndDate(2021, 3, 1)  #Set Start Date      
        self.SetCash(100000)  # Set Strategy Cash
        
        self.next_rebalance = self.Time      # Initialize next rebalance time
        self.rebalance_days = 30            # Rebalance every 30 days
        self.lookback = 365                  # Days of historical data
        self.num_components = 3             # Number of principal components in PCA
        self.num_equities = 10              # Number of the equities pool
        self.weights = pd.DataFrame()       # Pandas data frame (index: symbol) that stores the weight
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol 
        self.holding_months = 1
        self.num_screener = 100
        self.num_stocks = 30
        self.formation_days = 200
        self.lowmom = False
        self.month_count = self.holding_months
        self.Schedule.On(self.DateRules.MonthStart("SPY"), self.TimeRules.At(0, 0), Action(self.monthly_rebalance))
        self.Schedule.On(self.DateRules.MonthStart("SPY"), self.TimeRules.At(10, 0), Action(self.rebalance))
        # rebalance the universe selection once a month
        self.rebalence_flag = 0
        
        self.first_month_trade_flag = 1
        self.trade_flag = 0 
        self.symbols = None
    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        pass
    def CoarseSelectionFunction(self, coarse):
        if self.rebalence_flag or self.first_month_trade_flag:
            # drop stocks which have no fundamental data or have too low prices
            selected = [x for x in coarse if (x.HasFundamentalData) and (float(x.Price) > 5)]
            # rank the stocks by dollar volume 
            filtered = sorted(selected, key=lambda x: x.DollarVolume, reverse=True) 
            # Take the top 200
            return [ x.Symbol for x in filtered[:200]]
        else:
            return self.symbols
    def FineSelectionFunction(self, fine):
        if self.rebalence_flag or self.first_month_trade_flag:
            hist = self.History([i.Symbol for i in fine], 1, Resolution.Daily)
            try:
                # Filters by EVToEBITDA earnings
                filtered_fine = [x for x in fine if (x.ValuationRatios.EVToEBITDA > 0) 
                                                    and (x.EarningReports.BasicAverageShares.ThreeMonths > 0) 
                                                    and float(x.EarningReports.BasicAverageShares.ThreeMonths) * hist.loc[str(x.Symbol)]['close'][0] > 2e9]
            except:
                filtered_fine = [x for x in fine if (x.ValuationRatios.EVToEBITDA > 0) 
                                                and (x.EarningReports.BasicAverageShares.ThreeMonths > 0)] 
            # Returns the top 100
            top = sorted(filtered_fine, key = lambda x: x.ValuationRatios.EVToEBITDA, reverse=True)[:self.num_screener]
            self.symbols = [x.Symbol for x in top]
            
            self.rebalence_flag = 0
            self.first_month_trade_flag = 0
            self.trade_flag = 1
            return self.symbols
        else:
            return self.symbols
            
    def PCA(self, fine):
        # grab historical data of our symbols from fine selection
        history = self.History(symbols, self.lookback, Resolution.daily).close.unstack(level = 0)
        
        # select symbols and their weights for the portfolio from the fine selection symbols
        self.weights = self.GetWeights(history)
        # if there is no final selected symbols, return the unchanged universe
        if self.weights.empty:
            return Universe.Unchanged
        return [x for x in symbols if str(x) in self.weights.index]
        
    def GetWeights(self, history):
        
        # find the weights of our final selected symbols
        
        # sample data for each PCA
        sample = np.log(history.dropna(axis=1))
        sample -= sample.mean() # Center it column-wise
        # fit the PCA model for sample data
        model = PCA().fit(sample)
        # get the first num_components factors
        factors = np.dot(sample, model.components_.T)[:,:self.num_components]
        # add 1's to fit the linear regression
        factors = sm.add_constant(factors)
        # train ordinary least squares linear model for each stock
        OLSmodels = {ticker: sm.OLS(sample[ticker], factors).fit() for ticker in sample.columns}
        # get the residuals from the linear regression after PCA for each stock
        residuals = pd.DataFrame({ticker: model.resid for ticker, model in OLSmodels.items()})
        # get the Z scores by standarize the given pandas dataframe X
        z_scores = ((residuals - residuals.mean()) / residuals.std()).iloc[-1] # residuals of the most recent day
        # get the stocks far from mean (for mean reversion)
        selected = z_scores[z_scores < -1.5]
        # return the weights for each selected stock
        weights = selected * (1 / selected.abs().sum())
        return weights.sort_values()
        
    def OnData(self, data):
        # rebalance every 30 days
        
        # do nothing until next rebalance
        if self.Time < self.next_rebalance:
            return
        # open positions
        for ticker, weight in self.weights.items():
            # if the residual is way deviated from 0, we enter the position in the opposite way (mean reversion)
            self.SetHoldings(ticker, -weight)
        # update next rebalance time
        self.next_rebalance = self.Time + timedelta(self.rebalance_days)
    
    def monthly_rebalance(self):
        self.rebalence_flag = 1
    def rebalance(self):
        spy_hist = self.History([self.spy], 120, Resolution.Daily).loc[str(self.spy)]['close']
        if self.Securities[self.spy].Price < spy_hist.mean():
            for symbol in self.Portfolio.Keys:
                if symbol.Value != "VTI":
                    self.Liquidate()
            # Minimize risk factor and liquidate portfolio and buy VTI if protfolio is losing alot of money. 
            self.AddEquity("VTI")
            self.SetHoldings("VTI", 1)
            return
        if self.symbols is None: return
        chosen_df = self.calc_return(self.symbols)
        # Num_stocks is the amount of stocks chosen from list
        chosen_df = chosen_df.iloc[:self.num_stocks]
        
        self.existing_pos = 0
        add_symbols = []
        for symbol in self.Portfolio.Keys:
            if symbol.Value == 'SPY': continue
            if (symbol.Value not in chosen_df.index):  
                self.SetHoldings(symbol, 0)
            elif (symbol.Value in chosen_df.index): 
                self.existing_pos += 1
            
        weight = 0.99/len(chosen_df)
        for symbol in chosen_df.index:
            self.AddEquity(symbol)
            self.SetHoldings(symbol, weight)    
                
    def calc_return(self, stocks):
        hist = self.History(stocks, self.formation_days, Resolution.Daily)
        current = self.History(stocks, 1, Resolution.Minute)
        
        self.price = {}
        ret = {}
     
        for symbol in stocks:
            if str(symbol) in hist.index.levels[0] and str(symbol) in current.index.levels[0]:
                self.price[symbol.Value] = list(hist.loc[str(symbol)]['close'])
                self.price[symbol.Value].append(current.loc[str(symbol)]['close'][0])
        
        for symbol in self.price.keys():
            ret[symbol] = (self.price[symbol][-1] - self.price[symbol][0]) / self.price[symbol][0]
        df_ret = pd.DataFrame.from_dict(ret, orient='index')
        df_ret.columns = ['return']
        sort_return = df_ret.sort_values(by = ['return'], ascending = self.lowmom)
        
        return sort_return
