import pandas as pd
from sklearn.preprocessing import scale
import numpy as np
from datetime import timedelta


# main data


# ==============================
bitcoin = pd.read_csv('./data/bitcoin-usd.csv', parse_dates=['date'])



# ==============================
sp500 = pd.read_csv('./data/sp500.csv', parse_dates=['date'])




monthly_data = pd.read_csv('./data/monthly_data.csv', parse_dates=['date'])

bitcoin.dropna(inplace=True)


# Log-scale data
bitcoin_scaled = bitcoin.copy(deep=True)
bitcoin_log = bitcoin.copy(deep=True)
bitcoin_scaled['scaled_returns'] = scale(bitcoin_scaled['close'])
bitcoin_log['log_returns'] = np.log(bitcoin_log['close']/bitcoin_log['close'].shift(1))
bitcoin_log.dropna(inplace=True)

sp500_scaled = sp500.copy(deep=True)
sp500_log = sp500.copy(deep=True)
sp500_scaled['scaled_returns'] = scale(sp500_scaled['close'])
sp500_log['log_returns'] = np.log(sp500_log['close']/sp500_log['close'].shift(1))
sp500_log.dropna(inplace=True)


#Merge data
sp500_btc = sp500.merge(bitcoin, on = 'date', how = 'left', suffixes=['_sp500','_btc']).dropna()
sp500_btc_cp = sp500_btc.copy(deep=True)
sp500_btc_cp.set_index('date', inplace=True)

sp500_btc_monthly = sp500_btc_cp.resample('M').last()

monthly_data_cp = monthly_data.copy(deep=True)
monthly_data_cp['date'] = monthly_data_cp['date'] - timedelta(days=1)

closed_price = sp500_btc_monthly.merge(monthly_data_cp, on='date', how='left')
closed_price.drop(['open_sp500', 'high_sp500', 'low_sp500', 'volume_sp500', 'open_btc','high_btc','low_btc', 'volume_btc'], axis=1, inplace=True)

# ranking close price
rank_by_btc = closed_price.copy(deep=True)



# OBV data
bitcoin_obv = bitcoin.copy(deep=True)
bitcoin_obv['obv'] = (np.sign(bitcoin_obv['close'].diff()) * bitcoin_obv['volume']).fillna(0).cumsum()

sp500_obv = sp500.copy(deep=True)
sp500_obv['obv'] = (np.sign(sp500_obv['close'].diff()) * sp500_obv['volume']).fillna(0).cumsum()


# standardize gains on monthly

std_gains = closed_price.copy(deep=True)
std_gains['gold_usd'] = std_gains.gold_usd.fillna(method='ffill')
std_gains['btc_std'] = std_gains.close_btc / std_gains.close_btc[0] * 100
std_gains['sp_std'] = std_gains.close_sp500 / std_gains.close_sp500[0] * 100
std_gains['gold_std'] = std_gains.gold_usd / std_gains.gold_usd[0] * 100

# standardize gains on monthly
std_gains_cp = std_gains.copy(deep=True)
std_gains_cp = std_gains_cp[['btc_std', 'sp_std', 'gold_std', 'date']]
std_gains_cp.set_index('date', inplace=True)
std_year = std_gains_cp.resample('Y').first()
std_year['sp500_pct'] = std_year.sp_std.pct_change() * 100
std_year['btc_pct'] = std_year.btc_std.pct_change() * 100
std_year['gold_pct'] = std_year.gold_std.pct_change() * 100
std_year.drop(columns=['btc_std', 'sp_std', 'gold_std'], inplace=True)
std_year.reset_index(inplace=True)

# covariance of std_gains
covariance_table = std_gains[['sp_std', 'btc_std', 'gold_std']].cov()
covariance_table.style.format({'sp_std': '{:.2f}', 'btc_std': '{:.2f}', 'gold_std': '{:.2f}'})


# DPC
dcp_df = closed_price.copy(deep=True)

dcp_df['sp500_dcp'] = dcp_df.close_sp500.pct_change()
dcp_df['btc_dcp'] = dcp_df.close_btc.pct_change()
dcp_df['gold_dcp'] = dcp_df.gold_usd.pct_change()
dcp_df['cpi_us_dcp'] = dcp_df.cpi_us.pct_change()
dcp_df.fillna(0, inplace=True)


# DPC report 
def value_at_risk_single(data, confidence):
    
    """
    # Calculate value at risk of a single asset 
    ## Confidence on scale 100 
    """
    return data.quantile(1- confidence/100)


def expected_shortfall(data, confidence):
    """
    Expected Shortfall hay CVaR đc định nghĩa là mean của các dữ liệu vượt ngoài VaR
    """
    var_value = value_at_risk_single(data, confidence)
    
    return data[data <= var_value].mean()


def max_drawdown(data):
    cum_return = (1+data).cumprod()
    peak = cum_return.expanding(min_periods=1).max()
    drawdown = (cum_return/peak) - 1
    return drawdown.min()


class DCP_Dist_Moments_Report:
    """
    Generates a retport for the DCP Distribution moments for calculating the risk
    """
    def __init__(self, data) -> None:
        self.data = data
        
    def calc_moments(self):
        """
        Simple moments function from pd method
        """
        volatility = self.data.std()
        volatility.name = 'volatility ⬇'
        kurtosis = self.data.kurt(axis=0)
        kurtosis.name = 'kurtosis ⬇'
        
        self.volatility = volatility
        self.kurtosis = kurtosis
    
    def calc_lambda(self, confidence):
        VaR = self.data.apply(lambda x: value_at_risk_single(x, confidence))
        VaR.name = f'VaR (at {confidence}) ⬆'
        CVaR = self.data.apply(lambda x: expected_shortfall(x, confidence))
        CVaR.name = f'CVaR (at {confidence}) ⬆'
        mdd = self.data.apply(lambda x: max_drawdown(x))
        mdd.name = 'MDD ⬆'
        
        self.VaR = VaR
        self.CVaR = CVaR
        self.mdd = mdd
        
    def output(self):
        a = pd.concat([self.volatility, self.kurtosis, self.VaR, self.CVaR, self.mdd], axis=1).transpose()
        return a
    
    @staticmethod
    def make_pretty(styler):
        styler.format('{:.3f}')
        styler.set_caption('⬇ means the lower the better, ⬆ otherwise')
        styler.bar(axis=1, align=0, height=50, width=60, color=['red', 'blue'], props="width: 120px; border-right: 1px solid black;")
        return styler
    
    def __call__(self, confidence):
        self.calc_moments()
        self.calc_lambda(confidence=confidence)
        df = self.output()
        
        return df.style.pipe(self.make_pretty)



class CompareExtremeCVaR:
    """
    Compare CVaR at 80, 90, 95, 99 percentile to compare the incremental risk
    """
    def __init__(self, data) -> None:
        self.data = data
        
    def calc_CVaR(self):
        self.es80 = expected_shortfall(self.data, 80)
        self.es90 = expected_shortfall(self.data, 90)
        self.es95 = expected_shortfall(self.data, 95)
        self.es99 = expected_shortfall(self.data, 99)

    def change_from_(self):
        self.change_es80 = (self.es80 - self.es80)/self.es80
        self.change_es90 = (self.es90 - self.es80)/self.es80
        self.change_es95 = (self.es95 - self.es90)/self.es90
        self.change_es99 = (self.es99 - self.es95)/self.es95
        
    def output(self):
        a = pd.concat([self.es80, self.es90, self.es95, self.es99, self.change_es80, self.change_es90, self.change_es95, self.change_es99], axis=1).transpose()
        a.index = ['CVaR at 80', 'CVaR at 90', 'CVaR at 95', 'CVaR at 99', 'PCT - 80', 'PCT - 90', 'PCT - 95', 'PCT - 99']
        return a  

    @staticmethod
    def make_pretty(styler):
        styler.format('{:.3f}')
        styler.bar(axis=1, align=0, height=50, width=60, color=['red', 'blue'], props="width: 120px; border-right: 1px solid black;")
        return styler
    

    def __call__(self):
        self.calc_CVaR()
        self.change_from_()
        df = self.output()
        return df.style.pipe(self.make_pretty)
    
    

sharpe_ratio = lambda data, risk_free_rate=0: (data.mean() - risk_free_rate) / data.std()
sortino_ratio = lambda data, risk_free_rate=0: (data.mean() - risk_free_rate) / data[data<0].std()
m2_ratio = lambda data, benchmark_returns, risk_free=0: (sharpe_ratio(data, risk_free) * benchmark_returns.std()) + risk_free

def information_ratio(data, benchmark_returns):
    return_difference = data - benchmark_returns
    volatility = return_difference.std()
    if volatility == 0:
        return np.nan
    return return_difference.mean() / volatility


# ============================
class PortfolioPerformance:
    
    def __init__(self, data:pd.DataFrame, weights:np.array) -> None:
        self.data = data
        self.weights = weights

    def get_data(self):
        pct_returns = self.data.pct_change()
        mean_returns = pct_returns.mean()
        cov_matrix = pct_returns.cov()
        return pct_returns, mean_returns, cov_matrix
    
    def portfolio_performance(self):
        pct_returns, mean_returns, cov_matrix = self.get_data()
        weighted_returns = np.sum(pct_returns*self.weights)
        weighted_std = np.sqrt(np.dot( self.weights.T, np.dot(cov_matrix, self.weights) ))
        return weighted_returns, weighted_std
        
    def calc_portfolio_pct(self):
        pct_returns, _, _ = self.get_data()
        returned_data = self.data.copy(deep=True)
        returned_data['portfolio_pct'] = pct_returns.dot(self.weights)
        return returned_data
    
    
portfolio_data = closed_price.copy(deep=True)
portfolio_data.set_index('date', inplace=True)
portfolio_data = portfolio_data[['close_btc', 'close_sp500', 'gold_usd']]

# ============================
class TwoVariablePortfolioCalc:
    """
    # Display the ratio of two variable in a multiple assets portfolio
    """
    
    def __init__(self, data) -> None:
        self.data = data
        self.rp = {}
        
    def add_steps(self, var1, var2):
        self.var1 = var1
        self.var2 = var2
        
    def seed_weights(self, var1_weight, var2_weight):
        num_assets = len(self.data.columns)
        weights = np.zeros(num_assets)
        weights[0] = var1_weight
        weights[1] = var2_weight
        weights[2:] = ((1 - var1_weight - var2_weight) / (num_assets - 2))
        return weights
    
    def create_portfolio_stat(self):
        for i in self.var1:
            for j in self.var2:
                if i + j > 1:
                    rp = None
                else:
                    PP = PortfolioPerformance(self.data, weights=self.seed_weights(i, j))
                    rp = PP.calc_portfolio_pct()
                    rp.fillna(0, inplace=True)
                if i in self.rp:
                    self.rp[i].append((j, rp))
                else:
                    self.rp[i] = [(j, rp)]
    
    def create_volatility_rp(self, label1, label2):
        result_dict = {}
        for index, value in self.rp.items():
            renamed_index = f'{index*100:.0f}% {label1}'
            for i in value:
                if index + i[0] > 1:
                    volitality = np.nan
                else:
                    volitality = i[1]['portfolio_pct'].std()
                if renamed_index in result_dict:
                    result_dict[renamed_index].append(volitality)
                else:
                    result_dict[renamed_index] = [volitality]
                    
        df = pd.DataFrame(result_dict)
        df.index = [f'{i*100:.0f}% {label2}' for i in self.var2]
        return df.style.highlight_min(axis=None, color='gold').set_caption('Volitality report')
    
    def create_ratio_rp(self, ratio_fn, label1, label2, fn_label):
        result_dict = {}
        for index, value in self.rp.items():
            renamed_index = f'{index*100:.0f}% {label1}'
            for i in value:
                if index + i[0] > 1:
                    rp = np.nan
                else:
                    rp = ratio_fn(i[1]['portfolio_pct'])
                if renamed_index in result_dict:
                    result_dict[renamed_index].append(rp)
                else:
                    result_dict[renamed_index] = [rp]
                    
        df = pd.DataFrame(result_dict)
        df.index = [f'{i*100:.0f}% {label2}' for i in self.var2]
        return df.style.highlight_max(axis=None, color='gold').set_caption(f'{fn_label} report')
                
    def create_var_rp(self, var_fn, confidence, label1, label2, fn_label):
        result_dict = {}
        for index, value in self.rp.items():
            renamed_index = f'{index*100:.0f}% {label1}'
            for i in value:
                if index + i[0] > 1:
                    rp = np.nan
                else:
                    rp = var_fn(i[1]['portfolio_pct'], confidence)
                if renamed_index in result_dict:
                    result_dict[renamed_index].append(rp)
                else:
                    result_dict[renamed_index] = [rp]
                    
        df = pd.DataFrame(result_dict)
        df.index = [f'{i*100:.0f}% {label2}' for i in self.var2]
        return df.style.highlight_max(axis=None, color='gold').set_caption(f'{fn_label} at {confidence} percentile report')

bitcoin_ratio = (.2, .4, .6, .8)
sp500_ratio = (.2, .4, .6, .8)

tt = TwoVariablePortfolioCalc(portfolio_data) 
tt.add_steps(bitcoin_ratio, sp500_ratio)
tt.create_portfolio_stat()

# ============================
class SimulateEfficientFrontier:
    
    def __init__(self, data, n) -> None:
        self.data = data
        self.n = n
    
    @staticmethod
    def seed_random_weights(data):
        weights = np.random.random(len(data.columns))
        weights /= np.sum(weights)
        return weights
    
    def simulation_weight(self):
        np.random.seed(21)
        simul_weight = []
        for i in range(self.n):
            simul_weight.append(self.seed_random_weights(self.data))
            
        self.simul_weight = simul_weight
        
    def calculate_return(self):
        risk_return = []

        for i in self.simul_weight:
            PP = PortfolioPerformance(portfolio_data, i)
            portfolio_pct = PP.calc_portfolio_pct()['portfolio_pct']
            mean_return = portfolio_pct.mean()
            volatility = portfolio_pct.std()
            sortino = sortino_ratio(portfolio_pct)
            weight_seeded = i
            risk_return.append([mean_return, volatility, sortino, weight_seeded])
            
        portfolio_df = pd.DataFrame(risk_return)
        portfolio_df.columns = ['mean', 'volatility', 'sortino ratio', 'weight seed']
        portfolio_df['index_col'] = np.arange(len(portfolio_df))
        
        return portfolio_df
    
    def __call__(self):
        self.simulation_weight()
        return self.calculate_return()



closed_price_infl = closed_price.copy(deep=True)
closed_price_infl.set_index('date', inplace=True)
inflation_year = closed_price_infl.resample('Y').last()

inflation_year['inflation'] = inflation_year.cpi_us.pct_change() * 100
inflation_year['sp_pct'] = inflation_year.close_sp500.pct_change() *100
inflation_year['btc_pct'] = inflation_year.close_btc.pct_change() *100
inflation_year['gold_pct'] = inflation_year.gold_usd.pct_change() *100
inflation_year.fillna(0, inplace=True)
inflation_year.drop(columns=['close_sp500', 'close_btc', 'gold_usd','cpi_us'], inplace=True)



inflation_month = closed_price_infl.resample('M').last()
inflation_month['inflation'] = inflation_month.cpi_us.pct_change() * 100
inflation_month['sp_pct'] = inflation_month.close_sp500.pct_change() *100
inflation_month['btc_pct'] = inflation_month.close_btc.pct_change() *100
inflation_month['gold_pct'] = inflation_month.gold_usd.pct_change() *100
inflation_month.fillna(0, inplace=True)

# extract khoảng giá trị Bitcoin khi mà CPI percentile > 80% (extreme gains)

percent_80 = np.percentile(inflation_month.inflation, 80)
percent_20 = np.percentile(inflation_month.inflation, 20)

extreme_btc = inflation_month[inflation_month.inflation > percent_80]['btc_pct']
lowest_btc = inflation_month[inflation_month.inflation < percent_20]['btc_pct']
middle_btc = inflation_month[(inflation_month.inflation >= percent_20) & (inflation_month.inflation <= percent_80)]['btc_pct']

extreme_sp = inflation_month[inflation_month.inflation > percent_80]['sp_pct']
lowest_sp = inflation_month[inflation_month.inflation < percent_20]['sp_pct']
middle_sp = inflation_month[(inflation_month.inflation >= percent_20) & (inflation_month.inflation <= percent_80)]['sp_pct']

extreme_gold = inflation_month[inflation_month.inflation > percent_80]['gold_pct']
lowest_gold = inflation_month[inflation_month.inflation < percent_20]['gold_pct']
middle_gold = inflation_month[(inflation_month.inflation >= percent_20) & (inflation_month.inflation <= percent_80)]['gold_pct']

inflation_rate = 0.05    
    
from scipy.stats import linregress

sp_linear = linregress(inflation_year.inflation, inflation_year.sp_pct)
btc_linear = linregress(inflation_year.inflation, inflation_year.btc_pct)
gold_linear = linregress(inflation_year.inflation, inflation_year.gold_pct)


sp_slope, sp_r2 = sp_linear.slope, sp_linear.rvalue
btc_slope, btc_r2 = btc_linear.slope, btc_linear.rvalue
gold_slope, gold_r2 = gold_linear.slope, gold_linear.rvalue

sp_500_rp=pd.Series([sp_slope, sp_r2], name='Inflation vs S&P500')
btc_rp=pd.Series([btc_slope, btc_r2], name='Inflation vs Bitcoin')
gol_rp=pd.Series([gold_slope, gold_r2], name='Inflation vs Gold')

slope_rp = pd.concat([sp_500_rp, btc_rp, gol_rp], axis=1)
slope_rp.index = ['Slope', 'R-squared']


rank_by_btc = closed_price.copy(deep=True)
rank_by_btc['close_sp500'] = rank_by_btc['close_btc'] / rank_by_btc['close_sp500'] *100
rank_by_btc['gold_usd'] = rank_by_btc['close_btc'] / rank_by_btc['gold_usd'] *100
rank_by_btc.drop(columns=['close_btc', 'cpi_us'], inplace=True)