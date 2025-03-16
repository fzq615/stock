import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Union, Callable
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

class Q_factor(object):
    def __init__(self, securities: Union[str, list], watch_date: str, N: int = 10, 
                 frequency: str = '5m', mod: str = 'normal'):
        self.securities = securities if isinstance(securities, list) else [securities]
        self.watch_date = pd.to_datetime(watch_date)
        self.frequency = frequency
        self.N = N
        self.mod = mod
        self._get_count()
        
    def _get_count(self):
        ALL_DAY = 240
        if self.frequency[-1] != 'm':
            raise ValueError('frequency参数必须是X minute(X必须是整数)')
        self.minute_count = ALL_DAY / int(self.frequency.replace('m', ''))
        
    def get_data(self):
        data_list = []
        for stock in self.securities:
            print('stock',stock,self.frequency[:-1])

            try:
                df = ak.stock_zh_a_hist_min_em(symbol=stock, period=self.frequency[:-1], adjust='qfq')
                df['code'] = stock
                df['time'] = pd.to_datetime(df['时间'])
                data_list.append(df)
                print(len(df))
                
            except Exception as e:
                print(f"获取股票 {stock} 数据时出错: {str(e)}")
                
        self.data = pd.concat(data_list)
        self.data.to_csv('df0.csv')
    def calc(self, beta: float) -> pd.Series:
        self.beta = round(beta, 2)
        data = self.data.query('成交量 != 0').copy()
        data['return'] = data['收盘'] / data['开盘'] - 1
        data['abs_return'] = data['return'].abs()
        
        if self.mod == 'normal':
            data['S'] = data['abs_return'] / data['成交量'].pow(beta)
        else:
            data['S'] = data['abs_return'] / np.log(data['成交量'])
            
        # data按照‘s’列降序排列【从大到小排序】
        data = data.sort_values('S', ascending=False)
        


        result = data.groupby('code').apply(self.calc_Q)
        result.name = self.name
        return result

        

    def calc_Q(self, df: pd.DataFrame) -> float:
        def vwap(data: pd.DataFrame) -> float:
            return np.average(data['收盘'], weights=data['成交量'])
   
        df = df.sort_values('S', ascending=False)
        df['vol_cum_ratio'] = df['成交量'].cumsum() / df['成交量'].sum()
        df.to_csv('df2.csv')
        smart_trades = df[df['vol_cum_ratio'] <= 0.2]
        
        if smart_trades.empty and df['vol_cum_ratio'].iloc[0] > 0.2:
            smart_trades = df.iloc[:1]
            
        vwap_smart = vwap(smart_trades)
        vwap_all = vwap(df)
        
        return vwap_smart / vwap_all
    
    def plot_factor_analysis(self, stock_code: str):
        """绘制因子分析图表"""
        try:

            import matplotlib.pyplot as plt
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号


            print("数据列名:", self.data.columns.tolist())

            df = self.data[self.data['code'] == stock_code].copy()
            df.to_csv('df1.csv')
            # 计算S指标
            df.loc[:, 'return'] = df['收盘'] / df['开盘'] - 1
            if self.mod == 'normal':
                df.loc[:, 'S'] = df['return'].abs() / df['成交量'].pow(self.beta)
            else:
                df.loc[:, 'S'] = df['return'].abs() / np.log(df['成交量'])
            
            
            # 计算每个时间点的滚动Q因子
            #计算近10个交易日的window_size ，当前为5分钟k线，10个交易日为10*240=2400个5分钟k线
            window_size = 480


            #window_size = 20  # 使用20个周期计算一次Q因子
            Q_factors = []
            i=0
            for i in range(len(df)):
                if i < window_size - 1:
                    Q_factors.append(np.nan)
                    i+=1
                else:
                    window_data = df.iloc[i-window_size+1:i+1]
                    Q = self._calc_window_Q_from_df(window_data)
                    #('iqq',i,Q)

                    Q_factors.append(Q)
            df['Q_factor'] = Q_factors
            #df按时间排序
            df = df.sort_values('时间')
            # 准备K线数据
            df.loc[:, 'datetime'] = pd.to_datetime(df['时间'])
            df_ohlc = df[['datetime', '开盘', '最高', '最低', '收盘']].copy()
            df_ohlc['datetime'] = df_ohlc['datetime'].map(mdates.date2num)
            df.to_csv('smtdf.csv', index=False)
            # 创建图形
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])
            
            # 1. K线图
            ax1 = fig.add_subplot(gs[0])
            x = np.arange(len(df))  # 使用索引作为x轴
            ohlc = np.column_stack((x, df[['开盘', '最高', '最低', '收盘']].values))
            candlestick_ohlc(ax1, ohlc, width=0.6, colorup='red', colordown='green')
        
            def format_date(x, p):
                try:
                    if x >= 0 and x < len(df):
                        return pd.to_datetime(df['时间'].iloc[int(x)]).strftime('%H:%M')
                    return ''
                except:
                    return ''

            ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
            
            # 调整x轴刻度数量
            ax1.xaxis.set_major_locator(plt.MaxNLocator(10))  # 最多显示10个刻度
            ax1.set_xlim(-1, len(df))  # 设置x轴范围
            ax1.set_title(f'股票{stock_code}的K线图和聪明钱因子分析 (β={self.beta})')
            ax1.grid(True, linestyle='--', alpha=0.3)  # 添加网格线
            
            # 2. Q因子走势
            ax2 = fig.add_subplot(gs[1], sharex=ax1)  # 共享x轴
            ax2.plot(range(len(df)), df['Q_factor'], 'b-', label='Q因子')
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Q因子')
            ax2.legend(loc='upper left')
            
            # 3. 成交量和S指标
            ax3 = fig.add_subplot(gs[2], sharex=ax1)  # 共享x轴
            ax3.bar(range(len(df)), df['成交量'], color='blue', alpha=0.5, label='成交量')
            ax3.set_ylabel('成交量')
            ax3.legend(loc='upper left')
            
            ax3_twin = ax3.twinx()
            ax3_twin.plot(range(len(df)), df['S'], 'r.', label='S指标', alpha=0.7)
            ax3_twin.set_ylabel('S指标')
            ax3_twin.legend(loc='upper right')
            


            
            #plt.tight_layout()
            # 直接显示
            #plt.ion()  # 打开交互模式
            #plt.show()
            #plt.show(block=True)  # 使用 block=True 来阻塞程序执行
            
            return df['Q_factor'].iloc[-1]
            
        except Exception as e:
            print(f"绘制股票 {stock_code} 的分析图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _calc_window_Q_from_df(self, df: pd.DataFrame) -> float:
        """从DataFrame计算Q因子"""
        try:
            # 计算S指标
            df.loc[:, 'return'] = df['收盘'] / df['开盘'] - 1
            if self.mod == 'normal':
                df.loc[:, 'S'] = df['return'].abs() / df['成交量'].pow(self.beta)
            else:
                df.loc[:, 'S'] = df['return'].abs() / np.log(df['成交量'])
            
            # 按S指标排序
            df = df.sort_values('S', ascending=False)
            df['vol_cum_ratio'] = df['成交量'].cumsum() / df['成交量'].sum()
            smart_trades = df[df['vol_cum_ratio'] <= 0.2]
            
            if smart_trades.empty and df['vol_cum_ratio'].iloc[0] > 0.2:
                smart_trades = df.iloc[:1]
                
            # 计算VWAP
            vwap_smart = np.average(smart_trades['收盘'], weights=smart_trades['成交量'])
            vwap_all = np.average(df['收盘'], weights=df['成交量'])
            
            return vwap_smart / vwap_all
        except:
            return np.nan
import akshare as ak
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def get_hs300_stocks():
    """获取沪深300成分股"""
    hs300 = ak.index_stock_cons_weight_csindex(symbol="000300")
    return hs300['成分券代码'].tolist()

def calculate_smart_money():
    # 获取沪深300成分股
    stocks = get_hs300_stocks()
    print(f"获取到{len(stocks)}只沪深300成分股")
    
    # 计算结果存储
    results = []
    
    # 创建Q因子计算实例
    from smt import Q_factor
    factor = Q_factor(stocks[0], datetime.now().strftime('%Y-%m-%d'))
    
    # 遍历计算每只股票
    for stock in tqdm(stocks, desc="计算聪明钱因子"):
        try:
            # 更新股票代码
            factor.securities = [stock]
            # 获取数据
            factor.get_data()
            # 设置beta值
            factor.beta = -0.5
            # 计算Q因子
            Q = factor.plot_factor_analysis(stock)
            
            if Q is not None:
                results.append({
                    'stock': stock,
                    'Q_factor': Q
                })
                print(f"{stock} Q因子: {Q:.4f}")
            
        except Exception as e:
            print(f"计算股票 {stock} 时出错: {str(e)}")
            continue
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Q_factor', ascending=False)
    
    # 保存结果
    results_df.to_csv('hs300_smart_money.csv', index=False)
    print("\n计算完成，结果已保存至 hs300_smart_money.csv")
    
    # 显示前10名和后10名
    print("\n聪明钱因子最高的10只股票：")
    print(results_df.head(10))
    print("\n聪明钱因子最低的10只股票：")
    print(results_df.tail(10))


if __name__ == "__main__":

    calculate_smart_money()

    zzz
    try:
        factor = Q_factor("000001", datetime.now().strftime('%Y-%m-%d'))
        factor.get_data()
        factor.beta = -0.5
        Q = factor.plot_factor_analysis("000001")
        if Q is not None:
            print(f"最新的聪明钱因子Q = {Q:.4f}")
        else:
            print("计算聪明钱因子失败")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()