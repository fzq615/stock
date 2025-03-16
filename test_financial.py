import akshare as ak
import pandas as pd

try:
    # 获取财务指标数据
    print("正在获取股票财务指标数据...")
    df = ak.stock_financial_analysis_indicator(symbol='000001')
    print('获取财务指标成功，前3行数据:')
    pd.set_option('display.max_columns', None)
    print(df.head(3))
    print('\n列名:')
    print(list(df.columns))
    
    # 获取主要财务指标
    print("\n主要财务指标:")
    key_indicators = ['基本每股收益(元)', '净资产收益率(%)', '每股净资产(元)', 
                     '资产负债比率(%)', '每股资本公积金(元)', '每股未分配利润(元)',
                     '每股经营现金流(元)', '销售毛利率(%)', '存货周转率(次)']
    if set(key_indicators).issubset(set(df.columns)):
        print(df[['报表日期'] + key_indicators].head(5))
    else:
        print("未找到所有主要财务指标，可用的指标有:")
        print(df.columns[:10])
        
except Exception as e:
    print(f'执行出错: {e}')