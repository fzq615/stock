import akshare as ak
import pandas as pd
from datetime import datetime

# 获取上证指数数据
df = ak.stock_zh_index_daily_em(symbol='sh000001')

# 获取最新交易日期
latest_date = df['date'].iloc[-1]
print(f'最新交易日: {latest_date}')