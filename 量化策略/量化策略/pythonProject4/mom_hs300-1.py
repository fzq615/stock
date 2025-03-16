import akshare as ak
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import os

def get_stock_data(stock_list, days=25):
    """获取股票历史数据并计算动量得分
    
    Args:
        stock_list: 股票代码列表
        days: 计算动量的天数，默认25天
    """
    score_list = []
    
    # 获取所有A股的基本信息
    try:
        all_stocks = ak.stock_info_a_code_name()
        stock_names_dict = dict(zip(all_stocks['code'], all_stocks['name']))
    except Exception as e:
        print(f"获取股票基本信息时出错: {str(e)}")
        stock_names_dict = {}
    
    for stock in stock_list:
        try:
            # 从字典中获取股票名称
            if stock in stock_names_dict:
                stock_name = stock_names_dict[stock]
            else:
                stock_name = "未知"
                print(f"警告: 无法找到股票代码 {stock} 的名称")
            
            # 获取股票日线数据
            df = ak.stock_zh_a_hist(symbol=stock, period="daily", 
                                  start_date=(datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d'),
                                  end_date=datetime.now().strftime('%Y%m%d'))
            
            # 计算动量得分
            df = df.tail(days)
            y = np.log(df['收盘'].values)
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            annualized_returns = math.pow(math.exp(slope), 250) - 1
            r_squared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
            score = annualized_returns * r_squared
            
            score_list.append({
                'stock_code': stock, 
                'stock_name': stock_name,
                'score': score
            })
            print(f"股票 {stock_name}({stock}) 的动量得分: {score:.4f}")
            
        except Exception as e:
            print(f"处理股票 {stock} 时出错: {str(e)}")
            continue
    
    # 转换为DataFrame并排序
    df_scores = pd.DataFrame(score_list)
    df_scores = df_scores.sort_values(by='score', ascending=False)
    
    return df_scores

def main():
    # 创建保存结果的目录
    #output_dir = "momentum_results"
    output_dir = "D:/momentum_results"  # 指定具体路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取沪深300成分股
    stock_list = ak.index_stock_cons_weight_csindex(symbol="000300")['成分券代码'].tolist()
    stock_list = [str(code).zfill(6) for code in stock_list]
    
    # 计算动量得分
    result = get_stock_data(stock_list)
    
    # 格式化得分为保留4位小数
    result['score'] = result['score'].apply(lambda x: f"{x:.4f}")
    
    # 保存结果到CSV文件
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(output_dir, f'momentum_scores_{current_time}.csv')
    result.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到文件: {os.path.abspath(csv_filename)}")
    
    # 打印前20名
    print("\n动量因子得分前20名：")
    print(result.head(20))
    
    return result.head(20)

if __name__ == "__main__":
    top_stocks = main()
