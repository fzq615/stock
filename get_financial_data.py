import akshare as ak
import pandas as pd
import os
import time
from datetime import datetime
import concurrent.futures
import threading

def get_stock_financial_data(stock_code):
    """
    获取单个股票的财务指标数据
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    dict: 包含财务指标的字典
    """
    try:
        # 确保股票代码格式正确（6位数字）
        stock_code = stock_code.strip().zfill(6)
        print(f"正在获取股票 {stock_code} 的财务指标数据...")
        
        # 使用akshare获取财务指标数据
        df = ak.stock_financial_analysis_indicator(symbol=stock_code)
        
        if df.empty:
            print(f"股票 {stock_code} 没有财务指标数据")
            return None
        
        # 获取最新的财务数据（按报表日期排序）
        df = df.sort_values(by='报表日期', ascending=False)
        latest_data = df.iloc[0].to_dict()
        
        # 提取主要财务指标
        key_indicators = [
            '报表日期', '基本每股收益(元)', '净资产收益率(%)', '每股净资产(元)', 
            '资产负债比率(%)', '每股资本公积金(元)', '每股未分配利润(元)',
            '每股经营现金流(元)', '销售毛利率(%)', '存货周转率(次)', 
            '总资产(元)', '流动资产(元)', '固定资产(元)', '无形资产(元)',
            '总负债(元)', '流动负债(元)', '长期负债(元)', '资本公积金(元)',
            '营业收入(元)', '营业利润(元)', '净利润(元)', '现金及现金等价物(元)',
            '经营现金流量净额(元)', '投资现金流量净额(元)', '筹资现金流量净额(元)'
        ]
        
        # 创建结果字典
        result = {}
        result['获取时间'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result['股票代码'] = stock_code
        
        # 添加可用的财务指标
        for indicator in key_indicators:
            if indicator in latest_data:
                result[indicator] = latest_data[indicator]
        
        # 计算一些额外的财务比率
        try:
            if '净利润(元)' in result and '总资产(元)' in result and float(result['总资产(元)']) != 0:
                result['总资产收益率(%)'] = round(float(result['净利润(元)']) / float(result['总资产(元)']) * 100, 2)
            
            if '营业利润(元)' in result and '营业收入(元)' in result and float(result['营业收入(元)']) != 0:
                result['营业利润率(%)'] = round(float(result['营业利润(元)']) / float(result['营业收入(元)']) * 100, 2)
                
            if '流动资产(元)' in result and '流动负债(元)' in result and float(result['流动负债(元)']) != 0:
                result['流动比率'] = round(float(result['流动资产(元)']) / float(result['流动负债(元)']), 2)
        except (ValueError, TypeError) as e:
            print(f"计算财务比率时出错: {e}")
        
        print(f"成功获取股票 {stock_code} 的财务指标数据")
        return result
    
    except Exception as e:
        print(f"获取股票 {stock_code} 的财务指标数据失败: {str(e)}")
        return None

def save_financial_data(stock_code, financial_data, output_dir):
    """
    将财务数据保存到文件
    
    参数:
    stock_code (str): 股票代码
    financial_data (dict): 财务数据字典
    output_dir (str): 输出目录
    """
    if financial_data is None:
        return False
    
    try:
        # 确保股票代码格式正确
        stock_code = stock_code.strip().zfill(6)
        
        # 创建文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"financial_info_{stock_code}_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # 将数据写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"股票财务信息 - 代码: {stock_code}\n")
            f.write(f"获取时间: {financial_data.get('获取时间', '未知')}\n")
            f.write("="*50 + "\n\n")
            
            # 写入财务指标
            f.write("基本信息:\n")
            f.write(f"报表日期: {financial_data.get('报表日期', '未知')}\n\n")
            
            f.write("盈利能力:\n")
            f.write(f"基本每股收益(元): {financial_data.get('基本每股收益(元)', '未知')}\n")
            f.write(f"净资产收益率(%): {financial_data.get('净资产收益率(%)', '未知')}\n")
            f.write(f"总资产收益率(%): {financial_data.get('总资产收益率(%)', '未知')}\n")
            f.write(f"营业利润率(%): {financial_data.get('营业利润率(%)', '未知')}\n")
            f.write(f"销售毛利率(%): {financial_data.get('销售毛利率(%)', '未知')}\n\n")
            
            f.write("资产状况:\n")
            f.write(f"每股净资产(元): {financial_data.get('每股净资产(元)', '未知')}\n")
            f.write(f"资产负债比率(%): {financial_data.get('资产负债比率(%)', '未知')}\n")
            f.write(f"流动比率: {financial_data.get('流动比率', '未知')}\n")
            f.write(f"总资产(元): {financial_data.get('总资产(元)', '未知')}\n")
            f.write(f"流动资产(元): {financial_data.get('流动资产(元)', '未知')}\n")
            f.write(f"固定资产(元): {financial_data.get('固定资产(元)', '未知')}\n")
            f.write(f"无形资产(元): {financial_data.get('无形资产(元)', '未知')}\n\n")
            
            f.write("负债状况:\n")
            f.write(f"总负债(元): {financial_data.get('总负债(元)', '未知')}\n")
            f.write(f"流动负债(元): {financial_data.get('流动负债(元)', '未知')}\n")
            f.write(f"长期负债(元): {financial_data.get('长期负债(元)', '未知')}\n\n")
            
            f.write("现金流量:\n")
            f.write(f"每股经营现金流(元): {financial_data.get('每股经营现金流(元)', '未知')}\n")
            f.write(f"经营现金流量净额(元): {financial_data.get('经营现金流量净额(元)', '未知')}\n")
            f.write(f"投资现金流量净额(元): {financial_data.get('投资现金流量净额(元)', '未知')}\n")
            f.write(f"筹资现金流量净额(元): {financial_data.get('筹资现金流量净额(元)', '未知')}\n")
            f.write(f"现金及现金等价物(元): {financial_data.get('现金及现金等价物(元)', '未知')}\n\n")
            
            f.write("其他指标:\n")
            f.write(f"每股资本公积金(元): {financial_data.get('每股资本公积金(元)', '未知')}\n")
            f.write(f"每股未分配利润(元): {financial_data.get('每股未分配利润(元)', '未知')}\n")
            f.write(f"存货周转率(次): {financial_data.get('存货周转率(次)', '未知')}\n")
            
        print(f"股票 {stock_code} 的财务数据已保存到 {filepath}")
        return True
    
    except Exception as e:
        print(f"保存股票 {stock_code} 的财务数据失败: {str(e)}")
        return False

def process_stock(stock_code, output_dir):
    """
    处理单个股票：获取财务数据并保存
    
    参数:
    stock_code (str): 股票代码
    output_dir (str): 输出目录
    
    返回:
    bool: 是否成功处理
    """
    try:
        # 获取财务数据
        financial_data = get_stock_financial_data(stock_code)
        
        # 如果获取成功，保存数据
        if financial_data:
            return save_financial_data(stock_code, financial_data, output_dir)
        else:
            return False
    except Exception as e:
        print(f"处理股票 {stock_code} 时出错: {str(e)}")
        return False

def batch_process_stocks_from_csv(csv_file, output_dir, max_workers=5):
    """
    从CSV文件批量处理股票
    
    参数:
    csv_file (str): CSV文件路径
    output_dir (str): 输出目录
    max_workers (int): 最大线程数
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 确保code列存在
        if 'code' not in df.columns:
            print("CSV文件中没有'code'列")
            return
        
        # 提取唯一的股票代码
        df['code'] = df['code'].astype(str).str.zfill(6)
        unique_codes = df['code'].unique()
        print(f"从CSV文件中提取了 {len(unique_codes)} 个唯一股票代码")
        
        # 使用线程池并行处理
        success_count = 0
        fail_count = 0
        lock = threading.Lock()
        
        def process_with_counter(code):
            nonlocal success_count, fail_count
            result = process_stock(code, output_dir)
            with lock:
                if result:
                    success_count += 1
                else:
                    fail_count += 1
            # 添加延迟以避免API限制
            time.sleep(1)
        
        print(f"使用 {max_workers} 个线程开始处理...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_with_counter, code) for code in unique_codes]
            concurrent.futures.wait(futures)
        
        print(f"处理完成! 成功: {success_count}, 失败: {fail_count}, 总计: {len(unique_codes)}")
        
    except Exception as e:
        print(f"批量处理股票时出错: {str(e)}")

def main():
    # 设置文件路径
    csv_file = "E:\\20250305-bi\\pattern_results.csv"
    output_dir = "E:\\20250305-bi\\stock_info"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"CSV文件不存在: {csv_file}")
        return
    
    # 批量处理股票
    batch_process_stocks_from_csv(csv_file, output_dir, max_workers=5)

if __name__ == "__main__":
    main()