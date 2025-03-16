import pandas as pd
import os
import time
import threading
import concurrent.futures
from datetime import datetime
from stock_financial import get_stock_financial_data
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_financial_data(stock_code, financial_data, output_dir):
    """
    将财务数据保存到文件
    
    参数:
    stock_code (str): 股票代码
    financial_data (dict): 财务数据字典
    output_dir (str): 输出目录
    """
    if financial_data is None:
        logger.warning(f"股票 {stock_code} 的财务数据为空，跳过保存")
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
            
            # 处理基本财务指标
            if '基本财务指标' in financial_data:
                f.write("基本财务指标:\n")
                for key, value in financial_data['基本财务指标'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # 处理详细财务指标数据
            if '财务指标数据' in financial_data and financial_data['财务指标数据']:
                f.write("详细财务指标数据:\n")
                # 获取第一条记录的所有字段
                if len(financial_data['财务指标数据']) > 0:
                    first_record = financial_data['财务指标数据'][0]
                    for key, value in first_record.items():
                        f.write(f"{key}: {value}\n")
                    
                    # 如果有多条记录，添加其他记录的摘要
                    if len(financial_data['财务指标数据']) > 1:
                        f.write("\n其他历史财务数据记录数: " + str(len(financial_data['财务指标数据'])-1) + "\n")
            
            # 处理错误信息
            if '错误信息' in financial_data:
                f.write("\n错误信息:\n")
                f.write(f"{financial_data['错误信息']}\n")
                
                # 即使有错误，也确保基本财务指标部分存在
                if '基本财务指标' not in financial_data:
                    f.write("\n基本财务指标 (默认值):\n")
                    f.write("市盈率: None\n")
                    f.write("市净率: None\n")
                    f.write("总市值: None\n")
                    f.write("流通市值: None\n")
                    f.write("换手率: None\n")
                    f.write("量比: None\n")
        
        logger.info(f"股票 {stock_code} 的财务数据已保存到 {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"保存股票 {stock_code} 的财务数据失败: {str(e)}")
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
        # 确保股票代码格式正确（6位数字）
        stock_code = stock_code.strip().zfill(6)
        logger.info(f"开始处理股票 {stock_code}")
        
        # 获取财务数据
        financial_data = get_stock_financial_data(stock_code)
        
        # 如果获取成功，保存数据
        if financial_data:
            return save_financial_data(stock_code, financial_data, output_dir)
        else:
            logger.warning(f"获取股票 {stock_code} 的财务数据失败")
            return False
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 时出错: {str(e)}")
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
        logger.info(f"创建输出目录: {output_dir}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 确保code列存在
        if 'code' not in df.columns:
            logger.error("CSV文件中没有'code'列")
            return
        
        # 提取唯一的股票代码
        df['code'] = df['code'].astype(str).str.zfill(6)
        unique_codes = df['code'].unique()
        logger.info(f"从CSV文件中提取了 {len(unique_codes)} 个唯一股票代码")
        
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
        
        logger.info(f"使用 {max_workers} 个线程开始处理...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_with_counter, code) for code in unique_codes]
            concurrent.futures.wait(futures)
        
        logger.info(f"处理完成! 成功: {success_count}, 失败: {fail_count}, 总计: {len(unique_codes)}")
        
    except Exception as e:
        logger.error(f"批量处理股票时出错: {str(e)}")

def main():
    # 设置文件路径
    csv_file = "E:\\20250305-bi\\pattern_results.csv"
    output_dir = "E:\\20250305-bi\\stock_info"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        logger.error(f"CSV文件不存在: {csv_file}")
        return
    
    # 批量处理股票
    batch_process_stocks_from_csv(csv_file, output_dir, max_workers=5)

if __name__ == "__main__":
    main()