import akshare as ak
import pandas as pd
import os
from datetime import datetime
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_stock_financial_statements(stock_code):
    """
    获取股票的财务报表数据，包括业绩报表、业绩快报、业绩预告和利润表
    
    参数:
    stock_code (str): 股票代码
    
    返回:
    dict: 包含各类财务数据的字典
    """
    result = {
        "获取时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "股票代码": stock_code,
        "业绩报表": None,
        "业绩快报": None,
        "业绩预告": None,
        "利润表": None,
        "错误信息": []
    }
    
    try:
        # 确保股票代码格式正确
        stock_code = stock_code.strip().zfill(6)
        
        # 1. 获取业绩报表数据
        try:
            logger.info(f"获取股票 {stock_code} 的业绩报表数据")
            yjbb_data = ak.stock_yjbb_em(date="20241231")
            if not yjbb_data.empty:
                result["业绩报表"] = yjbb_data.to_dict('records')
                logger.info(f"成功获取业绩报表数据，共 {len(yjbb_data)} 条记录")
        except Exception as e:
            error_msg = f"获取业绩报表数据失败: {str(e)}"
            logger.error(error_msg)
            result["错误信息"].append(error_msg)
        
        # 2. 获取业绩快报数据
        try:
            logger.info(f"获取股票 {stock_code} 的业绩快报数据")
            yjkb_data = ak.stock_yjkb_em(date="20241231")
            if not yjkb_data.empty:
                result["业绩快报"] = yjkb_data.to_dict('records')
                logger.info(f"成功获取业绩快报数据，共 {len(yjkb_data)} 条记录")
        except Exception as e:
            error_msg = f"获取业绩快报数据失败: {str(e)}"
            logger.error(error_msg)
            result["错误信息"].append(error_msg)
        
        # 3. 获取业绩预告数据
        try:
            logger.info(f"获取股票 {stock_code} 的业绩预告数据")
            yjyg_data = ak.stock_yjyg_em(date="20241231")
            if not yjyg_data.empty:
                result["业绩预告"] = yjyg_data.to_dict('records')
                logger.info(f"成功获取业绩预告数据，共 {len(yjyg_data)} 条记录")
        except Exception as e:
            error_msg = f"获取业绩预告数据失败: {str(e)}"
            logger.error(error_msg)
            result["错误信息"].append(error_msg)
        
        # 4. 获取利润表数据
        try:
            logger.info(f"获取股票 {stock_code} 的利润表数据")
            lrb_data = ak.stock_lrb_em(date="20241231")
            if not lrb_data.empty:
                result["利润表"] = lrb_data.to_dict('records')
                logger.info(f"成功获取利润表数据，共 {len(lrb_data)} 条记录")
        except Exception as e:
            error_msg = f"获取利润表数据失败: {str(e)}"
            logger.error(error_msg)
            result["错误信息"].append(error_msg)
        
        # 如果所有数据都获取失败
        if all(v is None for k, v in result.items() if k not in ["获取时间", "股票代码", "错误信息"]):
            logger.warning(f"股票 {stock_code} 的所有财务数据获取失败")
        
        return result
        
    except Exception as e:
        error_msg = f"获取财务数据时发生错误: {str(e)}"
        logger.error(error_msg)
        result["错误信息"].append(error_msg)
        return result

def save_financial_statements_csv(stock_code, financial_data, output_dir):
    """
    将财务数据保存为CSV格式
    
    参数:
    stock_code (str): 股票代码
    financial_data (dict): 财务数据字典
    output_dir (str): 输出目录
    
    返回:
    bool: 是否成功保存
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 添加点击事件处理
        def on_row_click(event):
            # 处理点击事件
            pass
        # 设置文本框布局
        text_box = {
            'top': {'height': '30%', 'content': 'financial_业绩预告.csv'},
            'bottom': {'height': '70%', 'content': 'stock_info'}
        }
        
        # 处理各类财务数据
        for data_type in ['业绩报表', '业绩快报', '业绩预告', '利润表']:
            if financial_data[data_type]:
                # 创建DataFrame
                df = pd.DataFrame(financial_data[data_type])
                
                # 添加基本信息列
                #df['股票代码'] = stock_code
                df['获取时间'] = financial_data['获取时间']
                
                # 创建文件名
                filename = f"financial_{data_type}_{stock_code}_{timestamp}.csv"
                filepath = os.path.join(output_dir, filename)
                
                # 保存为CSV文件
                df.to_csv(filepath, encoding='utf-8-sig', index=False)
                logger.info(f"成功保存{data_type}数据到文件: {filepath}")
        
        # 如果有错误信息，保存到单独的文件
        if financial_data['错误信息']:
            error_df = pd.DataFrame({
                '股票代码': [stock_code],
                '获取时间': [financial_data['获取时间']],
                '错误信息': [str(financial_data['错误信息'])]
            })
            error_filepath = os.path.join(output_dir, f"financial_errors_{stock_code}_{timestamp}.csv")
            error_df.to_csv(error_filepath, encoding='utf-8-sig', index=False)
            logger.info(f"保存错误信息到文件: {error_filepath}")
        
        return True
        
    except Exception as e:
        logger.error(f"保存财务数据失败: {str(e)}")
        return False

def process_stock(stock_code, output_dir):
    """
    处理单个股票的财务数据获取和保存
    
    参数:
    stock_code (str): 股票代码
    output_dir (str): 输出目录
    
    返回:
    bool: 是否成功处理
    """
    try:
        # 获取财务数据
        financial_data = get_stock_financial_statements(stock_code)
        
        # 保存数据
        if financial_data:
            return save_financial_statements_csv(stock_code, financial_data, output_dir)
        return False
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 时出错: {str(e)}")
        return False

def main():
    # 测试单个股票
    stock_code = "000001"  # 以平安银行为例
    output_dir = "E:\\20250305-bi\\stock_info"
    
    if process_stock(stock_code, output_dir):
        logger.info(f"成功处理股票 {stock_code} 的财务数据")
    else:
        logger.error(f"处理股票 {stock_code} 的财务数据失败")

if __name__ == "__main__":
    main()