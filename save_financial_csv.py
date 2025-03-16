import pandas as pd
import os
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_financial_to_csv(stock_data, output_dir):
    """
    将股票财务数据保存为CSV格式
    
    参数:
    stock_data (dict): 股票财务数据字典
    output_dir (str): 输出目录
    
    返回:
    bool: 是否成功保存
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建DataFrame
        df = pd.DataFrame([{
            '序号': stock_data.get('序号'),
            '股票代码': stock_data.get('股票代码'),
            '股票简称': stock_data.get('股票简称'),
            '每股收益': stock_data.get('每股收益'),
            '营业收入-营业收入': stock_data.get('营业收入-营业收入'),
            '营业收入-同比增长': stock_data.get('营业收入-同比增长'),
            '营业收入-季度环比增长': stock_data.get('营业收入-季度环比增长'),
            '净利润-净利润': stock_data.get('净利润-净利润'),
            '净利润-同比增长': stock_data.get('净利润-同比增长'),
            '净利润-季度环比增长': stock_data.get('净利润-季度环比增长'),
            '每股净资产': stock_data.get('每股净资产'),
            '净资产收益率': stock_data.get('净资产收益率'),
            '每股经营现金流量': stock_data.get('每股经营现金流量'),
            '销售毛利率': stock_data.get('销售毛利率'),
            '所处行业': stock_data.get('所处行业'),
            '最新公告日期': stock_data.get('最新公告日期')
        }])
        
        # 创建文件名
        filename = f"financial_data_{stock_data.get('股票代码')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # 保存为CSV文件
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"成功将股票 {stock_data.get('股票代码')} 的财务数据保存到: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"保存财务数据到CSV失败: {str(e)}")
        return False

def main():
    # 测试数据
    test_data = {
        '序号': 1,
        '股票代码': '300437',
        '股票简称': '清水源',
        '每股收益': 0.0282,
        '营业收入-营业收入': 268824274.64,
        '营业收入-同比增长': -49.2638624588,
        '营业收入-季度环比增长': -23.8991,
        '净利润-净利润': 7301682.85,
        '净利润-同比增长': -90.42,
        '净利润-季度环比增长': 105.4917,
        '每股净资产': 6.214034417183,
        '净资产收益率': 0.46,
        '每股经营现金流量': 0.069699201156,
        '销售毛利率': 12.8683094844,
        '所处行业': '化学制品',
        '最新公告日期': '2025-02-28'
    }
    
    output_dir = "E:\\20250305-bi\\stock_info"
    if save_financial_to_csv(test_data, output_dir):
        logger.info("测试成功")
    else:
        logger.error("测试失败")

if __name__ == "__main__":
    main()