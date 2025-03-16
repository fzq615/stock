import os
import sys
from datetime import datetime

# 确保能够导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_info import get_stock_fundamental_info, batch_get_stock_info

def test_single_stock(stock_code):
    """测试获取单个股票的基本面信息"""
    print(f"\n正在测试获取股票 {stock_code} 的基本面信息...")
    
    # 获取基本面信息
    info = get_stock_fundamental_info(stock_code)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_stock_info_{stock_code}_{timestamp}.txt"
    
    # 保存为TXT文件
    with open(filename, 'w', encoding='utf-8') as f:
        # 写入基本信息
        f.write(f"股票代码：{info['股票代码']}\n")
        f.write(f"股票名称：{info.get('股票名称', info['股票代码'])}\n")
        f.write(f"获取时间：{info['获取时间']}\n\n")
        
        # 写入东方财富数据
        f.write("=== 东方财富数据 ===\n")
        if info['东方财富数据']:
            for key, value in info['东方财富数据'].items():
                # 将数值转换为字符串
                value_str = str(value)
                f.write(f"{key}：{value_str}\n")
        else:
            f.write("无数据\n")
        f.write("\n")
        
        # 写入特殊题材信息
        f.write("=== 特殊题材信息 ===\n")
        if info['特殊题材信息']:
            special_info = info['特殊题材信息']
            
            # 写入所属板块
            f.write("所属板块：\n")
            if special_info['所属板块']:
                for board in special_info['所属板块']:
                    f.write(f"- {board}\n")
            else:
                f.write("无数据\n")
            f.write("\n")
            
            # 写入其他基本信息
            for key in ['经营范围', '主营业务', '行业背景']:
                f.write(f"{key}：\n")
                content = special_info[key].strip() if special_info[key] else "无数据"
                f.write(f"{content}\n\n")
            
            # 写入核心竞争力
            f.write("核心竞争力：\n")
            if special_info['核心竞争力']:
                for item in special_info['核心竞争力']:
                    f.write(f"- {item['标题']}\n")
                    f.write(f"  {item['内容']}\n")
            else:
                f.write("无数据\n")
        else:
            f.write("无数据\n")
        f.write("\n")
        
        # 写入新闻信息
        f.write("=== 新闻列表 ===\n")
        if info['新闻信息']:
            for i, news in enumerate(info['新闻信息'], 1):
                f.write(f"\n新闻 {i}\n")
                f.write(f"标题：{news['标题']}\n")
                f.write(f"发布时间：{news['发布时间']}\n")
                f.write(f"链接：{news['链接']}\n")
                f.write(f"内容：{news['内容']}\n")
                f.write("\n" + "-"*50 + "\n")  # 分隔线
        else:
            f.write("无数据\n")
    
    print(f"测试完成，结果已保存到文件: {filename}")
    print(f"获取到的新闻数量: {len(info.get('新闻信息', []))}")
    return info

def test_batch_stocks(stock_codes):
    """测试批量获取多个股票的基本面信息"""
    print(f"\n正在测试批量获取 {len(stock_codes)} 个股票的基本面信息...")
    
    for code in stock_codes:
        test_single_stock(code)

if __name__ == "__main__":
    # 测试单个股票
    test_single_stock('000030')
    
    # 测试多个股票
    # test_batch_stocks(['000001', '000002', '000030'])
    
    print("\n所有测试完成！")