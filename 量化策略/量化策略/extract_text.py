from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time

def get_html_content(url):
    """
    使用Selenium获取动态加载的HTML内容
    
    参数:
    url (str): 网页URL
    
    返回:
    str: 网页的HTML内容
    """
    # 设置Chrome选项
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 无头模式
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # 初始化浏览器
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    
    try:
        # 打开网页
        driver.get(url)
        
        # 等待页面加载完成（可以根据实际情况调整等待时间）
        time.sleep(5)  # 简单的等待，也可以使用显式等待
        
        # 获取页面源码
        html_content = driver.page_source
        
        return html_content
    finally:
        # 关闭浏览器
        driver.quit()

def extract_stock_info(html_content):
    """
    从HTML内容中提取股票信息
    
    参数:
    html_content (str): HTML内容字符串
    
    返回:
    dict: 包含股票信息的字典
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    info = {}
    
    # 提取所属板块
    info['所属板块'] = []
    for li in soup.select('.board .boardName'):
        info['所属板块'].append(li.get_text(strip=True))
    
    # 提取经营范围
    scope = soup.select_one('.p_div:has(p.classfiy:-soup-contains("经营范围")) span')
    if scope:
        info['经营范围'] = scope.get_text(strip=True)
    
    # 提取主营业务
    main_business_div = soup.select_one('.p_div:has(p.classfiy:-soup-contains("主营业务"))')
    if main_business_div:
        # 提取整个div内容，移除标签文本
        main_business_parts = []
        for sibling in main_business_div.next_siblings:
            if sibling.name == 'div':
                break
            if isinstance(sibling, str) or (sibling.name == 'span' and sibling.text.strip()):
                main_business_parts.append(sibling.get_text(strip=True))
        
        # 合并所有部分，并清理多余空格
        info['主营业务'] = ' '.join(main_business_parts).replace('主营业务', '').strip()
        info['主营业务'] = ' '.join(info['主营业务'].split())
    
    # 提取行业背景
    industry_background_div = soup.select_one('.p_div:has(p.classfiy:-soup-contains("行业背景"))')
    if industry_background_div:
        # 提取整个div内容，移除标签文本
        industry_background_parts = []
        for sibling in industry_background_div.next_siblings:
            if sibling.name == 'div':
                break
            if isinstance(sibling, str) or (sibling.name == 'span' and sibling.text.strip()):
                industry_background_parts.append(sibling.get_text(strip=True))
        
        # 合并所有部分，并清理多余空格
        info['行业背景'] = ' '.join(industry_background_parts).replace('行业背景', '').strip()
        info['行业背景'] = ' '.join(info['行业背景'].split())
    
    # 提取核心框架
    core_framework = []
    for item in soup.select('.p_div span'):
        core_framework.append(item.get_text(strip=True))
    info['核心框架'] = core_framework
    
    return info

# 使用示例
if __name__ == "__main__":
    url = "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code=SZ000759&color=b#/hxtc"
    
    # 获取动态加载的HTML内容
    html_content = get_html_content(url)
    
    # 提取股票信息
    stock_info = extract_stock_info(html_content)
    
    # 打印提取的信息
    for key, value in stock_info.items():
        print(f"{key}: {value}")