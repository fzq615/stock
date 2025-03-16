from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def extract_web_content(url):
    """
    从指定网页提取内容
    
    参数:
    url (str): 网页URL
    
    返回:
    str: 提取的网页内容
    """
    driver = None
    try:
        # 设置Chrome选项
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # 无头模式
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        
        # 初始化WebDriver
        service = Service(executable_path='chromedriver.exe')
        driver = webdriver.Chrome(service=service, options=options)
        
        # 访问网页
        driver.get(url)
        
        # 等待页面加载完成
        time.sleep(5)  # 等待5秒确保页面完全加载
        
        # 查找目标元素
        content_div = driver.find_element(By.CSS_SELECTOR, '#app > div:nth-child(4) > div > div.section.tcxq > div.hxtccontent')
        
        if content_div:
            return content_div.text
        else:
            return "未找到指定内容"
            
    except Exception as e:
        return f"请求失败: {str(e)}"
    finally:
        if driver is not None:
            driver.quit()

# 使用示例
if __name__ == "__main__":
    url = "https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?type=web&code=SZ000759&color=b#/hxtc"
    content = extract_web_content(url)
    print(content)
