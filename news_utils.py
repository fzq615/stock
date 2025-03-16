import requests
import pandas as pd
import json
from datetime import datetime
from urllib.parse import quote
from bs4 import BeautifulSoup

def get_eastmoney_search_news(keyword):
    """
    获取东方财富搜索页面的新闻
    keyword: 搜索关键词，如 "恒立钻具"
    """
    print(f"开始获取 '{keyword}' 的相关新闻...")
    
    # 尝试使用股票代码直接获取新闻
    if keyword.isdigit() and len(keyword) == 6:
        print(f"检测到输入为股票代码，尝试获取股票 {keyword} 的相关新闻")
        try:
            # 尝试获取股票名称
            import akshare as ak
            try:
                stock_info = ak.stock_zh_a_spot_em()
                stock_name = stock_info[stock_info['代码'] == keyword]['名称'].values[0]
                print(f"获取到股票名称: {stock_name}，将使用此名称搜索新闻")
                keyword = stock_name  # 使用股票名称搜索
            except Exception as e:
                print(f"无法获取股票名称，将继续使用股票代码搜索: {str(e)}")
        except ImportError:
            print("未安装akshare，将继续使用股票代码搜索")
    
    # 构建API URL
    url = "http://search-api-web.eastmoney.com/search/jsonp"
    
    # URL编码处理
    encoded_keyword = quote(keyword)
    json_param = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "all",  # 修改为搜索全部内容，不仅限于标题
                "sort": "time",        # 按时间排序
                "pageIndex": 1,
                "pageSize": 100,
                "preTag": "<em>",
                "postTag": "</em>"
            }
        }
    }
    
    params = {
        "cb": "jQuery",
        "param": json.dumps(json_param, ensure_ascii=False),
        "_": str(int(datetime.now().timestamp() * 1000))
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': f'https://so.eastmoney.com/news/s?keyword={encoded_keyword}&type=title',
        'Accept': '*/*'
    }
    
    try:
        print(f"发送请求到东方财富搜索API: {url}")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.encoding = 'utf-8'  # 设置响应编码
        
        # 检查响应状态
        if response.status_code != 200:
            print(f"API请求失败，状态码: {response.status_code}")
            return pd.DataFrame()
            
        # 处理返回的jsonp数据
        data = response.text
        print(f"获取到响应数据，长度: {len(data)} 字符")
        
        # 检查响应格式
        if not data.startswith('jQuery(') or not data.endswith(')'):
            print(f"响应格式不正确: {data[:100]}...")
            # 尝试直接解析JSON
            try:
                json_data = json.loads(data)
            except:
                print("无法解析响应数据为JSON")
                return pd.DataFrame()
        else:
            # 正常解析JSONP
            json_str = data.strip('jQuery(')[:-1]
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {str(e)}")
                return pd.DataFrame()
        
        # 提取新闻列表
        news_list = []
        if 'result' in json_data and 'cmsArticleWebOld' in json_data['result']:
            news_list = json_data['result']['cmsArticleWebOld']
            print(f"从API响应中提取到 {len(news_list)} 条新闻")
        else:
            print("API响应中没有找到新闻列表")
            if 'result' in json_data:
                print(f"可用的结果类型: {list(json_data['result'].keys()) if isinstance(json_data['result'], dict) else '无'}")
        
        # 转换为DataFrame
        news_data = []
        for news in news_list:
            # 完全放宽筛选条件，接受所有返回的新闻
            news_title = news['title'].replace('<em>', '').replace('</em>', '')
            news_content = news['content'].replace('<em>', '').replace('</em>', '') if 'content' in news else ''
            
            # 直接添加所有新闻，不做关键词筛选
            news_data.append({
                '新闻标题': news_title,
                '新闻内容': news_content,
                '发布时间': news.get('date', ''),
                '文章来源': news.get('mediaName', ''),
                '新闻链接': news.get('url', '')
            })
            
            # 打印调试信息
            print(f"添加新闻: {news_title[:30]}...")
        
        df = pd.DataFrame(news_data)
        print(f"\n共获取到 {len(df)} 条相关新闻")
        
        # 确保DataFrame至少包含必要的列
        if df.empty:
            print("未找到相关新闻，尝试使用更宽松的搜索条件")
            
            # 如果没有找到新闻，尝试使用更宽松的搜索条件再次搜索
            # 例如，如果关键词是公司全名，尝试使用公司简称
            if len(keyword) > 4 and not keyword.isdigit():
                short_keyword = keyword[:2]  # 使用前两个字作为简称
                print(f"尝试使用简称 '{short_keyword}' 重新搜索")
                
                # 递归调用自身，但使用简称
                return get_eastmoney_search_news(short_keyword)
            else:
                # 创建空DataFrame
                df = pd.DataFrame(columns=['新闻标题', '新闻内容', '发布时间', '文章来源', '新闻链接'])
                print("未找到相关新闻，返回空DataFrame")
            
        return df
        
    except requests.exceptions.Timeout:
        print("请求超时，东方财富搜索API无响应")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"获取新闻列表失败: {str(e)}")
        print(f"请求URL: {url}")
        print(f"请求参数: {params}")
        return pd.DataFrame()

def get_full_content(url):
    """
    获取新闻页面的完整内容
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尝试不同的内容定位方式
        content = ''
        
        # 方式1：查找指定ID的div
        content_div = soup.find('div', {'id': 'ContentBody'}) or \
                     soup.find('div', {'id': 'content'}) or \
                     soup.find('div', {'class': 'newsContent'})
        
        if content_div:
            # 获取所有段落
            paragraphs = content_div.find_all('p')
            content = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # 如果没有找到内容，尝试其他方式
        if not content:
            # 方式2：查找文章主体
            article = soup.find('article') or \
                     soup.find('div', {'class': 'article'})
            if article:
                content = article.get_text().strip()

        return content or '无法获取完整内容'
    except Exception as e:
        print(f"获取新闻内容失败: {str(e)}")
        return '获取内容时出错'