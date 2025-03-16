import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model(cache_dir="D:/models_cache"):
    """
    下载 QWQ-32B 模型到指定目录
    
    参数:
    cache_dir: 模型缓存目录
    """
    # 设置环境变量
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    print(f"开始下载模型 {model_name}")
    print(f"缓存目录: {cache_dir}")
    
    try:
        # 首先下载tokenizer
        print("\n下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        print("Tokenizer下载完成！")
        
        # 然后下载模型
        print("\n下载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("模型下载完成！")
        
        print(f"\n所有文件已下载到: {cache_dir}")
        
    except Exception as e:
        print(f"下载过程中出错: {str(e)}")

if __name__ == "__main__":
    # 确保缓存目录存在
    cache_dir = "D:/models_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 开始下载
    download_model(cache_dir) 