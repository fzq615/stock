import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os
import re
from enhanced_financial_data import get_enhanced_financial_data, save_enhanced_financial_data

class QwenChat:
    def __init__(self, model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", cache_dir="D:/models_cache", device="cuda"):
        """
        初始化Qwen聊天模型
        
        参数:
        model_path: 模型路径或huggingface模型名称
        cache_dir: 模型缓存目录
        device: 使用的设备 ("cuda" 或 "cpu")
        """
        self.device = device
        print(f"正在加载模型 {model_path}...")
        
        # 判断是否为本地路径
        is_local_path = os.path.exists(model_path)
        
        # 设置模型加载配置
        if is_local_path:
            # 如果是本地路径，直接使用local_files_only=True
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            # 如果是Hugging Face模型ID，使用cache_dir
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
        
        # 使用float16精度加载模型以减少显存占用
        if is_local_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",  # 自动处理模型部署
                trust_remote_code=True,
                torch_dtype=torch.int8,
                local_files_only=True
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                cache_dir=cache_dir
            ).eval()

        # 设置生成配置
        if is_local_path:
            self.model.generation_config = GenerationConfig.from_pretrained(
                model_path, 
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            self.model.generation_config = GenerationConfig.from_pretrained(
                model_path, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
        
        print("模型加载完成！")

    def chat(self, prompt, max_length=2048, temperature=0.7, top_p=0.9):
        """
        与模型进行对话
        
        参数:
        prompt: 输入的提示文本
        max_length: 生成文本的最大长度
        temperature: 生成的随机性（0-1之间）
        top_p: 核采样的概率阈值
        
        返回:
        生成的回复文本
        """
        try:
            # 准备输入
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=1024  # 限制输入长度
            ).to(self.device)
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 生成回复
            with torch.no_grad():  # 不计算梯度
                response = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            # 解码回复
            response_text = self.tokenizer.decode(
                response[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            # 提取模型的回复部分
            return response_text.split(prompt)[-1].strip()
            
        except Exception as e:
            print(f"生成回复时出错: {str(e)}")
            return "抱歉，处理您的请求时出现错误。请尝试输入更短的文本或重新启动程序。"
        except KeyboardInterrupt:
            print("\n生成被用户中断")
            return "生成被中断，您可以继续输入新的问题。"

    def chat_loop(self):
        """
        启动交互式聊天循环
        """
        print("\n开始聊天！(输入 'quit' 或 'exit' 结束对话)")
        
        while True:
            # 获取用户输入
            user_input = input("\n你: ").strip()
            
            # 检查是否退出
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            
            # 如果输入为空，继续下一轮
            if not user_input:
                continue
            
            # 获取模型回复
            response = self.chat(user_input)
            
            # 打印回复
            if response:
                print("\nQwen: " + response)
            else:
                print("\nQwen: 抱歉，我现在无法生成回复。")

def main():
    """
    主函数
    """
    # 检查是否有可用的GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if device == "cpu":
        print("警告：未检测到GPU，模型运行可能会很慢！")
    
    # 创建模型实例
    chat_bot = QwenChat(device=device)
    
    # 启动聊天循环
    chat_bot.chat_loop()

if __name__ == "__main__":
    main()