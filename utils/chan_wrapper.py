import importlib.util
import sys
import os
import logging

logger = logging.getLogger('ChanWrapper')

def load_chan_module(compiled_dir="..", module_name="chan_fs_jiasu_compiled_unique"):
    """
    加载编译后的缠论模块
    
    参数:
        compiled_dir: 编译文件目录
        module_name: 模块名称
    
    返回:
        加载的模块对象
    """
    try:
        # 将编译文件路径添加到搜索路径最前面
        # 获取绝对路径
        compiled_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), compiled_dir))
        
        # 将编译文件路径添加到搜索路径最前面
        if compiled_dir not in sys.path:
            sys.path.insert(0, compiled_dir)
            logger.info(f"已将 {compiled_dir} 添加到搜索路径最前面")
            
        
        # 确保模块未被加载
        modules_to_remove = [
            'chan_fs_jiasu',
            'chan_fs_jiasu_compiled',
            'chan_fs_jiasu_compiled_unique'
        ]
        
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # 导入编译后的模块
        compiled_file_path = os.path.join(compiled_dir, f"{module_name}.pyc")
        
        # 移除可能存在的同名模块
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        spec = importlib.util.spec_from_file_location(module_name, compiled_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # 验证导入的模块路径
        logger.info(f"模块文件路径: {module.__file__}")
        
        # 注入辅助函数
        def log_info(*args):
            logger.info(*args)
        module.LogInfo = log_info
        
        logger.info("缠论模块导入成功")
        return module
        
    except Exception as e:
        logger.error(f"导入缠论模块时出错: {str(e)}")
        raise

def get_chan_functions(module):
    """
    从加载的模块中获取所需的缠论函数
    
    参数:
        module: 已加载的缠论模块
    
    返回:
        包含所需函数的字典
    """
    try:
        functions = {
            'cal_fenbi': getattr(module, 'cal_fenbi'),
            'xd_js': getattr(module, 'xd_js'),
            'kxian_baohan_js_0': getattr(module, 'kxian_baohan_js_0'),
            'fenbi_js': getattr(module, 'fenbi_js'),
            'repeat_bi_js': getattr(module, 'repeat_bi_js'),
            'xianDuan_js': getattr(module, 'xianDuan_js'),
            'Xian_Duan': getattr(module, 'Xian_Duan')
        }
        
        # 验证函数来源
        logger.info(f"Xian_Duan函数的来源: {functions['Xian_Duan'].__module__}")
        
        return functions
    except AttributeError as e:
        logger.error(f"获取缠论函数时出错: {str(e)}")
        raise