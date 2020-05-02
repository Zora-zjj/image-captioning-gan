import os


class FilePathManager:
    base_dir = os.path.dirname(os.path.abspath(__file__))  #os.path.dirname  去掉文件名，返回首目录    os.path.abspath  返回绝对路径

    @staticmethod        # @staticmethod    修饰器，静态方法 
    def resolve(path):   #拼接两个路径
        return f"{FilePathManager.base_dir}/{path}"   #python格式化字符串f，例如f'Hello, my name is {name}'，将name的具体值代入
