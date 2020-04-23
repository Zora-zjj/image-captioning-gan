import os


class FilePathManager:
    base_dir = os.path.dirname(os.path.abspath(__file__))  #os.path.dirname  去掉文件名，返回首目录    os.path.abspath  返回绝对路径

    @staticmethod        # @staticmethod    修饰器，静态方法 
    def resolve(path):
        return f"{FilePathManager.base_dir}/{path}"
