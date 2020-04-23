from abc import abstractmethod, ABCMeta


class BaseExtractor(metaclass=ABCMeta):          # metaclass=ABCMeta 多态性指具有不同功能的函数可以使用相同的函数名
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

    @abstractmethod
    def extract(self, image):
        raise NotImplementedError()

    def __call__(self, image):
        return self.forward(image)
