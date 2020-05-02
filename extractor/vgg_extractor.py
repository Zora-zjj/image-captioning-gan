import cv2
import numpy as np
import pretrainedmodels.utils as utils
import torch.nn as nn
import torchvision.transforms as transforms
from dlt.util import cv2torch
from pretrainedmodels import vgg16
from torch.autograd import Variable

from extractor.base_extractor import BaseExtractor
from file_path_manager import FilePathManager


class VggExtractor(BaseExtractor):

    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)
        self.use_gpu = use_gpu
        self.cnn = vgg16()
        self.trans = utils.TransformImage(self.cnn)    #？？
        self.trans = transforms.Compose([transforms.ToPILImage(), self.trans])
        if use_gpu:
            self.cnn = self.cnn.cuda()
        self.cnn.eval()        #model.eval()，让model变成测试模式，对pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值 dropout和batch normalization的操作在训练和测试的时候是不一样的，
        for param in self.cnn.parameters():
            param.requires_grad = False

    def extract(self, image):      #image是图片路径
        if isinstance(image, str):
            image = cv2.imread(image)   #cv2.imread(filepath,flags)读入一副图片,filepath：要读入图片的完整路径,flags：读入图片的标志,默认彩色
        if isinstance(image, np.ndarray):
            image = cv2torch(image)   #？？？
        image = self.trans(image)     #先transform
        image = image.float()
        if len(image.size()) == 3:        #[c,h,w]
            image = image.unsqueeze(0)  #[b,c,h,w]
        if self.use_gpu:
            image = image.cuda()
        image = Variable(image)
        temp = self.cnn.features(image)     # cnn.features() ????
        return temp                         #提取到的特征


if __name__ == '__main__':
    extractor = VggExtractor(use_gpu=True)
    image_path = FilePathManager.resolve("test_images/image_1.png")
    print(extractor.extract(image_path))  
