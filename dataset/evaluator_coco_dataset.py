import pickle
import random

import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.corpus import Corpus
from file_path_manager import FilePathManager


class EvaluatorCocoDataset(Dataset):

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                          annFile=FilePathManager.resolve(
                                              f"data/annotations/captions_train2017.json"),      # 2017 train 训练集
                                          transform=transforms.ToTensor())
        with open(FilePathManager.resolve("data/embedded_images.pkl"), "rb") as f:
            self.images = pickle.load(f)     #embedding_images
        self.length = len(self.images) * 5   #乘5 , caption数量

    def __getitem__(self, index):  #caption 的 index
        temp = index // 5          #图片id
        image = self.images[temp]
        image = image.view(-1)
        item = self.get_captions(temp)
        caption = item[index % 5]
        caption = self.corpus.embed_sentence(caption, one_hot=False)   #embed的caption
        s = set(range(self.length // 5))   #图片id的范围
        s.remove(temp)                     #list.remove(obj)：移除列表中某个值的第一个匹配项
        s = list(s)
        other_index = random.choice(s)     #取得非该图片的真实caption
        other_caption = self.get_captions(other_index)
        other_index = random.choice(range(5))
        other_caption = other_caption[other_index]
        other_caption = self.corpus.embed_sentence(other_caption, one_hot=False)
        return image, caption, other_caption    #非该图片的真实caption

    def get_captions(self, index):
        coco = self.captions.coco
        img_id = self.captions.ids[index]   #图片id
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)       #图片对应的caption
        target = [ann['caption'] for ann in anns]   #index图片的5个caption组成target
        return target

    def __len__(self):
        return self.length
