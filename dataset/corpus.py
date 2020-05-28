import pickle
import string
from collections import defaultdict

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from gensim.models.wrappers.fasttext import FastText

from file_path_manager import FilePathManager


class Corpus:
    START_SYMBOL = "<start>"   #加单书名号？？？
    END_SYMBOL = "<end>"
    UNK = "<unk>"
    PAD = "<pad>"

    def __init__(self, word2idx=None, idx2word=None, word_embeddings=None):
        self.word2idx = word2idx if word2idx is not None else {}
        self.idx2word = idx2word if idx2word is not None else {}
        self.fast_text = word_embeddings if word_embeddings is not None else {}   #此word_embeddings不是下面函数word_embeddings
        self.vocab_size = len(self.word2idx)
        self.embed_size = 300
        self.max_sentence_length = 18
        self.min_word_freq = 5
    #中间
    def word_embedding(self, word):           #将word转成embedding，result = [embed_size]
        if word not in self.word2idx and word not in self.idx2word:
            word = self.UNK
        if isinstance(word, int):             #当word是idx
            word = self.word_from_index(word)
        result = torch.from_numpy(self.fast_text[word]).view(-1)
        return result
    #中间
    def word_embeddings(self, words):       #批量将words转成embeddings向量, result = [len(words）,embed_size]
        temp = len(words)
        result = torch.zeros(temp, self.embed_size)
        for i in range(temp):
            result[i] = self.word_embedding(words[i])
        return result
    #中间
    def word_embedding_from_index(self, index):    #对指定index的word进行embedding
        return self.word_embedding(self.word_from_index(index))
    #中间
    def word_embeddings_from_indices(self, indices):   #对指定indices的words进行embeddings
        words = [self.word_from_index(i) for i in indices]
        return self.word_embeddings(words)
    #中间
    def word_one_hot(self, word):    #对某个word进行one-hot，列向量
        if word not in self.word2idx and word not in self.idx2word:
            word = self.UNK
        result = torch.zeros(self.vocab_size).view(-1)    #[vocab_size]一维张量
        if isinstance(word, str):
            word = self.word_index(word)
        result[word] = 1
        return result.long()
    #中间
    def word_index(self, word):    #由word得index
        if word not in self.word2idx and word not in self.idx2word:
            word = self.UNK
        return self.word2idx[word]
    #中间
    def word_from_index(self, index):   #由index得word
        return self.idx2word[index]
    #中间
    def words_from_indices(self, indices):   #由indices得words
        return [self.word_from_index(index) for index in indices]

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word
            self.vocab_size += 1
            
    #将cococaption的词加入词典
    def prepare(self):                       # word2idx：单词-计数
        self.word2idx = defaultdict(int)     #defaultdict(int)，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
        # to make sure start_symbol, end_symbol, pad, and unk will be included
        self.word2idx[self.START_SYMBOL] = self.word2idx[self.END_SYMBOL] = self.word2idx[self.UNK] = self.word2idx[    #将这4个词的id都设为5？？？应该是频率5吧？？
            self.PAD] = self.min_word_freq
        for dataset_type in ["train", "val"]:                                                #f表示格式化字符串,相似str.format()
            #class torchvision.datasets.CocoCaptions(root, annFile, transform=None, target_transform=None)
            caps = dset.CocoCaptions(root=FilePathManager.resolve(f'data/{dataset_type}'),    #data固定词
                                     annFile=FilePathManager.resolve(                 
                                         f"data/annotations/captions_{dataset_type}2017.json"),
                                     transform=transforms.ToTensor())        
            for _, captions in caps:
                for capt in captions:
                    tokens = self.tokenize(capt)    #toknize后面函数，分词
                    for token in tokens:
                        self.word2idx[token] += 1   #错误，应该是word2freq
        temp = {}  #word:索引，第几个词
        embeddings = {}
        fast_text = FastText.load(FilePathManager.resolve("data/fasttext.model"), mmap="r")  #????
        for k, v in self.word2idx.items():  # word,freq
            if v >= self.min_word_freq:
                temp[k] = len(temp)
                embeddings[k] = fast_text[k] if k in fast_text else fast_text[self.UNK]
        self.word2idx = temp
        # swap keys and values
        self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))
        self.fast_text = embeddings
    #中间
    @staticmethod                         #处理：移除
    def remove_nonalpha(word: str):       #函数参数中的冒号是参数的类型建议符，告诉程序员希望传入的实参的类型。函数后面跟着的箭头是函数返回值的类型建议符
        return word.strip().strip(".")    #strip()移除字符串头尾指定的字符（默认为空格或换行符）
    #中间
    @staticmethod
    def preprocess_sentence(sentence: str):  #处理：小写，替换
        sentence = sentence.lower().strip().strip(".").replace("'", "").replace(",,", ",").replace(",", " , ").replace(
            "\"", "")                        
        return sentence
    #中间
    def tokenize(self, sentence: str):  #分词
        temp = self.preprocess_sentence(sentence).split(" ")
        return [self.remove_nonalpha(x)
                for x in temp
                if not x.isspace() and x != "" and all(c in string.printable for c in x)]  #str.isspace()检测字符串是否只由空格组成
    #中间                                                         #string.printable :包含所有可打印字符的字符串,ASCII码中第33～126号是可打印字符
    def pad_sentence(self, tokens):    #填补句子长度
        tokens = tokens[:self.max_sentence_length]
        temp = len(tokens)
        if temp != self.max_sentence_length:
            tokens.extend([self.PAD] * (self.max_sentence_length - temp))
        return tokens
    #中间
    def embed_sentence(self, sentence: str, one_hot=False, pad: bool = True):   #对句子编码，one-hot或者embedding
        sentence = f"{self.START_SYMBOL} {sentence} {self.END_SYMBOL}"
        tokens = self.tokenize(sentence)
        if pad:
            tokens = self.pad_sentence(tokens)
        result = torch.zeros(self.max_sentence_length, self.vocab_size if one_hot else self.embed_size)
        for i in range(self.max_sentence_length):
            result[i] = self.word_one_hot(tokens[i]) if one_hot else self.word_embedding(tokens[i])
        return result    #[max_sentence_length, embed_size]
    #中间
    def sentence_indices(self, sentence):    #某句子的词indices
        sentence = f"{self.START_SYMBOL} {sentence} {self.END_SYMBOL}"
        tokens = self.tokenize(sentence)
        tokens = self.pad_sentence(tokens)
        return torch.LongTensor([self.word_index(token) for token in tokens])

    def __call__(self, sentence, one_hot: bool = False):  #对句子one-hot编码
        return self.embed_sentence(sentence, one_hot)

    def store(self, file_path):              #将Word2idx、idx2word、fast_text 保存
        with open(file_path, "wb") as f:
            pickle.dump((self.word2idx, self.idx2word, self.fast_text), f)    #pickle.dump(obj, file, [,protocol])序列化对象，将对象obj保存到文件file中去。

    @staticmethod
    def load(file_path):                    #将保存在的文件打开，与上面相反
        with open(file_path, "rb") as f:
            word2idx, idx2word, fast_text = pickle.load(f)  #pickle.load(file)反序列化对象，将文件中的数据解析为一个python对象。file中有read()接口和readline()接口
        return Corpus(word2idx, idx2word, fast_text)


if __name__ == '__main__':
    # corpus = Corpus()
    # corpus.prepare()
    # corpus.store(FilePathManager.resolve("data/corpus.pkl"))
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl")) 
    unknown = corpus.word_one_hot(corpus.UNK)
    print(corpus.vocab_size)
    print((corpus.word_one_hot(", ") == unknown).sum())
