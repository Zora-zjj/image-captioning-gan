import gensim

from gensim.models.wrappers.fasttext import FastText

from file_path_manager import FilePathManager

if __name__ == '__main__':
    model = FastText.load_fasttext_format(FilePathManager.resolve("data/wiki.en"))    # load_fasttext_format ：由原始Fasttext实施兼容的格式加载模型，一般存储/经由其加载save()和 load()方法，(cap_path, full_model=False)
    model.save(FilePathManager.resolve("data/fasttext.model"))
