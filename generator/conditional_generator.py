import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from dataset.corpus import Corpus
from extractor.vgg_extractor import VggExtractor
from file_path_manager import FilePathManager
from misc.beam_search import BeamSearch
from policy_gradient.rollout import Rollout


class ConditionalGenerator(nn.Module):

    def __init__(self,
                 corpus: Corpus,
                 mean: torch.FloatTensor = torch.zeros(1024),
                 std: torch.FloatTensor = torch.ones(1024),
                 cnn_output_size: int = 4096,
                 input_encoding_size: int = 512,
                 max_sentence_length: int = 18,
                 num_layers: int = 1,
                 dropout: float = 0):
        super().__init__()
        self.cnn_output_size = cnn_output_size
        self.input_encoding_size = input_encoding_size
        self.max_sentence_length = max_sentence_length
        self.embed = corpus
        self.dist = Normal(Variable(mean), Variable(std))  # noise variable # torch.distributions.Normal定义正态分布
        self.lstm = nn.LSTM(input_size=corpus.embed_size,
                            hidden_size=self.input_encoding_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        self.output_linear = nn.Linear(self.input_encoding_size, corpus.vocab_size)
        self.features_linear = nn.Sequential(
            nn.Linear(cnn_output_size + len(mean), input_encoding_size),
            nn.ReLU()
        )  
        self.rollout = Rollout(max_sentence_length, corpus)    

    def init_hidden(self, image_features):

        # generate rand
        rand = self.dist.sample((image_features.shape[0],)).cuda()   #.sample直接在定义的正态分布上采样，此处的sample是后面定义的函数

        # hidden of shape (num_layers * num_directions, batch, hidden_size)
        hidden = self.features_linear(torch.cat((image_features, rand), 1).unsqueeze(0))   # h0由图片特征和噪声综合

        # cell of shape (num_layers * num_directions, batch, hidden_size)
        cell = Variable(torch.zeros(image_features.shape[0], self.input_encoding_size).unsqueeze(0))  # c0值为0

        return hidden.cuda(), cell.cuda()

    def forward(self, features, captions):  # 由groundtruth生成caption
        states = self.init_hidden(features)
        hiddens, _ = self.lstm(captions, states)
        outputs = self.output_linear(hiddens[0])
        return outputs

    def reward_forward(self, image_features, evaluator, monte_carlo_count=16):
        self.lstm.flatten_parameters()      #.flatten_parameters 重置参数的数据指针，让数据存放成连续的块
        batch_size = image_features.size(0)
        hidden = self.init_hidden(image_features)
        # embed the start symbol
        inputs = self.embed.word_embeddings([self.embed.START_SYMBOL] * batch_size).unsqueeze(1).cuda()
        rewards = torch.zeros(batch_size, self.max_sentence_length)
        props = torch.zeros(batch_size, self.max_sentence_length)
        current_generated = inputs
        self.rollout.update(self)
        for i in range(self.max_sentence_length):
            _, hidden = self.lstm(inputs, hidden)
            outputs = self.output_linear(hidden[0]).squeeze(0)
            outputs = F.softmax(outputs, -1)
            predicted = outputs.multinomial(1)      #.multinomial(input,num_samples) 对每一行做num次取值，返回下标
            prop = torch.gather(outputs, 1, predicted)     #torch.gather(input,dim,index)在dim维度上对input取index的数据
            props[:, i] = prop.view(-1)
            # embed the next inputs, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
            inputs = self.embed.word_embeddings_from_indices(predicted.view(-1).cpu().data.numpy()).unsqueeze(1).cuda()
            current_generated = torch.cat([current_generated, inputs], dim=1)
            reward = self.rollout.reward(current_generated, image_features, hidden, monte_carlo_count, evaluator)
            rewards[:, i] = reward.view(-1)
        return rewards, props

    def sample(self, image_features, return_sentence=True):
        batch_size = image_features.size(0)

        # init the result with zeros and lstm states
        result = []
        hidden = self.init_hidden(image_features)

        # embed the start symbol
        # inputs = self.embed.word_embeddings(["car"] * batch_size).unsqueeze(1).cuda()
        inputs = self.embed.word_embeddings([self.embed.START_SYMBOL] * batch_size).unsqueeze(1).cuda()

        for i in range(self.max_sentence_length):
            inputs = Variable(inputs)
            _, hidden = self.lstm(inputs, hidden)
            outputs = self.output_linear(hidden[0]).squeeze(0)
            predicted = outputs.max(-1)[1]

            # embed the next inputs, unsqueeze is required 'cause of shape (batch_size, 1, embedding_size)
            inputs = self.embed.word_embeddings_from_indices(predicted.cpu().data.numpy()).unsqueeze(1).cuda()

            # store the result
            result.append(self.embed.word_from_index(predicted.cpu().numpy()[0]))

        if return_sentence:
            result = " ".join(
                list(filter(lambda x: x not in [self.embed.END_SYMBOL, self.embed.UNK, self.embed.PAD], result)))

        return result

    def sample_single_with_embedding(self, image_features):
        batch_size = image_features.size(0)

        # init the result with zeros, and lstm states
        result = torch.zeros(self.max_sentence_length, self.embed.embed_size)
        hidden = self.init_hidden(image_features)

        inputs = self.embed.word_embeddings([self.embed.START_SYMBOL] * batch_size).unsqueeze(1).cuda()

        for i in range(self.max_sentence_length):
            result[i] = inputs.squeeze(1)
            _, hidden = self.lstm(inputs, hidden)
            outputs = self.output_linear(hidden[0]).squeeze(0)
            predicted = outputs.max(-1)[1]

            # embed the next inputs, unsqueeze is required 'cause of shape (batch_size, 1, embedding_size)
            inputs = self.embed.word_embeddings_from_indices(predicted.cpu().data.numpy()).unsqueeze(1).cuda()

        return result

    def beam_sample(self, image_features, beam_size=5):
        # self.beam_size = 5
        batch_size = image_features.size(0)
        beam_searcher = BeamSearch(beam_size, 1, 17)

        # init the result with zeros and lstm states
        states = self.init_hidden(image_features)
        states = (states[0].repeat(1, beam_size, 1).cuda(), states[1].repeat(1, beam_size, 1).cuda())

        # embed the start symbol
        words_feed = self.embed.word_embeddings([self.embed.START_SYMBOL] * batch_size) \
            .repeat(beam_size, 1).unsqueeze(1).cuda()

        for i in range(self.max_sentence_length):
            hidden, states = self.lstm(words_feed, states)
            outputs = self.output_linear(hidden.squeeze(1))
            beam_indices, words_indices = beam_searcher.expand_beam(outputs=outputs)

            if len(beam_indices) == 0 or i == 15:
                generated_captions = beam_searcher.get_results()[:, 0]
                outcaps = self.embed.words_from_indices(generated_captions.cpu().numpy())
            else:
                words_feed = torch.stack([self.embed.word_embeddings_from_indices(words_indices)]).view(
                    beam_size, 1, -1).cuda()
        return " ".join(outcaps).split(self.embed.END_SYMBOL)[0]

    def sample_with_embedding(self, images_features):
        batch_size = images_features.size(0)

        # init the result with zeros and lstm states
        result = torch.zeros(batch_size, self.max_sentence_length, self.embed.embed_size).cuda()
        hidden = self.init_hidden(images_features)

        # embed the start symbol
        inputs = self.embed.word_embeddings([self.embed.START_SYMBOL] * batch_size).unsqueeze(1).cuda()

        for i in range(self.max_sentence_length):
            # store the result
            result[:, i] = inputs.squeeze(1)
            inputs = Variable(inputs)
            _, hidden = self.lstm(inputs, hidden)
            outputs = self.output_linear(hidden[0]).squeeze(0)
            predicted = outputs.max(-1)[1]     #.max(-1)[1]    （-1）表示按照最后一个维度求最大值，[1]表示返回最大值的索引，即返回0或1

            # embed the next inputs, unsqueeze is required 'cause of shape (batch_size, 1, embedding_size)
            inputs = self.embed.word_embeddings_from_indices(predicted.cpu().data.numpy()).unsqueeze(1).cuda()

        return Variable(result)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self):
        torch.save({"state_dict": self.state_dict()}, FilePathManager.resolve("models/generator.pth"))

    @staticmethod
    def load(corpus: Corpus, max_sentence_length=17):
        state_dict = torch.load(FilePathManager.resolve("models/generator.pth"))
        state_dict = state_dict["state_dict"]
        generator = ConditionalGenerator(corpus, max_sentence_length=max_sentence_length)
        generator.load_state_dict(state_dict)
        return generator


if __name__ == '__main__':
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    generator = ConditionalGenerator.load(corpus).cuda()
    extractor = VggExtractor()
    image = extractor.extract(FilePathManager.resolve("test_images/image_1.png"))
    print(generator.sample(image))
