# -*- coding:utf-8 -*-

import copy

import torch
import torch.nn.functional as F

                     #调用到rollout的模块：ConditionalGenerator、
class Rollout:       # Rollout(max_sentence_length, corpus) 
    """Roll-out policy"""

    def __init__(self, max_sentence_length, corpus):
        self.embed = corpus
        self.lstm = None
        self.max_sentence_length = max_sentence_length
        self.output_linear = None
                                                         
    def reward(self, generated, image_features, hidden, monte_carlo_count, evaluator, steps=1):   # rollout.reward(current_generated, image_features, hidden, monte_carlo_count, evaluator)
        assert monte_carlo_count % steps == 0, "Monte Carlo Count can't be divided by Steps"    #assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常或者执行包含语句
        monte_carlo_count //= steps            # /求商，//求商的整数部分，%求余数

        with torch.no_grad():
            batch_size = generated.size(0)     #generated : [batch,句子长度，单词维度]
            result = torch.zeros(batch_size, 1).cuda()     # [batch_size, 1] ,行
            remaining = self.max_sentence_length - generated.shape[1]
            h, c = hidden
            generated = generated.repeat(monte_carlo_count, 1, 1)    #.repeat(a,repeats,axis=None)在轴上重复a数次，repeats是次数
            for _ in range(steps):
                hidden = (h.repeat(1, monte_carlo_count, 1), c.repeat(1, monte_carlo_count, 1))
                inputs = generated[:, -1].unsqueeze(1)
                current_generated = generated
                for i in range(remaining): 
                    _, hidden = self.lstm(inputs, hidden)
                    outputs = self.output_linear(hidden[0]).squeeze(0)
                    outputs = F.softmax(outputs, -1)
                    predicted = outputs.multinomial(1)      #multinomial ??? 对outputs取值1次？？？
                    # embed the next inputs, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                    inputs = self.embed.word_embeddings_from_indices(predicted.view(-1).cpu().data.numpy()).unsqueeze(
                        1).cuda()
                    current_generated = torch.cat([current_generated, inputs], dim=1)
                reward = evaluator(image_features.repeat(monte_carlo_count, 1), current_generated)    # rewoard
                reward = reward.view(batch_size, monte_carlo_count, -1).sum(1)
                result += reward
                result /= monte_carlo_count
            return result

    def update(self, original_model):   # rollout.update(self)
        self.lstm = copy.deepcopy(original_model.lstm)      #opy.copy()与copy.deepcopy()的区别，浅拷贝与深拷贝
        self.lstm.flatten_parameters()  #重置参数的数据指针,调用flatten_parameters让参数的数据存放成连续的块，提高内存的利用率和效率
        self.output_linear = copy.deepcopy(original_model.output_linear)
