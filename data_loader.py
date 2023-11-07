import torch
import json
import numpy as np
import os, re
import random
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import source_tokenizer_load
from utils import target_tokenizer_load

import config
DEVICE = config.device


def built_dataset(txt_folder, train_data_path, dev_data_path, max_length, prob=0.85):
    train_data=[]
    for file in os.listdir(txt_folder):
        if not file.endswith('.txt'):
            continue
        if not os.path.exists(f'{txt_folder}/{file[:-3]}json'):
            continue
        with open(f'{txt_folder}/{file[:-3]}json', 'r', encoding='utf-8') as f:
            phone_lists = json.load(f)
        with open(f'{txt_folder}/{file}', 'r', encoding='utf-8') as f:
            zh_lines = f.readlines()
            i=0
            while i < len(zh_lines):
                if zh_lines[i]=='':
                    i+=1
                    continue
                phone_list = phone_lists[i]
                zh_line = zh_lines[i]
                if len(zh_line) > max_length:
                    i+=1
                    continue
                while random.random() < prob and i+1<len(zh_lines) and zh_lines[i+1]!='' and len(zh_line)+len(zh_lines[i+1])+1 <= max_length:
                    i+=1
                    phone_list += ['.'] + phone_lists[i]
                    zh_line += zh_lines[i]
                train_data.append((phone_list, zh_line.strip()))
                i+=1
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(dev_data_path, 'w', encoding='utf-8') as f:
        json.dump(random.sample(train_data, 100), f, ensure_ascii=False)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            trg = trg.to(DEVICE)
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path, mask_prob=0):
        self.out_src_sent, self.out_tgt_sent = self.get_dataset(data_path, sort=True)
        self.sp_src = source_tokenizer_load()[0]
        self.sp_tgt = target_tokenizer_load()[0]
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.MAS = 3
        self.mask_prob = mask_prob

    @staticmethod
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        dataset = json.load(open(data_path, 'r', encoding='utf-8'))
        out_src_sent = []
        out_tgt_sent = []
        for src_sent, tgt_sent in dataset:
            out_src_sent.append(src_sent)
            out_tgt_sent.append(tgt_sent)
        if sort:
            sorted_index = self.len_argsort(out_tgt_sent)
            out_src_sent = [out_src_sent[i] for i in sorted_index]
            out_tgt_sent = [out_tgt_sent[i] for i in sorted_index]
        return out_src_sent, out_tgt_sent

    def __getitem__(self, idx):
        return self.out_src_sent[idx], self.out_tgt_sent[idx]

    def __len__(self):
        return len(self.out_tgt_sent)
    
    def random_mask(self, input_list):
        return [self.MAS if random.random() < self.mask_prob else num for num in input_list]

    def collate_fn(self, batch):
        src_lists = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.random_mask(self.sp_src(sent)) + [self.EOS] for sent in src_lists]
        tgt_tokens = [[self.BOS] + self.sp_tgt(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_lists, tgt_text, batch_input, batch_target, self.PAD)
