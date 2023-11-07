import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 1
eos_idx = 2
src_vocab_size = 1734
tgt_vocab_size = 5759
batch_size = 8
epoch_num = 10
save_interval = 1000
early_stop = 5
lr = 3e-4

# 遮挡原文token的概率
mask_prob = 0.15
# greed decode的最大句子长度
max_len = 800
# beam size for bleu
beam_size = 3
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True

data_dir = './data'
txt_folder = './data/text/txt'
train_data_path = './data/text/train.json'
dev_data_path = './data/text/dev.json'
test_data_path = './data/text/test.json'
model_dir = './experiment'
log_path = f'{model_dir}/train.log'
output_path = f'{model_dir}/output.txt'

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')
