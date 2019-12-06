# coding: utf-8

from __future__ import print_function
from builtins import range
import os
import sys
import shutil
import pickle
import time
from utils import blockPrint, enablePrint
from caption import Vocabulary
from data import get_train_loader
from model import BiLSTM
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate
from args import vocab_pkl_path, train_caption_pkl_path, feature_h5_path
from args import num_epochs, batch_size, learning_rate, ss_factor
from args import projected_size, word_size, hidden_size
from args import feature_size, max_frames, max_words
from args import use_checkpoint
from args import model_pth_path, optimizer_pth_path
from args import best_meteor_pth_path, best_meteor_optimizer_pth_path
from args import best_cider_pth_path, best_cider_optimizer_pth_path
from args import test_range, test_prediction_txt_path, test_reference_txt_path
# from args import val_range, val_prediction_txt_path, val_reference_txt_path
from args import log_environment
from tensorboard_logger import configure, log_value
sys.path.append('./coco-caption/')
from pycocotools.coco import COCO

configure(log_environment, flush_secs=10)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载词典
with open(vocab_pkl_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# 构建模型
bi_lstm = BiLSTM(feature_size, projected_size, hidden_size, word_size, max_frames, max_words, vocab)
print('Total parameters:', sum(param.numel() for param in bi_lstm.parameters()))

if os.path.exists(model_pth_path) and use_checkpoint:
    bi_lstm.load_state_dict(torch.load(model_pth_path))
bi_lstm.to(DEVICE)

# 初始化损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(bi_lstm.parameters(), lr=learning_rate)
if os.path.exists(optimizer_pth_path) and use_checkpoint:
    optimizer.load_state_dict(torch.load(optimizer_pth_path))

# 打印训练环境的参数设置情况
print('Learning rate: %.5f' % learning_rate)
print('Batch size: %d' % batch_size)

# 初始化数据加载器
train_loader = get_train_loader(train_caption_pkl_path, feature_h5_path, batch_size)
total_step = len(train_loader)

# 准备一下验证用的ground-truth
reference_json_path = '{0}.json'.format(test_reference_txt_path)
reference = COCO(reference_json_path)

# 开始训练模型
best_meteor = 0
best_cider = 0
loss_count = 0
lr_decay_flag = 0
count = 0
v = None
t = None
e = None
saving_schedule = [int(x * total_step) for x in [0.25, 0.5, 0.75, 1.0]]
print('total: ', total_step)
print('saving_schedule: ', saving_schedule)
for epoch in range(num_epochs):
    start_time = time.time()
    if epoch % 10 ==0 and epoch > 0:
        learning_rate /= 10
    epsilon = max(0.6, ss_factor / (ss_factor + np.exp(epoch / ss_factor)))
    # epsilon = max(0.75, 1 - int(epoch / 5) * 0.05)
    print('epoch:%d\tepsilon:%.8f' % (epoch, epsilon))
    log_value('epsilon', epsilon, epoch)
    for i, (videos, captions, cap_lens, video_ids) in enumerate(train_loader, start=1):
        # 构造mini batch的Variable
        videos = Variable(videos)
        targets = Variable(captions)
        videos = videos.to(DEVICE)
        targets = targets.to(DEVICE)
        v, t = videos, targets
        e = epsilon

        optimizer.zero_grad()
        outputs = bi_lstm(videos, targets, epsilon)
        tokens = outputs
        # 因为在一个epoch快要结束的时候，有可能采不到一个刚好的batch
        # 所以要重新计算一下batch size
        bsz = len(captions)
        # 把output压缩（剔除pad的部分）之后拉直
        outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], 0)
        outputs = outputs.view(-1, vocab_size)
        # 把target压缩（剔除pad的部分）之后拉直
        targets = torch.cat([targets[j][:cap_lens[j]] for j in range(bsz)], 0)
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        log_value('loss', loss.item(), epoch * total_step + i)
        loss_count += loss.item()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 or bsz < batch_size:
            loss_count /= 10 if bsz == batch_size else i % 10
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                  (epoch, num_epochs, i, total_step, loss_count,
                   np.exp(loss_count)))
            loss_count = 0
            tokens = tokens.max(2)[1]
            tokens = tokens.data[0].squeeze()
            we = bi_lstm.decoder.decode_tokens(tokens)
            gt = bi_lstm.decoder.decode_tokens(captions[0].squeeze())
            print('[vid:%d]' % video_ids[0])
            print('WE: %s\nGT: %s' % (we, gt))

        if i in saving_schedule:
            torch.save(bi_lstm.state_dict(), model_pth_path)
            torch.save(optimizer.state_dict(), optimizer_pth_path)

            # 计算一下在val集上的性能并记录下来
            blockPrint()
            start_time_eval = time.time()
            bi_lstm.eval()
            metrics = evaluate(vocab, bi_lstm, test_range, test_prediction_txt_path, reference)
            end_time_eval = time.time()
            enablePrint()
            print('evaluate time: %.3fs' % (end_time_eval-start_time_eval))

            for k, v in metrics.items():
                log_value(k, v, epoch*len(saving_schedule)+count)
                print('%s: %.6f' % (k, v))
                if k == 'METEOR' and v > best_meteor:
                    # 备份在val集上METEOR值最好的模型
                    shutil.copy2(model_pth_path, best_meteor_pth_path)
                    shutil.copy2(optimizer_pth_path, best_meteor_optimizer_pth_path)
                    best_meteor = v
                if k == 'CIDEr' and v > best_cider:
                    # 备份在val集上CIDEr值最好的模型
                    shutil.copy2(model_pth_path, best_cider_pth_path)
                    shutil.copy2(optimizer_pth_path, best_cider_optimizer_pth_path)
                    best_cider = v
                    lr_decay_flag = 0
                if k == 'CIDEr' and v < best_cider:
                    lr_decay_flag += 1

            # learning rate decay
            # if lr_decay_flag == 16:
            #     learning_rate /= 5
            #     lr_decay_flag = 0
            print('Step: %d, Learning rate: %.8f' % (epoch*len(saving_schedule)+count, learning_rate))
            optimizer = torch.optim.Adam(bi_lstm.parameters(), lr=learning_rate)
            log_value('Learning rate', learning_rate, epoch*len(saving_schedule)+count)
            count += 1
            count %= 4
            bi_lstm.train()


    end_time = time.time()
    print("*******One epoch time: %.3fs*******\n" % (end_time - start_time))

# with SummaryWriter(log_dir='./graph') as writer:
#     writer.add_graph(bi_lstm, (v, t))