# coding: utf-8
'''
在给定的数据集划分上生成描述结果，并且计算各种评价指标
'''

from __future__ import unicode_literals
from __future__ import absolute_import
import pickle
from utils import CocoResFormat, Vocabulary, decode_tokens
import torch
from torch.autograd import Variable
from data import get_eval_loader
import models
from options import args
import sys
import argparse

sys.path.append('./coco-caption/')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def measure(prediction_txt_path, reference):
    # 把txt格式的预测结果转换成检验程序所要求的格式
    crf = CocoResFormat()
    crf.read_file(prediction_txt_path, True)

    # crf.res就是格式转换之后的预测结果
    cocoRes = reference.loadRes(crf.res)
    cocoEval = COCOEvalCap(reference, cocoRes)

    cocoEval.evaluate()

    print('********** SCORES **********')
    for metric, score in cocoEval.eval.items():
        print('%s: %.2f' % (metric, score*100))
    return cocoEval.eval


def evaluate(vocab, net, eval_range, prediction_txt_path, reference):
    # 载入测试数据集
    eval_loader = get_eval_loader(eval_range, args.feature_h5_path)

    result = {}
    for i, (videos, video_ids) in enumerate(eval_loader):
        # 构造mini batch的Variable
        videos = Variable(videos)
        videos = videos.to(DEVICE)
    
        print('videos.size()', videos.size())
        outputs = net(videos, None)
        for (tokens, vid) in zip(outputs, video_ids):
            s = decode_tokens(tokens.data, vocab)
            result[vid] = s

    prediction_txt = open(prediction_txt_path, 'w')
    for vid, s in result.items():
        prediction_txt.write('%d\t%s\n' % (vid, s))  # 注意，MSVD数据集的视频文件名从1开始

    prediction_txt.close()

    # 开始根据生成的结果计算各种指标
    metrics = measure(prediction_txt_path, reference)
    return metrics


if __name__ == '__main__':
    with open(args.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    # 载入预训练模型
    ## initialize model
    if (args.model == 'S2VT'):
        model = models.S2VT(args.max_frames, 
                            args.max_words,
                            args.feature_size, 
                            args.projected_size,
                            args.hidden_size, 
                            args.word_size,
                            vocab,
                            args.drop_out,
                            DEVICE)
    elif (args.model == 'BiLSTM_attention_deepout'):
        model = models.BiLSTM_attention_deepout(args.feature_size, 
                                        args.projected_size, 
                                        args.hidden_size, 
                                        args.word_size, 
                                        args.max_frames, 
                                        args.max_words, 
                                        vocab)

    if not args.optimal_metric:
        model.load_state_dict(torch.load(args.model_pth_path))
    elif args.optimal_metric == 'METEOR':
        model.load_state_dict(torch.load(args.best_meteor_pth_path))
    elif args.optimal_metric == 'CIDEr':
        model.load_state_dict(torch.load(args.best_cider_pth_path))
    else:
        print('Please choose the metric from METEOR|CIDEr to obtain its maximum')
    model.to(DEVICE)
    model.eval()
    reference_json_path = '{0}.json'.format(args.test_reference_txt_path)
    reference = COCO(reference_json_path)
    evaluate(vocab, model, args.test_range, args.test_prediction_txt_path, reference)
