# coding: utf-8
'''
在给定的数据集划分上生成描述结果，并且计算各种评价指标
'''

from __future__ import unicode_literals
from __future__ import absolute_import
import pickle
from utils import CocoResFormat
import torch
from torch.autograd import Variable
from caption import Vocabulary
from data import get_eval_loader
from model import BiLSTM
from args import vocab_pkl_path, feature_h5_path
from args import model_pth_path, best_meteor_pth_path, best_cider_pth_path
from args import feature_size, max_frames, max_words
from args import projected_size, hidden_size, word_size
from args import test_range, test_prediction_txt_path, test_reference_txt_path
import sys
sys.path.append('./coco-caption/')
import argparse
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
    eval_loader = get_eval_loader(eval_range, feature_h5_path)

    result = {}
    for i, (videos, video_ids) in enumerate(eval_loader):
        # 构造mini batch的Variable
        videos = Variable(videos)
        videos = videos.to(DEVICE)

        outputs = net(videos, None)
        for (tokens, vid) in zip(outputs, video_ids):
            s = net.decoder.decode_tokens(tokens.data)
            result[vid] = s

    prediction_txt = open(prediction_txt_path, 'w')
    for vid, s in result.items():
        prediction_txt.write('%d\t%s\n' % (vid, s))  # 注意，MSVD数据集的视频文件名从1开始

    prediction_txt.close()

    # 开始根据生成的结果计算各种指标
    metrics = measure(prediction_txt_path, reference)
    return metrics


if __name__ == '__main__':
    with open(vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)
    parser = argparse.ArgumentParser(description='evaluate a video captioning model')

    parser.add_argument('--metric', dest='metric', type=str,
                        help='choose the metric from METEOR|CIDEr')
    args = parser.parse_args()

    # 载入预训练模型
    bi_lstm = BiLSTM(feature_size, projected_size, hidden_size, word_size, max_frames, max_words, vocab)
    if not args.metric:
        bi_lstm.load_state_dict(torch.load(model_pth_path))
    elif args.metric == 'METEOR':
        bi_lstm.load_state_dict(torch.load(best_meteor_pth_path))
    elif args.metric == 'CIDEr':
        bi_lstm.load_state_dict(torch.load(best_cider_pth_path))
    else:
        print('Please choose the metric from METEOR|CIDEr')
    bi_lstm.to(DEVICE)
    bi_lstm.eval()
    reference_json_path = '{0}.json'.format(test_reference_txt_path)
    reference = COCO(reference_json_path)
    evaluate(vocab, bi_lstm, test_range, test_prediction_txt_path, reference)
