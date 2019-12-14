# parameters setting

import argparse
import os
#import time
import datetime

parser = argparse.ArgumentParser(description = 'DeepVC')

parser.add_argument('--phase', type = str, default = 'train', 
                    help = 'choose the run phase of the program from {train, evaluate, apply}')

# paths need to be specified
parser.add_argument('--path_dir_MSVD', type = str, default = '/home/dlt/workspace/Data/MSVD_DeepVC', 
                    help = 'path of directory to save MSVD Dataset')
parser.add_argument('--path_dir_MSRVTT', type = str, default = '/home/dlt/workspace/Data/MSRVTT_DeepVC', 
                    help = 'path of directory to save MSRVTT Dataset')
parser.add_argument('--path_dir_feats', type = str, default = '/home/dlt/workspace/Data/feats_DeepVC', 
                    help = 'path of directory to save video frames\' features extracted by CNN')
parser.add_argument('--path_dir_PModels', type = str, default = '/home/dlt/workspace/PModels', 
                    help = 'path of directory to save pretrained models')
parser.add_argument('--path_dir_results', type = str, default = '/home/dlt/workspace/Codes/DeepVC/results', 
                    help = 'path of directory to save training results')

# dataset used to train
parser.add_argument('--dataset', type = str, default = 'msvd', 
                    help = 'dataset used to train, choose from {msvd, msrvtt}')

# hyper-parameters about training
parser.add_argument('--num_epochs', type = int, default = 50, 
                    help = 'number of epoches of training')
parser.add_argument('--batch_size', type = int, default = 100, 
                    help = 'batch size')
parser.add_argument('--learning_rate', type = float, default = 1e-4, 
                    help = 'learning rate')
parser.add_argument('--ss_factor', type = float, default = 20, 
                    help = '?')
parser.add_argument('--drop_out', type = float, default = 0.5, 
                    help = 'dropout rate')
parser.add_argument('--use_cuda', type = int, default = 1, 
                    help = 'whether to use CUDA')
parser.add_argument('--use_checkpoint', type = int, default = 0, 
                    help = 'whether to use checkpoint')

# parameters about model
parser.add_argument('--model', type = str, default = 'BiLSTM_attention_deepout',
                    help = 'choose a model from {S2VT, BiLSTM_attention_deepout}')
parser.add_argument('--projected_size', type = int, default = 1000, 
                    help = '?')
parser.add_argument('--word_size', type = int, default = 300, 
                    help = '?')
parser.add_argument('--hidden_size', type = int, default = 1024, 
                    help = '?')
parser.add_argument('--beam_size', type = int, default = 5, 
                    help = 'width of beam search')
parser.add_argument('--frame_shape', type = tuple, default = (3, 299, 299), 
                    help = 'size of every video frame')
parser.add_argument('--a_feature_size', type = int, default = 1536, 
                    help = '?')
parser.add_argument('--m_feature_size', type = int, default = 1024, 
                    help = '?')
parser.add_argument('--feature_size', type = int, default = 1536, 
                    help = '？')
parser.add_argument('--frame_sample_rate', type = int, default = 10, 
                    help = 'the interval of sampling video frames')
parser.add_argument('--max_frames', type = int, default = 26, 
                    help = 'maximum of sampling video frames')
parser.add_argument('--max_words', type = int, default = 26, 
                    help = '?')
parser.add_argument('--use_cpt', type = int, default = 0, 
                    help = 'whether to use checkpoint')

# parameters about evaluating
parser.add_argument('--optimal_metric', type = str, default = 'METEOR',
                    help = 'choose the metric from {METEOR, CIDEr} to obtain its maximum')

args = parser.parse_args()

args.str_current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
args.log_environment = os.path.join('logs', args.str_current_time)  # tensorboard的记录环境

# parameters abount datasets
## MSVD
args.msvd_video_root = args.path_dir_MSVD + '/youtube_videos'
args.msvd_csv_path = args.path_dir_MSVD + '/video_corpus.csv'  # 手动修改一些数据集中的错误,这么多怎么修改？
args.msvd_video_name2id_map = args.path_dir_MSVD + '/youtube_mapping.txt'
args.msvd_anno_json_path = args.path_dir_MSVD + '/annotations.json'  # MSVD并未提供这个文件，需要自己写代码生成（build_msvd_annotation.py）
args.msvd_video_sort_lambda = lambda x: int(x[5:-4]) # used to extract video ID
args.msvd_train_range = (0, 1200)
args.msvd_val_range = (1200, 1300)
args.msvd_test_range = (1300, 1970)

## MSRVTT
args.msrvtt_video_root = args.path_dir_MSRVTT + '/Videos'
args.msrvtt_anno_trainval_path = args.path_dir_MSRVTT + '/train_val_videodatainfo.json'
args.msrvtt_anno_test_path = args.path_dir_MSRVTT + '/test_videodatainfo.json'
args.msrvtt_anno_json_path = args.path_dir_MSRVTT + '/datainfo.json'
args.msrvtt_video_sort_lambda = lambda x: int(x[5:-4]) # used to extract video ID
args.msrvtt_train_range = (0, 6513)
args.msrvtt_val_range = (6513, 7010)
args.msrvtt_test_range = (7010, 10000)

## dataset to use
if (args.dataset == 'msvd'):
    args.video_root = args.msvd_video_root
    args.video_sort_lambda = args.msvd_video_sort_lambda
    args.anno_json_path = args.msvd_anno_json_path
    args.train_range = args.msvd_train_range
    args.val_range = args.msvd_val_range
    args.test_range = args.msvd_test_range
    args.path_dir_Dataset = args.path_dir_MSVD
elif (args.dataset == 'msrvtt'):
    args.video_root = args.msrvtt_video_root
    args.video_sort_lambda = args.msrvtt_video_sort_lambda
    args.anno_json_path = args.msrvtt_anno_json_path
    args.train_range = args.msrvtt_train_range
    args.val_range = args.msrvtt_val_range
    args.test_range = args.msrvtt_test_range
    args.path_dir_Dataset = args.path_dir_MSRVTT

if not os.path.exists(args.path_dir_feats):
    os.mkdir(args.path_dir_feats)

args.vocab_pkl_path = os.path.join(args.path_dir_feats, args.dataset + '_vocab.pkl')
args.vocab_embed_path = os.path.join(args.path_dir_feats, args.dataset + '_vocab_embed.pkl')
args.caption_pkl_path = os.path.join(args.path_dir_feats, args.dataset + '_captions.pkl')
args.caption_pkl_base = os.path.join(args.path_dir_feats, args.dataset + '_captions')
args.train_caption_pkl_path = args.caption_pkl_base + '_train.pkl'
args.val_caption_pkl_path = args.caption_pkl_base + '_val.pkl'
args.test_caption_pkl_path = args.caption_pkl_base + '_test.pkl'

args.feature_h5_path = os.path.join(args.path_dir_feats, args.dataset + '_features.h5')
args.feature_h5_feats = 'feats'
args.feature_h5_lens = 'lens'

# 结果评估相关的参数
args.path_dir_result = args.path_dir_results + '/' + args.model + '_' + args.dataset + '_' + args.str_current_time
if not os.path.exists(args.path_dir_result):
    os.mkdir(args.path_dir_result)

args.val_reference_txt_path = os.path.join(args.path_dir_Dataset, args.dataset + '_val_references.txt')
args.val_prediction_txt_path = os.path.join(args.path_dir_Dataset, args.dataset + '_val_predictions.txt')

args.test_reference_txt_path = os.path.join(args.path_dir_Dataset, args.dataset + '_test_references.txt')
args.test_prediction_txt_path = os.path.join(args.path_dir_Dataset, args.dataset + '_test_predictions.txt')

# checkpoint相关的超参数
args.resnet_checkpoint = args.path_dir_PModels + '/resnet50-19c8e357.pth'  # 直接用pytorch训练的模型
args.IRV2_checkpoint = args.path_dir_PModels + '/inceptionresnetv2-520b38e4.pth'
args.vgg_checkpoint = args.path_dir_PModels + '/vgg16-00b39a1b.pth'  # 从caffe转换而来
args.c3d_checkpoint = args.path_dir_PModels + '/c3d.pickle'

args.model_pth_path = os.path.join(args.path_dir_result, args.dataset + '_model.pth')
args.best_meteor_pth_path = os.path.join(args.path_dir_result, args.dataset + '_best_meteor.pth')
args.best_cider_pth_path = os.path.join(args.path_dir_result, args.dataset + '_best_cider.pth')
args.optimizer_pth_path = os.path.join(args.path_dir_result, args.dataset + '_optimizer.pth')
args.best_meteor_optimizer_pth_path = os.path.join(args.path_dir_result, args.dataset + '_best_meteor_optimizer.pth')
args.best_cider_optimizer_pth_path = os.path.join(args.path_dir_result, args.dataset + '_best_cider_optimizer.pth')

# 图示结果相关的超参数 ?
#args.visual_dir = '/visuals' + '/' + args.model + '_' + args.dataset + '_' + args.str_current_time
#if not os.path.exists(args.visual_dir):
#    os.mkdir(args.visual_dir)
