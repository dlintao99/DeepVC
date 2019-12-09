# coding: utf-8

'''
这里存放一些参数
'''
import os
import time

# paths need to be modified
path_dir_MSRVTT = '/home/dlt/workspace/Data/MSRVTT_DeepVC'
path_dir_MSVD = '/home/dlt/workspace/Data/MSVD_DeepVC'
path_dir_feats = '/home/dlt/workspace/Data/feats_DeepVC'
path_dir_PModels = '/home/dlt/workspace/PModels'
ds = 'msvd'

# 训练相关的超参数
num_epochs = 50
batch_size = 100
learning_rate = 1e-4
ss_factor = 20
drop_out = 0.5
use_cuda = True
use_checkpoint = False
time_format = '%m-%d_%X'
current_time = time.strftime(time_format, time.localtime())
env_tag = '%s' % (current_time)
log_environment = os.path.join('logs', env_tag)  # tensorboard的记录环境


# 模型相关的超参数
projected_size = 1000
word_size = 300  # word embedding size
hidden_size = 1024  # 循环网络的隐层单元数目
beam_size = 5

frame_shape = (3, 299, 299)  # 视频帧的形状
a_feature_size = 1536  # 表观特征的大小
m_feature_size = 1024  # 运动特征的大小
# feature_size = a_feature_size + m_feature_size  # 最终特征大小
feature_size = a_feature_size  # 最终特征大小
frame_sample_rate = 10  # 视频帧的采样率
max_frames = 26  # 图像序列的最大长度
max_words = 26  # 文本序列的最大长度


# 数据相关的参数
# 提供两个数据集：MSR-VTT和MSVD
#path_dir_MSRVTT = '/home/dlt/workspace/Data/MSRVTT'
msrvtt_video_root = path_dir_MSRVTT + '/Videos'
msrvtt_anno_trainval_path = path_dir_MSRVTT + '/train_val_videodatainfo.json'
msrvtt_anno_test_path = path_dir_MSRVTT + '/test_videodatainfo.json'
msrvtt_anno_json_path = path_dir_MSRVTT + '/datainfo.json'
msrvtt_video_sort_lambda = lambda x: int(x[5:-4])
msrvtt_train_range = (0, 6513)
msrvtt_val_range = (6513, 7010)
msrvtt_test_range = (7010, 10000)

#path_dir_MSVD = '/home/dlt/workspace/Data/MSVD'
msvd_video_root = path_dir_MSVD + '/youtube_videos'
msvd_csv_path = path_dir_MSVD + '/video_corpus.csv'  # 手动修改一些数据集中的错误,这么多怎么修改？
msvd_video_name2id_map = path_dir_MSVD + '/youtube_mapping.txt'
msvd_anno_json_path = path_dir_MSVD + '/annotations.json'  # MSVD并未提供这个文件，需要自己写代码生成（build_msvd_annotation.py）
msvd_video_sort_lambda = lambda x: int(x[5:-4])
msvd_train_range = (0, 1200)
msvd_val_range = (1200, 1300)
msvd_test_range = (1300, 1970)


dataset = {
    'msr-vtt': [msrvtt_video_root, msrvtt_video_sort_lambda, msrvtt_anno_json_path,
                msrvtt_train_range, msrvtt_val_range, msrvtt_test_range],
    'msvd': [msvd_video_root, msvd_video_sort_lambda, msvd_anno_json_path,
             msvd_train_range, msvd_val_range, msvd_test_range]
}

# 用video_root和anno_json_path这两个变量来切换所使用的数据集
# video_sort_lambda用来对视频按照名称进行排序
#ds = 'msvd'
#ds = 'msr-vtt'
video_root, video_sort_lambda, anno_json_path, \
    train_range, val_range, test_range = dataset[ds]

#feat_dir = '/home/dlt/workspace/Data/DeepVC_feats'
if not os.path.exists(path_dir_feats):
    os.mkdir(path_dir_feats)

vocab_pkl_path = os.path.join(path_dir_feats, ds + '_vocab.pkl')
vocab_embed_path = os.path.join(path_dir_feats, ds + '_vocab_embed.pkl')
caption_pkl_path = os.path.join(path_dir_feats, ds + '_captions.pkl')
caption_pkl_base = os.path.join(path_dir_feats, ds + '_captions')
train_caption_pkl_path = caption_pkl_base + '_train.pkl'
val_caption_pkl_path = caption_pkl_base + '_val.pkl'
test_caption_pkl_path = caption_pkl_base + '_test.pkl'

feature_h5_path = os.path.join(path_dir_feats, ds + '_features.h5')
feature_h5_feats = 'feats'
feature_h5_lens = 'lens'

# 结果评估相关的参数
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

val_reference_txt_path = os.path.join(result_dir, ds + '_val_references.txt')
val_prediction_txt_path = os.path.join(result_dir, ds + '_val_predictions.txt')

test_reference_txt_path = os.path.join(result_dir, ds + '_test_references.txt')
test_prediction_txt_path = os.path.join(result_dir, ds + '_test_predictions.txt')

# checkpoint相关的超参数
resnet_checkpoint = path_dir_PModels + '/resnet50-19c8e357.pth'  # 直接用pytorch训练的模型
IRV2_checkpoint = path_dir_PModels + '/inceptionresnetv2-520b38e4.pth'
vgg_checkpoint = path_dir_PModels + '/vgg16-00b39a1b.pth'  # 从caffe转换而来
c3d_checkpoint = path_dir_PModels + '/c3d.pickle'

model_pth_path = os.path.join(result_dir, ds + '_model.pth')
best_meteor_pth_path = os.path.join(result_dir, ds + '_best_meteor.pth')
best_cider_pth_path = os.path.join(result_dir, ds + '_best_cider.pth')
optimizer_pth_path = os.path.join(result_dir, ds + '_optimizer.pth')
best_meteor_optimizer_pth_path = os.path.join(result_dir, ds + '_best_meteor_optimizer.pth')
best_cider_optimizer_pth_path = os.path.join(result_dir, ds + '_best_cider_optimizer.pth')

# 图示结果相关的超参数
visual_dir = 'visuals'
if not os.path.exists(visual_dir):
    os.mkdir(visual_dir)
