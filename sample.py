# coding: utf-8

'''
得到某个视频的caption和attention vector
'''

import cv2
import os
import sys
import pickle
import torch
import h5py
import numpy as np
from options import args
from utils import Vocabulary
import models

def open_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frame_list = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        # cv2.imshow('Video', frame)
        # cv2.waitKey(30)
        frame_list.append(frame)
        frame_count += 1
    indices = np.linspace(0, frame_count, args.max_frames, endpoint=False, dtype=int)
    frame_list = np.array(frame_list)[indices]
    return frame_list


def sample(vocab, video_feat, model, video_path, vid):
    # 为每个视频建立保存可视化结果的目录
    #img_dir = os.path.join(args.visual_dir, str(vid))
    #if not os.path.exists(img_dir):
    #    os.mkdir(img_dir)

    # frame_list = open_video(video_path)
    if args.use_cuda:
        video_feat = video_feat.cuda()
    video_feat = video_feat.unsqueeze(0)
    outputs = model(video_feat, None)
    # outputs = outputs.max(2)[1]
    words = []
    for i, token in enumerate(outputs.data.squeeze()):
        if token == vocab('<end>'):
            break
        word = vocab.idx2word[token]
        # print(word)
        words.append(word)
    caption = ' '.join(words)
    print(caption)


if __name__ == '__main__':
    with open(args.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    features = h5py.File(args.feature_h5_path, 'r')[args.feature_h5_feats]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        model = models.BiLSTM_attention(args.feature_size, 
                                        args.projected_size, 
                                        args.hidden_size, 
                                        args.word_size, 
                                        args.max_frames, 
                                        args.max_words, 
                                        vocab)
    
    #model.load_state_dict(torch.load(best_banet_pth_path))
    model.load_state_dict(torch.load(args.best_meteor_pth_path))
    model.cuda()
    model.eval()

    videos = sorted(os.listdir(args.video_root), key=args.video_sort_lambda)

    if len(sys.argv) > 1:
        vid = int(sys.argv[1])
        video_path = os.path.join(args.video_root, videos[vid])
        video_feat = torch.autograd.Variable(torch.from_numpy(features[vid]))
        sample(vocab, video_feat, model, video_path, vid)
    else:
        # selected_videos = [1412, 1420, 1425, 1466, 1484, 1554, 1821, 1830, 1841,
        #                    1848, 1849, 1850, 1882, 1884, 1931, 1934, 1937, 1944,
        #                    1949, 1950, 1951, 1962]
        # for vid in selected_videos:
        for vid in range(1000,1100):
            print(vid)
            video_path = os.path.join(args.video_root, videos[vid])
            video_feat = torch.autograd.Variable(torch.from_numpy(features[vid]))
            sample(vocab, video_feat, model, video_path, vid)
