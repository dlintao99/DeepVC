import os
from args import msvd_video_name2id_map,video_root
video_name2id = {}
with open(msvd_video_name2id_map, 'r') as f:
    lines = f.readlines()
    for line in lines:
        name, vid = line.strip().split()
        vid = int(vid[3:]) - 1
        vid = 'video%d.avi' % vid
        video_name2id[name+'.avi'] = vid
video_list=os.listdir(video_root)
for video in video_list:
    os.rename(os.path.join(video_root,video),os.path.join(video_root,video_name2id[video]))