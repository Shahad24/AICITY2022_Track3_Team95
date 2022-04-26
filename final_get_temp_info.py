#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:56:27 2022

@author: noura
"""

from glob import glob
import os
import numpy as np


x = []
classess = { '0': [], '1': [], '2': [], '3': [],
            '4': [], '5': [], '6': [], '7': [],
            '8': [], '9': [], '10': [], '11': [],
            '12': [], '13': [], '14': [], '15': [],
            '16': [], '17': []}
def overlaps(x, y):
    return max(x.start, y.start) < min(x.stop, y.stop)


def prob(Prob_files, x, classess):

    for npz_file in Prob_files:
        # npz_file= "/home/noura/slowfast/data/features_videos/A2_crop_features_probabilities/Rear_view_user_id_42271_NoAudio_3/P_00086.npz"
        file_name = os.path.basename(npz_file)[:-4]
        # print(file_name)
        file_name = int(file_name[2:])
        x.append(file_name)
        # print(file_name)
        prob = np.load(npz_file)
        # print("len", len(prob))
        prob = prob['arr_0']
        # print('shape', prob.shape)
        # print(prob)
        counter = 0
        for num in classess:
            classess[num].append(prob[0][counter])
            counter += 1


def top_k_temproal(classess, class_num, N):

    # 	print(classess[class_num])
    files = []
    prob = []
    res = sorted(range(len(classess[class_num])),
                 key=lambda sub: classess[class_num][sub])[-N:]
    print('res',res)
    for r in res:
        files.append(x[r])
        prob.append(classess[class_num][r])
        print("r features file", x[r], 'prob = ',classess[class_num][r])
    return files, prob


def get_start_end(files, prob, tries):

    avg_prob = []

    seq_prob = []
    seq_files = []

    all_prob = []
    all_files = []
    # files = [173, 160, 167, 169, 171, 240, 172, 170, 58, 164, 163, 165]
    # prob = [0.9996454, 0.16775487, 0.96975374, 0.98636496, 0.26861706, 0.09451051, 0.7345511, 0.9206978, 0.7588486, 0.89265794, 0.7354297, 0.98865175]
    order = np.argsort(files)
    # order = sorted(range(len(files)),key=files.__getitem__)
    # files = [x for i, x in (zip(order, files))]
    # prob = [x for i, x in (zip(order, prob))]

    print("files len", len(files))
    print("prob len", len(prob))

    print("files len", (files))
    print("prob len", (prob))

    seq_prob.append(prob[order[0]])
    seq_files.append(files[order[0]])

    # order = [56, 556, 453]
    for i in range(len(order)):
        # print(i)
        
        # print(files[order[i+1]])
        if i != len(files)-1:
            print(files[order[i]])
            if files[order[i+1]] <= (files[order[i]]+5):
                seq_prob.append(prob[order[i+1]])
                seq_files.append(files[order[i+1]])
                # print("in if seq_prob", seq_prob)
                # print("in if seq_files", seq_files)
                if i+1 == len(files)-1:
                    all_prob.append(seq_prob)
                    all_files.append(seq_files)

            else:
                if (seq_files[0] != seq_files[-1]) and ((seq_files[-1] - seq_files[0]) > 4):
                    all_prob.append(seq_prob)
                    all_files.append(seq_files)

                # print("in else seq_prob", all_prob)
                # print("in else seq_files", all_files)

                seq_prob = []
                seq_files = []

                seq_prob.append(prob[order[i+1]])
                seq_files.append(files[order[i+1]])

    print("all_prob", all_prob)
    print("all_files", all_files)
    for i in all_prob:
        avg_prob.append(sum(i)/len(i))

    if len(avg_prob) < tries:
        return -1, -1
    for i in range(tries):
        max_avg = max(avg_prob)
        ind = avg_prob.index(max_avg)
        del avg_prob[ind]
        del all_prob[ind]
        del all_files[ind]
        print("in tries")

    if len(avg_prob) != 0:
        max_avg = max(avg_prob)
        ind = avg_prob.index(max_avg)
        start = all_files[ind][0]
        end = all_files[ind][-1]
    else:
        start = -1
        end = -1

    # print("max_avg", max_avg)
    # print("ind", ind)

    return start, end


# Prob_files = glob(
#     'C:/Users/Taghreed/Desktop/A2_crop_features_probabilities/Rear_view_user_id_56306_NoAudio_2/P_*')


vid_ids = {'1': 'Rear_view_user_id_42271_NoAudio_3', '2':'Rear_view_user_id_42271_NoAudio_4',
          '3': 'Rear_view_user_id_56306_NoAudio_2','4': 'Rear_view_user_id_56306_NoAudio_3' , 
          '5': 'Rear_view_User_id_65818_NoAudio_1', '6': 'Rear_view_User_id_65818_NoAudio_2', 
          '7': 'Rearview_mirror_User_id_72519_NoAudio_2', '8': 'Rearview_mirror_User_id_72519_NoAudio_3', 
          '9': 'Rear_view_User_id_79336_NoAudio_0', '10': 'Rear_view_User_id_79336_NoAudio_2'}


Prob_folder = glob(
    '/workspace/AICITY2022_Track3_Team95-main/slowfast_Inference/features_videos/last')
for k in Prob_folder:
    
    Prob_files = glob(k+'/P_*')
    print("Prob_files", Prob_files[0])
    vid_name = k.split('\\')[-1]  


    vid_id = -1 # if the video not from A2 

    for i in vid_ids: # if video from A2 to match the required format :)  
        if vid_ids[i] == vid_name:
            vid_id = i

    
    x = []
    classess = { '0': [],'1': [], '2': [], '3': [],
            '4': [], '5': [], '6': [], '7': [],
            '8': [], '9': [], '10': [], '11': [],
            '12': [], '13': [], '14': [], '15': [],
            '16': [], '17': []}
    prob(Prob_files, x, classess)
    
    all_temp = { '1': [], '2': [], '3': [],
                '4': [], '5': [], '6': [], '7': [],
                '8': [], '9': [], '10': [], '11': [],
                '12': [], '13': [], '14': [], '15': [],
                '16': [], '17': []}
    
    
    class_num = ['1', '2', '5', '7', '10', '11', '13', '14', '15', '17', '8', '9',
                 '12', '3', '4', '16', '6']
    print("k", k)
    for i in class_num:
        # print("class_num", i)
        tries = 0
        N = 12
        files, prob_val = top_k_temproal(classess, i, N)
        start, end = get_start_end(files, prob_val, tries)
        # print(start, end)
        all_temp[i].append([start, end])
        for j in class_num:
            if i == j :
                break
            if len(all_temp[j]) == 0 or (all_temp[j][0][0] == -1 and all_temp[j][0][1] == -1):
                continue
            else:
                while(overlaps(range(all_temp[j][0][0], all_temp[j][0][1]), range(start, end))):
                    tries += 1
                    start, end = get_start_end(files, prob_val, tries)
                    print("tries", tries, j)
        all_temp[i][0] = [start, end]
    activity_id = 0
    print("all_temp", all_temp)
    
    with open ("/workspace/AICITY2022_Track3_Team95-main/slowfast_Inference/shahad_LAST.txt", "a") as file:
        for t in all_temp:
            if (all_temp[t][0][0]) == -1 :
                activity_id +=1
                continue
            activity_id +=1
            if ( activity_id == 6 or activity_id == 12 or activity_id == 16 ) :
                continue
            file.write("%s " % vid_id )
            file.write("%s " % activity_id )
            file.write("%s " % (2 * all_temp[t][0][0] - 2))
            file.write("%s \n" % (2 * all_temp[t][0][1] - 1))
        # file.writelines(["%s\n" % item  for item in list])
        
