#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:56:27 2022
@author: Taghreed
"""

from glob import glob
import os
import numpy as np
import argparse 
import csv

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
        file_name = os.path.basename(npz_file)[:-4]
        file_name = int(file_name[2:])
        x.append(file_name)
        prob = np.load(npz_file)
        prob = prob['arr_0']
        counter = 0
        for num in classess:
            classess[num].append(prob[0][counter])
            counter += 1


def top_k_temproal(classess, class_num, N):

    files = []
    prob = []
    res = sorted(range(len(classess[class_num])),
                 key=lambda sub: classess[class_num][sub])[-N:]
    for r in res:
        files.append(x[r])
        prob.append(classess[class_num][r])
    
    return files, prob


def get_start_end(files, prob, tries):

    avg_prob = []

    seq_prob = []
    seq_files = []

    all_prob = []
    all_files = []
    order = np.argsort(files)

    seq_prob.append(prob[order[0]])
    seq_files.append(files[order[0]])

    for i in range(len(order)):
        
        if i != len(files)-1:
            if files[order[i+1]] <= (files[order[i]]+5):
                seq_prob.append(prob[order[i+1]])
                seq_files.append(files[order[i+1]])
               
                if i+1 == len(files)-1:
                    all_prob.append(seq_prob)
                    all_files.append(seq_files)

            else:
                if (seq_files[0] != seq_files[-1]) and ((seq_files[-1] - seq_files[0]) > 4):
                    all_prob.append(seq_prob)
                    all_files.append(seq_files)

                seq_prob = []
                seq_files = []

                seq_prob.append(prob[order[i+1]])
                seq_files.append(files[order[i+1]])


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
       

    if len(avg_prob) != 0:
        max_avg = max(avg_prob)
        ind = avg_prob.index(max_avg)
        start = all_files[ind][0]
        end = all_files[ind][-1]
    else:
        start = -1
        end = -1

    return start, end


parser = argparse.ArgumentParser(description='temporal loclization')
    
parser.add_argument('--prob_path', metavar='path', required=True,
                        help='the path to the folder that contains a folders of probabilities .. ')
    
parser.add_argument('--out_file', metavar='path', required=True,
                        help= 'the path where the text file will be saved .. ')
args = parser.parse_args()   
Prob_folder = glob(args.prob_path+'/*')
out_file = args.out_file+"/temporal_locations.txt" 
print("number of videos is", len(Prob_folder)) 


reader = csv.reader(open('video_ids.csv', 'r'))
vid_ids = {}
ignore_header = True
for row in reader:
    if ignore_header:
        ignore_header = False
        continue
    vid_id, dashboard, rear, right = row
    vid_ids[vid_id] = rear[:-4]  



for k in Prob_folder:
    
    Prob_files = glob(k+'/P_*')
    vid_name = k.split('/')[-1] 
    print("start processing: ", vid_name)


    vid_id = -1 # if the video not from A2 

    for i in vid_ids: # if video from A2 to match the required format :)  
        if vid_ids[i] == vid_name:
            vid_id = i
            
    if vid_id == -1 : # if the video not from A2, rather than write ID, will write it's name
           vid_id =  vid_name

    
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
    
    for i in class_num:
        tries = 0
        N = 12
        files, prob_val = top_k_temproal(classess, i, N)
        start, end = get_start_end(files, prob_val, tries)
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
                    
        all_temp[i][0] = [start, end]
    activity_id = 0
    
    
    with open (out_file, "a") as file:
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
    print(vid_name, "done")
