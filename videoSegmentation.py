#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 04:50:01 2022

@author: shahad
"""


'''
this script read frames of videos and save each 30 + 2 frames in single video, then get the next 30 + 2 frame 
as so on. 

the 2 in the end of the previous video will be the first frames in the next video ..


'''

import glob 
import os 
import numpy 
import pandas
import cv2
import argparse 
import math 


num_frames = 64


def init(video_path):
    cap = cv2.VideoCapture(video_path)
    full_rate =  rate = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    vid_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # width = vcap.get(3)
    # height = vcap.get(4)
    # fps = vcap.get(5)
    return full_rate, width, height, vid_length



    

def video_segment_type1(file_paths, out_file) : 
    
    
    videos_paths = file_paths
    
    for path in videos_paths:
        
        # path = path[0].replace("/workspace", 'VideosD')
        video_name = os.path.basename(path)[:-4]
        # parent_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        print(video_name)
        # out_file_name = os.path.join(video_out_file, parent_name+'_'+video_name)
        out_file_name = os.path.join(out_file, video_name)

        print(out_file_name)
        if not os.path.exists(out_file_name) :
            
            print("create path ")
            os.mkdir(out_file_name)
             
        full_rate, width, height, vid_length = init(path)
        print(full_rate, width, height, vid_length )
        reminder = vid_length % 60
        print('reminder', reminder)
        sub_videos_number = math.ceil(vid_length / 60)
        print('sub_videos_number', reminder)

        stream = cv2.VideoCapture(path)

        frame_counter = 0
        sub_videos = 0
        prev_frames_list = []
        sub_video_frame_counter = 0
        sub_videos = 0
        while (1):
            
            ret, frame = stream.read()


            if not ret:
                break

            if sub_video_frame_counter == num_frames or sub_videos == 0 :
                

                sub_videos += 1
                out_sub_video = os.path.join(out_file_name, '{:05d}.mp4'.format(sub_videos))
                output = cv2.VideoWriter(out_sub_video, 
                                     cv2.VideoWriter_fourcc(*'mp4v'), 
                                     full_rate, 
                                     (int(width), int(height)))
            
                print("starting new video = ", sub_videos)
                print('1 length  prev_frames_list = ', len(prev_frames_list))
                sub_video_frame_counter = 0
                for prev_frame in prev_frames_list:
                    
                    sub_video_frame_counter += 1
                    output.write(prev_frame)
                prev_frames_list = []
            # sub_videos += 1
            



            output.write(frame)
            sub_video_frame_counter += 1

        

            if reminder != 0 and sub_videos_number-1 == sub_videos:
                
                print("reminder = ",  reminder)
                max_frame = num_frames - reminder
            else:
                
                max_frame = num_frames - 4

            if sub_video_frame_counter > max_frame:
                
                print('appending prev frame = ', frame_counter)
                print(' max_frame = ', max_frame)
                prev_frames_list.append(frame)
                print('2 length  prev_frames_list = ', len(prev_frames_list))

            frame_counter += 1




def video_segment_type2(file_paths, out_file) : 
    
    
    
    
    for f in file_paths : 
# Renaming the file
        print(f)
    
        f_frames = glob.glob(f+"/*")
        f_frames.sort()
    
        pre_frame_index = 0 
    
        secs = len(f_frames) / 30
    
        vid_id = 1  
    
        ind = 45 

    
    
        while ( vid_id < secs ) : 
            
        
            f_name = f.split("/")[-1]  # get folder name 
        
            f_out_path = os.path.join(out_file, f_name) # out folder 
        
            # create folder for this video 
        
            if not os.path.exists(f_out_path) :
                os.mkdir(f_out_path)
        
        
            video_name = f_name+"/"+str(vid_id) # video name 
        
            # video mp4 path 
        
            video_out_path = os.path.join(out_file, video_name+'.mp4') # or f_out_path + "/" + vid_id + ".mp4"
        
            img =  cv2.imread(f_frames[0])
            width = img.shape[1]
            height = img.shape[0]
            #print('width, height', width, height)
            output = cv2.VideoWriter(os.path.join(f_out_path, '{:05d}.mp4'.format(vid_id)), 
                                    cv2.VideoWriter_fourcc(*'mp4v'), 
                                    30, 
                                    (width, height))
        
        
            # get last index frame for the file 
        
            if vid_id == 1 :
            
                post_frame_index = 64 
        
            elif vid_id < secs  :
                        
                pre_frame_index = ind - 32
            
                post_frame_index = ind + 32
            
                ind = ind + 30 
            
            else :
                
                pre_frame_index = ind - 32 
                post_frame_index = len(f_frames) # get last index 
            
            
            
            # Handling : to double check after elif post_frame_index value not exceed len(f_frames)
            
            if (vid_id > secs ) : 
            
                post_frame_index = len(post_frame_index)
            
        
            
            # get the id of the frames that will be wrtien     
            img_ids = f_frames[pre_frame_index : post_frame_index ]
        
        
            # write frames in the created video 
            for img in img_ids:
            
                img = cv2.imread(img)
                output.write(img)
                
                
                
            # update values for next iteration 
        
        
            vid_id = vid_id + 1 
    
    




def main (file_paths_frames, file_paths_video, out_file, segmentation_type) : 
    
    if segmentation_type == '1' : 
        
        video_segment_type1(file_paths_video, out_file)
        print(file_paths_video)
        
        
    else :
    
        video_segment_type2(file_paths_frames, out_file)

    
    
        
    
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='video segmentation')
    
    parser.add_argument('--file_paths_frames', metavar='path',
                        help='the path to the folder that contains folders for all video frames .. ')
    
    parser.add_argument('--file_paths_video', metavar='path',
                        help='the path to the folder that contains for all video in .mp4 format .. ')
    
    parser.add_argument('--out_file', metavar='path', required=True,
                        help='the path where the segments will be saved ..')
    
    parser.add_argument('--segmentation_type', metavar='path', required=True,
                        help='segmentation type 1 or 2  ..')
    
    
    
    
    
    args = parser.parse_args()
    
    if args.segmentation_type == '1' : 

        file_paths_video = glob.glob(args.file_paths_video+"/*")  
        file_paths_frames = ''  
        
    else :
        
    	file_paths_frames = glob.glob(args.file_paths_frames+"/*") 
    	file_paths_video = ''
    
    out_file = args.out_file
    

    segmentation_type = args.segmentation_type
    print(" call main")
    main(file_paths_frames, file_paths_video, out_file, segmentation_type)
    
    
    
