# -*- coding: utf-8 -*-

import logging
import warnings
import os
import shutil
import glob 
import time

import numpy as np
import math

import cv2

import torchvision #Used on non_max_suppression
import torch
from utils.torch_utils import select_device


def convert(xyxy):
   
    x1, y1 = xyxy[0], xyxy[1]
    w = int(xyxy[2]) - int(x1)
    h = int(xyxy[3]) - int(y1)
    return (x1,y1,w,h)


def init(video_path):
    #video_path = '/media/noura/Extreme SSD/AI city challane/2022/A1/user_id_35133/Rear_view_user_id_35133_NoAudio_0.MP4'
    cap = cv2.VideoCapture(video_path)
    full_rate =  rate = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    vid_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # width = vcap.get(3)
    # height = vcap.get(4)
    # fps = vcap.get(5)
    return full_rate, width, height, vid_length


def compute_IoU(box1, box2, x1y1x2y2=True,
                GIoU=False, DIoU=False, 
                CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter = max((min(b1_x2, b2_x2) - max(b1_x1, b2_x1)),0) * \
            max((min(b1_y2, b2_y2) - max(b1_y1, b2_y1)),0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou  # Iou

def crop_save_vehicle_info_video(vid_name,
                      vid_path,
                      car,
                      max_xyxy_list,
                      full_rate,
                      out_file = '/workspace/noura/yolov5/AIcity/A2_cropped/'):
   
    video_path = os.path.join(out_file, vid_name)
    if not os.path.exists(video_path) :
        os.mkdir(video_path)

    video_bbox_path =  os.path.join(video_path, "person_{:05d}".format(car['id']))
    if not os.path.exists(video_bbox_path):
        os.mkdir(video_bbox_path)
    
    out_file_name = os.path.join(video_bbox_path, '{:05d}.mp4'.format(car['id']))

    x, y, width, height = convert(max_xyxy_list)
    print("in crop_save_vehicle_info_video")
    # print("max_xyxy_list", max_xyxy_list)
    print("width, height", int(width), int(height) )
    
    output = cv2.VideoWriter(out_file_name, 
                             cv2.VideoWriter_fourcc(*'mp4v'), 
                             full_rate, 
                             (int(width), int(height)))
    

    # print("vid_path",vid_path)
    stream = cv2.VideoCapture(vid_path)
    counter = 0
    while (1):
        ret, frame = stream.read()
        if not ret:
             break
        frame = (frame[int(max_xyxy_list[1]):int(max_xyxy_list[3]),
                       int(max_xyxy_list[0]):int(max_xyxy_list[2])])
        #print("# width", frame.shape[1],frame.shape[0] )
        
        output.write(frame)
        frame = "{:05d}".format(counter)
        counter += 1
        if counter% 1000 == 0:
            print("counter",counter)
            
    # print("# width", frames[0].shape[1],frames[0].shape[0] )
    # print("before saving")
    # counter = 0
    # print("len(frames)", len(frames))

    print(car['id'], " box id saved " )

    return




car_id = 0

def vehicles_tube_construction( vid_name,
                                vid_path,
                                frame,
                                frame_id,
                                frame_predictions,
                                vehicles_tubes,
                                full_rate,
                                iou_threshhold =0.90,
                                miss_detection =7):
    miss_detection = 60
    global car_id 
    global saved 
    saved = 1
    indices_array = []
    # for each vehicle 
    for x1y1x2y2 in frame_predictions:
        if x1y1x2y2[5] !=0 or x1y1x2y2[4]<=0.50: ###class number 
            continue
        
        # flag for appending bbox == new car   False ==> not yet appended  True ==> appended
        append_flag = False

        # if there is pervious detected car check if the car is the same 
        if vehicles_tubes:
            for bbox_num in range(len(vehicles_tubes)):

                iou = compute_IoU(vehicles_tubes[bbox_num]['xyxy_list'][-1], x1y1x2y2)

                if iou >= iou_threshhold:
                 
                    vehicles_tubes[bbox_num]['xyxy_list'].append(x1y1x2y2.tolist())
                    vehicles_tubes[bbox_num]['frame_id'].append(frame_id)
                    vehicles_tubes[bbox_num]['miss_detection'] = miss_detection

                    indices_array.append(bbox_num)
                        
                    append_flag = True
                    # end -- searching for matching car has been completed

        # append new detected car 
        if not append_flag:
            vehicles_tubes.append({'id': car_id,
                                  'xyxy_list':[x1y1x2y2.tolist()],
                                  'frame_id': [frame_id],
                                  'miss_detection':miss_detection})
            indices_array.append(len(vehicles_tubes)-1)
            print('append new car ', car_id)
            car_id +=1

 
    # max_area = 0
    # max_xyxy_list = []
    # car = 0
    for index in reversed(range(len(vehicles_tubes))):
        # if there is pervious detected car check if the car is the same 
        if vehicles_tubes:
            for bbox_num in range(len(vehicles_tubes)):
                iou = compute_IoU(vehicles_tubes[bbox_num]['xyxy_list'][-1], x1y1x2y2)
                if iou >= iou_threshhold:
                    vehicles_tubes[bbox_num]['xyxy_list'].append(x1y1x2y2.tolist())
                    
    #     if index not in indices_array:
    #         vehicles_tubes[index]['miss_detection'] -= 1
    #         if vehicles_tubes[index]['miss_detection'] < 0:
    #             car = vehicles_tubes.pop(index)
    #             # print("pop")
    #             area = []
    #             for i in range(len(car['xyxy_list'])):
    #                 x, y, width, hight = convert(car['xyxy_list'][i])
    #                 area.append(width * hight)
                    
    #             tmp = max(area)
    #             indexx = area.index(tmp)   
    #             xyxy_list = car['xyxy_list'][indexx]
    #             # print("area",tmp )
    #             # print("max_area",max_area )
    #             if tmp > max_area:
    #                 max_area = tmp
    #                 max_xyxy_list = xyxy_list
    #                 car = car
    #             saved = 0
                
    # if saved == 0:          
    #     result = crop_save_vehicle_info_video(vid_name,vid_path, car, max_xyxy_list, full_rate)
    #     saved += 1

    return

def main(model, vid_name, video_path, weights):
    global saved 
    saved = 1
    full_rate, width, height, vid_length = init(video_path)
    
    count = 0
    print(vid_name, full_rate, width, height, vid_length )
    
    stream = cv2.VideoCapture(video_path)
    first_frame = True

    vehicles_tubes = [] 
    iou_threshhold = 0.30
    frame_id = 1
    
    while (1):
        
        ret, frame = stream.read()
        
        if not ret:
             break

        
        #model infrence function 
        predictions = model(frame)
        #for each  frame
        for frame_predictions in predictions.xyxy:
            vehicles_tube_construction( vid_name,
                                        video_path,
                                        frame,
                                        frame_id,
                                        frame_predictions,
                                        vehicles_tubes,
                                        full_rate = full_rate,
                                        iou_threshhold = iou_threshhold)

        count +=1
        frame_id +=1
        
        
        
        
    max_area = 0
    max_xyxy_list = []
    car = 0
    for index in reversed(range(len(vehicles_tubes))):
        car = vehicles_tubes.pop(index)
        # print("pop")
        area = []
        for i in range(len(car['xyxy_list'])):
                x, y, width, hight = convert(car['xyxy_list'][i])
                area.append(width * hight)

        tmp = max(area)
        indexx = area.index(tmp)
        xyxy_list = car['xyxy_list'][indexx]
        # print("area", tmp)
        # print("max_area", max_area)
        if tmp > max_area:
              max_area = tmp
              max_xyxy_list = xyxy_list
              car = car
        saved = 0
        
    if saved == 0:         
        result = crop_save_vehicle_info_video(vid_name,video_path, car, max_xyxy_list, full_rate)
        saved += 1

if __name__ == "__main__":
    # global car_id 
    #print("hi")
    #start = time.time()
    weights ='/yolov5/yolov5s.pt'
    model = torch.hub.load('yolov5','yolov5s', pretrained=True, source='local', force_reload=True).autoshape()
    device = select_device(0)
    model = model.to(device)
    videos_path = '/workspace/noura/yolov5/AIcity/A2/*/*'
   
    videos_files_path = glob.glob(videos_path)
    video_paths = []
    for video_path in videos_files_path:
                print(video_path)
                car_id = 0
                vid_name = str(os.path.basename(video_path))
                main(model, vid_name, video_path, weights)
        
                break
            
    # video_path = "/workspace/noura/Downloads/Rear_view_user_id_24491_0_0_No_11.mp4"    
    # car_id = 0
    # vid_name = str(os.path.basename(video_path))
                
    # main(model, vid_name, video_path, weights)
    
