#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 00:14:58 2022

@author: shahad
"""


import glob 
import pandas as pd


test = glob.glob('/home/shahad/AICITY2022_Track3_Team95-main/slowfast_Inference/data/00000/*')

test.sort()


# Test 

rows_list = []

for row in test : 
    
    
    rows_list.append(['/workspace/'+row.split('shahad/')[1], 0])

    
df_test = pd.DataFrame(rows_list)      


df_test.to_csv("/home/shahad/AICITY2022_Track3_Team95-main/slowfast_Inference/data/00000/test.csv", index=False, header = False)



