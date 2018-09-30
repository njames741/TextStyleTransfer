# -*- coding: utf8 -*-
#切割train、test、validation

import os
import logging
import re
import string
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

inputFile = 'novel_sentence5/JINYONG_seg5_clear.txt'
trainFile = 'novel_sentence5/JINYONG_seg5_clear_train.txt'
testFile = 'novel_sentence5/JINYONG_seg5_clear_test.txt'
validationFile = 'novel_sentence5/JINYONG_seg5_clear_validation.txt'

alldata = []
with open(inputFile,'r') as inFile:
    for line in tqdm(inFile.readlines()):
        alldata.append(line)

train, validation = train_test_split(alldata,test_size=0.055)
# test, validation = train_test_split(tv,test_size=0.5)


with open(trainFile,'w') as trFile:
    for sentence in tqdm(train):
        trFile.write(sentence)


with open(validationFile,'w') as vaFile:
    for sentence in tqdm(validation):
        vaFile.write(sentence)
