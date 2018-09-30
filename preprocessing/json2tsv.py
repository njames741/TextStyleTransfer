# -*- coding: utf8 -*-
#將原本是json的id2word做成TSV檔以作為embedding layer視覺化的輸入檔案
import os
import logging
import re
import string
from tqdm import tqdm
import csv
import json


jsonPath = "novel_sentence5/step1_id2word"
tsvPath = "novel_sentence5/step1_id2word.tsv"

with open(jsonPath,'r') as jsonFile:
    with open(tsvPath,'w') as tsvFile:
        tsvWriter = csv.writer(tsvFile, delimiter='\t')
        id2word = json.load(jsonFile)
        id2word = sorted(id2word.items(), key=lambda item:int(item[0]), reverse = False)
        tsvWriter.writerow(["word","id"])
        print(type(id2word))
        for i in id2word:
            tsvWriter.writerow([i[1],i[0]])