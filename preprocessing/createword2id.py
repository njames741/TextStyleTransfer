# -*- coding: utf8 -*-
#製作word2id和id2word的表，存成JSON檔案
import os
import logging
import re
import string
from tqdm import tqdm
import csv
import json

#詞頻檔案路徑
wordCountFile = 'novel_sentence5/step1_seg5_wordcount.csv'
#輸出檔案路徑
word2idPath = 'novel_sentence5/step1_word2id'
id2wordPath = 'novel_sentence5/step1_id2word'

#詞彙表大小
vocabularySize = 50000

word2id = {'__PAD__': 0, '__GO__': 1, '__EOS__': 2, '__UNK__': 3, '__NUM__': 4, '__ENG__': 5}
id2word = {0: '__PAD__', 1: '__GO__', 2: '__EOS__', 3: '__UNK__', 4: '__NUM__', 5: '__ENG__'}
start = 6

with open(wordCountFile, 'r') as f:
    csvdata = csv.reader(f)
    csvList = list(csvdata)

    print(len(csvList))
    print(len(csvList[:vocabularySize]))
    for i in csvList[:vocabularySize]:
        if i[0] in word2id:
            continue
        word2id[i[0]] = start
        id2word[start] = i[0]
        start += 1


with open(word2idPath, 'w') as out:
    json.dump(word2id, out, ensure_ascii=False)
with open(id2wordPath, 'w') as out:
    json.dump(id2word, out, ensure_ascii=False)