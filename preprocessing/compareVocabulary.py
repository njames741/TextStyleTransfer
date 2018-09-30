# -*- coding: utf8 -*-
#比對原本size為5萬的vocabulary和size為2W5的武俠vocabulary，將兩者同樣擁有的word製作成新的vocab(decoder_word2id)
import os
import logging
import re
import string
from tqdm import tqdm
import csv
import json

word2idFile = 'novel_sentence5/step1_word2id'
wuxia_word2idFile = 'novel_sentence5/JINYONG_word2id'

word2id = json.load(open(word2idFile,'r'))
wuxia_word2id = json.load(open(wuxia_word2idFile,'r'))

decoder_word2idFile = 'novel_sentence5/decoder_word2id'
decoder_id2wordFile = 'novel_sentence5/decoder_id2word'

decoder_word2id = {}
decoder_id2word = {}

count = 0

for wk,wv in tqdm(wuxia_word2id.items()):
    for k,v in word2id.items():
        if wk == k:
            count += 1
            decoder_word2id[k] = v
            decoder_id2word[v] = k

with open(decoder_word2idFile, 'w') as out:
    json.dump(decoder_word2id, out, ensure_ascii=False)
with open(decoder_id2wordFile, 'w') as out:
    json.dump(decoder_id2word, out, ensure_ascii=False)

print(count)