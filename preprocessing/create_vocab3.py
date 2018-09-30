# -*- coding: utf8 -*-
#將decoder_word2id重新給值，使得ID從0~25000不跳號
import os
import logging
import re
import string
from tqdm import tqdm
import csv
import json

decoder_word2idFile = 'novel_sentence5/decoder_word2id'
decoder_id2wordFile = 'novel_sentence5/decoder_id2word'

decoder_word2id = json.load(open(decoder_word2idFile,'r'))
decoder_id2word = json.load(open(decoder_id2wordFile,'r'))

decoder_word2id_compress_path = 'novel_sentence5/decoder_word2id_compress'
decoder_id2word_compress_path = 'novel_sentence5/decoder_id2word_compress'

decoder_word2id_compress = {}
decoder_id2word_compress = {}

decoder_id2word = sorted(decoder_id2word.items(), key=lambda item:int(item[0]), reverse = False)

count = 0

for wk,wv in decoder_id2word:
    decoder_word2id_compress[wv] = count
    decoder_id2word_compress[count] = wv
    count+=1

with open(decoder_word2id_compress_path, 'w') as out:
    json.dump(decoder_word2id_compress, out, ensure_ascii=False)
with open(decoder_id2word_compress_path, 'w') as out:
    json.dump(decoder_id2word_compress, out, ensure_ascii=False)