# -*- coding: utf8 -*-
#清洗segment_sentence.py處理過的資料，依照word2id表將不再dict中的字詞替換成__UNK__，
#接著有英文的砍掉，有UNK的砍掉，長度大於30和小於3的砍掉，非中文詞占比超過30%砍掉

import os
import logging
import re
import string
import json
from tqdm import tqdm

inputFile = 'novel_sentence5/JINYONG_seg5.txt'

outputFile = 'novel_sentence5/JINYONG_seg5_clear.txt'

word2idFile = 'novel_sentence5/decoder3_word2id_compress'

clearCount = 0
usefulCount = 0
count = 0

#判斷該詞是否有包含非中文的字
def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            pass
        else:
            return False
    return True

with open(word2idFile,'r') as w2idFile:
    w2id = json.load(w2idFile)
    with open(inputFile,'r') as inFile:
        with open(outputFile,'w') as outFile:
            for line in tqdm(inFile.readlines()):
                line=line.strip('\n')
                sentenceTemp = []
                UNKcount = 0
                sentenceList = line.split(" ")
                notchinese = 0

                for word in sentenceList:
                    if word.isspace():
                        continue
                    if word not in w2id:
                        sentenceTemp.append("__UNK__")
                        UNKcount += 1
                    else:
                        sentenceTemp.append(word)

                for i in sentenceTemp:
                    if check_contain_chinese(i):
                        pass
                    else:
                        notchinese+=1

                sentenceLen = len(sentenceTemp)   

                if sentenceLen > 30 or sentenceLen < 3:
                    clearCount += 1
                    continue
                elif "__ENG__" in sentenceTemp:
                    clearCount += 1
                    continue
                elif UNKcount >= 1:
                    clearCount += 1
                    continue
                elif (notchinese/sentenceLen) > 0.2:
                    clearCount += 1
                    continue           
                else:
                    outFile.write(" ".join(sentenceTemp)+'\n')
                    usefulCount +=1


print("clearCount: ", clearCount)
print("usefulCount: ", usefulCount)