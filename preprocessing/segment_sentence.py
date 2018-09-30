#encoding=utf-8

# 將小說切成句子(multi-processing版)和斷詞還有替換英文和數字
# novel >> novel_sentence2

#修改replace function把非中文非ENG非NUM非標點符號的詞都替換為UNK

import json
import os
import logging
import unicodedata
import re
from tqdm import tqdm
from opencc import OpenCC
import time
import multiprocessing
import jieba
# from udicOpenData.dictionary import *

tStart = time.time() 
cpus = 11
openCCs2twp = OpenCC('s2twp')
jieba.load_userdict('NameDict_Ch_v2')

def mp_worker(filepath):

    def jiebaSegment(sentence):
        def _filter_junk_word(seg_list):
            """過濾斷詞後不要的東西"""
            clean_seg = []
            for word in seg_list:
                if word != ' ':
                    clean_seg.append(word)
            return clean_seg

        lineTemp = _filter_junk_word(jieba.cut(sentence, cut_all=False))
        lineTemp = (" ".join(lineTemp))
        return lineTemp

    def replace(sentence):
        english_re = re.compile(r'[A-Za-z]')
        punctuation_re = re.compile(r'["!！?？｡＂#＃$＄%％^&＆＇（）*＊()_+|}{":~!@/*-?<>＋,，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."]')
        
        def check_contain_chinese(check_str):
            for ch in check_str:
                if u'\u4e00' <= ch <= u'\u9fff':
                    pass
                else:
                    return False
            return True

        lineTemp = []
        for word in sentence.split(" "):
            if  word.isdigit():
                lineTemp.append('__NUM__')
            elif re.match(english_re, word):
                lineTemp.append('__ENG__')
            elif re.match(punctuation_re, word):
                lineTemp.append(word)
            elif check_contain_chinese(word):
                lineTemp.append(word)
            else:
                lineTemp.append('__UNK__')
        lineTemp = (" ".join(lineTemp))
        return lineTemp



    sentenceTemp = ''

    #傳進來的filepath是一個檔案(一本小說)
    with open(filepath,'r') as load_f:
        load_dict = json.load(load_f)
        count = 0
        for chapter in load_dict['內容']:
            # count += 1
            #簡轉繁
            chapter = openCCs2twp.convert(chapter)
            #以。?!:"「」【】‘’()”“ 作為切割點切割出句子
            chapter = re.split('[。!?:"【】「」()”“ ]', unicodedata.normalize('NFKC', chapter)) #「」【】‘’()
            for sentence in chapter:
                sentence = sentence.replace('.','').replace('-','').replace('‧','').replace('—','')\
                                   .replace('~','').replace('#','').replace('@','').replace('*','')\
                                   .replace('＊','').replace('＃','')   #.replace('”','').replace('“','')~
                if sentence != "":
                    sentence = jiebaSegment(sentence)
                    sentence = replace(sentence)
                    sentenceTemp = sentenceTemp+sentence+'\n'
            # if count == 10:
            #     break
            # break
    return sentenceTemp,filepath 

def mp_handler():
    # cpus = multiprocessing.cpu_count()
    #設定要用幾顆CPU
    p = multiprocessing.Pool(cpus)

    #輸出在哪個資料夾下
    newdir = 'novel_sentence4'
    if not os.path.isdir(newdir):
        os.mkdir(newdir)


    #將要處理的檔案路徑放進listX，並把輸出路徑放在outputpath
    listX = []
    outputpath = []
    for subdir in os.listdir('novel'):
        if subdir != '非武俠':
            continue
        sentencePath = os.path.join(newdir, subdir+'_seg.txt')
        # sentenceTXT = open(sentencePath,'w')
        outputpath.append(sentencePath)
        subdirPath = os.path.join('novel', subdir)
        for file in os.listdir(subdirPath):
            # if file != '3160.json':
            #     continue
            filePath = os.path.join(subdirPath, file)
            listX.append(filePath)


    #將輸出檔案一次性打開
    outputfile = [open(path, 'w') for path in outputpath]

    #p.imap(mp_worker, listX)多工處理，將回傳的資料不斷寫進相對應的檔案
    for result in tqdm(p.imap(mp_worker, listX), total = len(listX)):
        for i in range(len(outputpath)):
            #判斷這個檔案是哪個類別的，把它跟output的路徑比較
            if result[1].split('/')[-2] in outputpath[i]:
                outputfile[i].write(result[0])

    #一次全部關閉
    for i in outputfile:
        i.close()



if __name__=='__main__':
    mp_handler()

tEnd = time.time()
print(tEnd - tStart)