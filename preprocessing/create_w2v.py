# -*- coding: utf-8 -*-
#製作pre-train word embedding檔，用於模型輸入，使embedding layer初始化

from gensim.models import word2vec
from gensim import models
import logging
import json
import numpy as np
import pickle

id2wordPath = '../novel_sentence5/step1_id2word'
#gensim的word2vec模型載入
model = models.Word2Vec.load('med250.model.bin')

with open(id2wordPath, 'r') as id2wordFile:
    id2word = json.load(id2wordFile)
    id2word = sorted(id2word.items(), key=lambda item:int(item[0]), reverse = False)
    print(type(id2word))
    count = 0
    count_unk = 0
    vectorList = []
    for k in id2word:
        try:
            vectorList.append(model[k[1]])
        except:
            print(k[1])
            #若該字於gensim的word2vec model裡面不存在則給予一隨機向量
            vectorList.append(np.random.normal(loc = 0,size = 200))
            count_unk += 1
        count+=1


    with open("../novel_sentence5/gensim_5W_2.txt", "wb") as fp:
        pickle.dump(vectorList, fp) 

    print(count)
    print(count_unk)