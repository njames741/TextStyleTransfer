import os
import random
import json
from tqdm import tqdm
import jieba
import numpy as np
import unicodedata


padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3
#jieba詞典，也可以使用自己的詞典
jieba.load_userdict('../NameDict_Ch_v2')

class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []

#載入資料，並利用word2id表將文字序列轉成word id序列，如[1,2,3,4,5,6]
#intputFile為欲處理的資料(句子)，Number為資料數量參數(要處理多少筆資料)
def loadDataset(intputFile, word2idFile, Number=None):
    print("load data......")
    data = []
    sentenceCount = 0

    with open(word2idFile,'r') as w2idFile:
        w2id = json.load(w2idFile) #w2id = dict
        with open(intputFile, 'r') as infile:
            for line in tqdm(infile.readlines()):
                line=line.strip('\n') #去除結尾的換行符號
                sentenceCount += 1
                sentenceTemp = []
                for word in line.split(" "):
                    if word == '\n':
                        continue
                    try:
                        sentenceTemp.append(w2id[word])
                    except Exception as e:
                        print(" ")
                        print(e)
                        print("Error: "+line)
                data.append(sentenceTemp)

                #資料數量到達設定值則break
                if sentenceCount == Number:
                    break

    #data的格式為一句一句的word id序列，例如[[1,2,3],[3,5,8],[9,55,1]]
    return data

#為輸入的句子添加雜訊，詳情請查詢Denoising Autoencoder
def add_noise(sentence, prob):
        #複製list
        sentence = sentence.copy()

        omit_prob = prob
        #將要忽略的概率乘上句子長度即是要忽略的單詞個數
        num_omissions = int(omit_prob * len(sentence))
        #建立一list紀錄要忽略的單詞index
        inds_to_omit = np.random.permutation(len(sentence))[:num_omissions]
        #依序將要忽略的單詞更換為unknownToken
        for i in inds_to_omit:
            sentence[i] = unknownToken

        swap_prob = prob
        #將要交換順序的概率乘上句子長度-1即是要交換的單詞個數
        num_swaps = int(swap_prob * (len(sentence)))
        #因為是與後一個單詞交換順序，最後一個單詞沒辦法與後一個單詞交換，固-1避開最後一個index
        inds_to_swap = np.random.permutation(len(sentence) - 1)[:num_swaps]
        #依序將要交換的單詞與後一個單詞交換順序
        for i in inds_to_swap:
            sentence[i], sentence[i+1] = sentence[i+1], sentence[i]


        return sentence


def change_target(sentence, id2word, target_word2id):
    changed_sentence = []
    #逐詞將id用id2word表換回word再用target_word2id換成目標詞彙表id，若沒有該詞彙就換成unknownToken
    for i in sentence:
        changed_sentence.append(target_word2id.get(id2word[str(i)], unknownToken))

    return changed_sentence


def createBatch(samples, prob, id2word, target_word2id, only_wuxia=False, only_normal=False, noise=False):
    #建立batch的資料格式
    batch = Batch()
    #若noise=True就呼叫add_noise()添加雜訊
    if noise:
        inputSentences = [add_noise(sample, prob) for sample in samples]  
    else:
        inputSentences = samples

    #若only_wuxia或only_normal=True就呼叫change_target，將label句子利用target_word2id更換id
    if only_wuxia or only_normal:
        targetsSentences = [change_target(sample, id2word, target_word2id) for sample in samples]
    else:
        targetsSentences = samples     

    #紀錄該batch中的每個句子長度，decoder_targets_length因label句子有額外添加EOS所以長度+1
    batch.encoder_inputs_length = [len(inputSentence) for inputSentence in inputSentences]
    batch.decoder_targets_length = [len(targetsSentence)+1 for targetsSentence in targetsSentences]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    #將輸入句子反轉(聽說不使用雙向LSTM的話，將輸入句子反轉有不錯的效果)，並將句子padding到同樣的長度
    for sample in inputSentences:
        source = list(reversed(sample))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

    #將標籤句子加入EOS，並且一樣將batch中每個句子padding到同樣的長度
    for sample in targetsSentences:
        target = sample + [eosToken]
        pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)

    return batch


def getBatches(data, batch_size, prob, id2word, target_word2id, only_wuxia=False, only_normal=False, noise=False):
    '''
    呼叫getBatches以獲得一個batch的資料
    data為資料，格式如loadDataset()所產生的data
    batch_size是每個批次中所含的資料數量
    prob為添加雜訊的概率
    id2word為id轉word表
    target_word2id為預測時所使用的目標word轉id表，與word2id不同
    only_wuxia參數決定是否要將輸出的目標詞彙限縮在僅有武俠詞彙
    only_normal參數決定是否要將輸出的目標詞彙限縮在僅有白話詞彙
    noise參數決定是否需要將輸入句子添加雜訊
    '''

    #打亂資料後，迴圈將資料依照batch_size大小做成一個一個batch
    random.shuffle(data)
    data_len = len(data)
    for i in range(0, data_len, batch_size):
        batch = createBatch(data[i:min(i + batch_size, data_len)], prob, id2word, target_word2id, only_wuxia, only_normal, noise)
        yield batch

#此function用於predict時，將使用者輸入的句子轉換為模型可以吃的輸入格式
def sentence2enco(sentence, word2id, prob, id2word, target_word2id, only_wuxia=False, only_normal=False, noise=False):
    #去除最後的換行符號
    sentence = sentence.strip('\n')
    #統一全形半形
    sentence = unicodedata.normalize('NFKC', sentence)
    if sentence == '':
        return None
    #分词
    tokens = list(jieba.cut(sentence, cut_all=False))
    #將單詞轉為ID
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    #呼叫createBatch建立batch，就是一個batch size為一的batch
    print("wordIds:", wordIds)
    batch = createBatch([wordIds], prob, id2word, target_word2id, only_wuxia, only_normal, noise)
    return batch