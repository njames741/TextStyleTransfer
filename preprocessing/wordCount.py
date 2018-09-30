# -*- coding: utf8 -*-

#用於統計詞頻

import os
import logging
import re
import string
from tqdm import tqdm
import csv

inputPath = 'novel_sentence5/step1_seg5.txt'
outputPath = 'novel_sentence5/step1_seg5_wordcount.csv'

vocabulary  = {}

def saveCSV(wordCount):
    with open(outputPath, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for i in wordCount:
            spamwriter.writerow([i[0], i[1]])

with open(inputPath,'r') as inFile:
    count = 0
    for line in tqdm(inFile.readlines()):
        count += 1
        for word in line.split():
            if word in vocabulary:
               vocabulary[word] += 1
            else:
               vocabulary[word] = 1
    wordCount = [(k, vocabulary[k]) for k in sorted(vocabulary, key=vocabulary.get, reverse=True)]

    print(len(wordCount))
    saveCSV(wordCount)
