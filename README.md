# Tensorflow Unsupervised Text Style Transfer

<img width="80%" height="80%" src="https://github.com/njames741/TextStyleTransfer/blob/master/Architecture.png"/>

## 環境
+	Ubuntu 16.04.3 LTS
+	Python 3.5.2
+	Tensorflow 1.6.0
+	CUDA Version 9.0.176

## 說明
如架構圖，模型訓練分為兩階段  
第一階段使用兩風格混合語料進行訓練
第二階段使用單一風格語料進行訓練

## 資料預處理步驟

1.	segment_sentence.py斷句斷詞並替換英文和數字
2.	wordCount.py統計詞頻
3.	createword2id.py建立詞彙表
4.	replaceUNK.py替換OOV並進行資料清洗
5.	splitdata.py切出訓練資料與測試資料
6.	json2tsv.py製作projector視覺化需要的metadata
7.	creat_w2v.py製作pre-train word embedding檔

## 執行

訓練模型

	python train.py --model_dir=model path

開啟tensorboard

	tensorboard --logdir=model path

使用模型

	python predict.py --model_dir=model path

## 待解決

+	偶發性一開始訓練LOSS就是nan，需把model砍掉重開新訓練(先嘗試換model資料夾路徑，以及換GPU跑)
+	遇到EOS不會馬上停止輸出單詞，會再接續輸出幾個EOS，目前觀察是不影響結果