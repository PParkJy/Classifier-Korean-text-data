'''
2020.03.28
#한국어 데이터 이진 분류기
#데이터 -> 네이버의 영화 리뷰 데이터
#토큰화 -> soyNLP
#벡터화 -> BOW
#학습 -> MLP with Keras
'''

import csv
import glob
import json

from sklearn.model_selection import train_test_split
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import nltk
import numpy as np
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', baseline=0.75)

#데이터 읽어오기
def read_train():
    raw_data = []
    for i in glob.glob("preprocessed/*.csv"):
        f = open(i, 'r', encoding='utf-8')
        raw = csv.reader(f)
        for line in raw:
            raw_data.append([line[0],line[1],line[2]])
        f.close()
    return sorted(raw_data)

def read_test(f_path):
    f = open(f_path, 'r', encoding='euc-kr')
    raw_time = []
    raw_chat = []
    f = open(f_path,"r",encoding='euc-kr')
    raw = csv.reader(f)
    for line in raw:
        raw_time.append(line[1])
        raw_chat.append(line[2])
    return raw_time, raw_chat

#일단 라벨과 텍스트를 보고 하이라이트인지 아닌지를 판단
#일단 나누는데 오름차순으로 일단 정렬하고, 그 중 뒤에서 30% 정도를 테스트로 쓰자

raw_data = read_train()
x_test, test_time = read_test("399807785.csv")

x_train = []
y_train = []
for i in raw_data:
    x_train.append(i[1])
    y_train.append(i[2])

#x_data = []
#_data = []

#for i in raw_data:
#    x_data.append(i[1])
#    y_data.append(i[2])

#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

word_extractor = WordExtractor(
    min_frequency=150,
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
)

word_extractor.train(x_train)
train_words = word_extractor.extract()
train_score = {word : score.cohesion_forward for word, score in train_words.items()}
tokenizer = LTokenizer(scores=train_score)
train_list = []
cnt = 0
for sent in x_train:
    train_list.append([tokenizer.tokenize(sent),y_train[cnt]])
    cnt += 1

word_extractor.train(x_test)
test_words = word_extractor.extract()
test_score = {word:score.cohesion_forward for word, score in test_words.items()}
tokenizer = LTokenizer(scores=test_score)
test_list = []
cnt = 0
for sent in x_test:
    test_list.append([tokenizer.tokenize(sent)])
    cnt += 1

train_tokens = [token for data in train_list for token in data[0]]
test_tokens = [token for data in test_list for token in data[0]]

train_text = nltk.Text(train_tokens)
test_text = nltk.Text(test_tokens)

print('=====================selecting token======================') #시간 개오래걸림;
selected_tokens_1 = [t[0] for t in train_text.vocab().most_common(500)] #출현 빈도가 높은 상위 10000개의 토큰 선택
selected_tokens_2 = [t[0] for t in test_text.vocab().most_common(500)]
#벡터화 -> BOW(Bag of Words)
def term_frequency1(data):
    return [data.count(word) for word in selected_tokens_1]

def term_frequency2(data):
    return [data.count(word) for word in selected_tokens_2]

train_x = [term_frequency1(d) for d, _ in train_list]
test_x = [term_frequency2(d) for d in test_list]
train_y = [l for _, l in train_list]

X_train = np.asarray(train_x).astype('float32')
X_test = np.asarray(test_x).astype('float32')
Y_train = np.asarray(train_y).astype('float32')

#모델 설정
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(lr=0.00001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
model.fit(X_train, Y_train, epochs=500, batch_size=50, callbacks=[es])

model_json = model.to_json()
with open("middle.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights('middle.h5')
print("모델을 저장했습니다.")

print("===========evaluating with test data=============")
results = model.evaluate(x=X_test,y=None)
print("loss: ", results[0])
print("accuracy: ", results[1])
