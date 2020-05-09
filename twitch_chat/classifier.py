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
def read_data():
    raw_data = []
    for i in glob.glob("preprocessed/*.csv"):
        f = open(i, 'r', encoding='utf-8')
        raw = csv.reader(f)
        for line in raw:
            raw_data.append([line[0],line[1],line[2]])
        f.close()
    return sorted(raw_data)

#일단 라벨과 텍스트를 보고 하이라이트인지 아닌지를 판단
#일단 나누는데 오름차순으로 일단 정렬하고, 그 중 뒤에서 30% 정도를 테스트로 쓰자

raw_data = read_data()

x_data = []
y_data = []

for i in raw_data:
    x_data.append(i[1])
    y_data.append(i[2])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

word_extractor = WordExtractor(
    min_frequency=100,
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
    test_list.append([tokenizer.tokenize(sent),y_test[cnt]])
    cnt += 1

train_tokens = [token for data in train_list for token in data[0]]
test_tokens = [token for data in test_list for token in data[0]]

train_text = nltk.Text(train_tokens)
test_text = nltk.Text(test_tokens)

print('=====================selecting token======================') #시간 개오래걸림;
selected_tokens = [t[0] for t in train_text.vocab().most_common(1000)] #출현 빈도가 높은 상위 10000개의 토큰 선택

#벡터화 -> BOW(Bag of Words)
def term_frequency(data):
    return [data.count(word) for word in selected_tokens]

train_x = [term_frequency(d) for d, _ in train_list]
test_x = [term_frequency(d) for d, _ in test_list]
train_y = [l for _, l in train_list]
test_y = [l for _, l in test_list]

X_train = np.asarray(train_x).astype('float32')
X_test = np.asarray(test_x).astype('float32')
Y_train = np.asarray(train_y).astype('float32')
Y_test = np.asarray(test_y).astype('float32')

#학습 시작
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(lr=0.00001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
model.fit(X_train, Y_train, epochs=500, batch_size=50, callbacks=[es])

print("===========evaluating with test data=============")
results = model.evaluate(X_test, Y_test)
print("loss: ", results[0])
print("accuracy: ", results[1])
