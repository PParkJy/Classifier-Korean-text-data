'''
2020.03.28
#한국어 데이터 이진 분류기
#데이터 -> 네이버의 영화 리뷰 데이터
#토큰화 -> soyNLP
#벡터화 -> BOW
#학습 -> MLP with Keras
'''

from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import nltk
import numpy as np

def read_data(filename):
    with open(filename,'r',encoding='UTF-8') as f:
        data = [line.split('\t')[1] for line in f.read().splitlines()]
        '''
        data = []
        for line in f.read().splitlines():
            data.append(line.split('\t')[1])
        '''
    data = data[1:] #id, document, label을 표시하는 행 제외
    return data

def read_label(filename):
    with open(filename,'r',encoding='UTF-8') as f:
        data = [line.split('\t')[2] for line in f.read().splitlines()]
    data = data[1:]
    return data

train_data = read_data('./ratings_train.txt')
test_data = read_data('./ratings_test.txt')
train_label = read_label('./ratings_train.txt')
test_label = read_label('./ratings_test.txt')

word_extractor = WordExtractor(
    min_frequency=100,
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
)

word_extractor.train(train_data)
train_words = word_extractor.extract()
train_score = {word : score.cohesion_forward for word, score in train_words.items()}
tokenizer = LTokenizer(scores=train_score)
train_list = []
cnt = 0
for sent in train_data:
    train_list.append([tokenizer.tokenize(sent),train_label[cnt]])
    cnt += 1

word_extractor.train(test_data)
test_words = word_extractor.extract()
test_score = {word:score.cohesion_forward for word, score in test_words.items()}
tokenizer = LTokenizer(scores=test_score)
test_list = []
cnt = 0
for sent in test_data:
    test_list.append([tokenizer.tokenize(sent),test_label[cnt]])
    cnt += 1

train_tokens = [token for data in train_list for token in data[0]]
test_tokens = [token for data in test_list for token in data[0]]

train_text = nltk.Text(train_tokens)
test_text = nltk.Text(test_tokens)

print('=====================selecting token======================') #시간 개오래걸림;
selected_tokens = [t[0] for t in train_text.vocab().most_common(10000)] #출현 빈도가 높은 상위 10000개의 토큰 선택

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
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(lr=0.00001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
model.fit(X_train, Y_train, epochs=10, batch_size=500)

print("===========evaluating with test data=============")
results = model.evaluate(X_test, Y_test)
print("loss: ", results[0])
print("accuracy: ", results[1])

'''
참고 사이트
https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
https://github.com/lovit/soynlp
'''
