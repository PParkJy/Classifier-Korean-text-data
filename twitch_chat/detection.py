import csv
from keras.models import model_from_json
import json
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor
import nltk
import numpy as np

jsonPath = "middle.json"
weightsPath = "middle.h5"

with open(jsonPath,'r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights(weightsPath)
print("모델을 불러옵니다.")

def read_data(f_path):
    raw_time = []
    raw_chat = []
    f = open(f_path,"r",encoding='euc-kr')
    raw = csv.reader(f)
    for line in raw:
        raw_time.append(line[1])
        raw_chat.append(line[2])
    return raw_time, raw_chat

word_extractor = WordExtractor(
    min_frequency=100,
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
)

def laugh_trans(raw_chat):
    trans_raw = []
    for chat in raw_chat:
        laugh_len = chat.count("ㅋ")
        if laugh_len: #ㅋ이 있다면
            #ㅋ이 3개 미만이면 소거, 3개 이상이면 3개로 통일
            idx = 0
            laugh_cnt = 0
            chat2 = list(chat)
            chat = chat + "_"
            for i in chat:
                if i == "ㅋ":
                    laugh_cnt += 1
                else:
                    if laugh_cnt > 0 and laugh_cnt < 3:
                        for j in range(idx - laugh_cnt, idx):
                            chat2[j] = "*"
                    else:
                        for j in range(idx - laugh_cnt, idx-3):
                            chat2[j] = "*"
                    laugh_cnt = 0
                idx += 1
            chat2 = ''.join(list(filter(("*").__ne__, chat2)))
            laugh_len = chat2.count("ㅋ")
            chat2 = chat2.replace("ㅋ", "", laugh_len - 3)
            trans_raw.append(chat2)
        else:
            trans_raw.append(chat)
    return trans_raw

raw_time, raw_chat = read_data("399807785.csv")
raw_chat = laugh_trans(raw_chat)

word_extractor.train(raw_chat)
test_words = word_extractor.extract()
test_score = {word:score.cohesion_forward for word, score in test_words.items()}
tokenizer = LTokenizer(scores=test_score)
test_list = []
cnt = 0
for sent in raw_chat:
    test_list.append([tokenizer.tokenize(sent)])
    cnt += 1

test_tokens = [token for data in test_list for token in data[0]]

test_text = nltk.Text(test_tokens)
selected_tokens= [t[0] for t in test_text.vocab().most_common(500)]
def term_frequency(data):
    return [data.count(word) for word in selected_tokens]
test_x = [term_frequency(d) for d in test_list]
X_test = np.asarray(test_x).astype('float32')

cnt = 0
print(len(model.predict_classes(X_test))) #13374
for i in range(len(X_test)):
    if model.predict_classes(X_test)[i] == [0]:
        #print(raw_time[i],"   , highlight")
        cnt += 1
        print(cnt)

