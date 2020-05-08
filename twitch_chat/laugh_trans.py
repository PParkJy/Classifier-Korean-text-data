'''
<<<<<<< HEAD
컴퓨터정보통신공학전공 175704 박지연
=======
>>>>>>> 297c8b250bee0ab5b5d00d9b0c429d73325baa25
ㅋ의 길이 = 3을 기준으로 ㅋ의 길이를 통일하고자 함
길이 3 미만일 경우 ㅋ 시퀀스 소거
길이 3 이상일 경우 ㅋ 시퀀스의 길이 3으로 통일
'''

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor

<<<<<<< HEAD
file_name = './data/highlight/79매치_1경기/하이라이트/399807785_'
single_char = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ",
               "ㅣ", "ㅔ", "ㅐ", "ㅏ", "ㅜ", "ㅗ", "ㅓ", "ㅡ", "ㅟ", "ㅚ", "ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅒ", "ㅖ", "ㅘ", "ㅝ", "ㅙ", "ㅞ", "ㅢ"]
=======
file_name = './data/highlight/79매치_2경기/하이라이트/399807785_10.csv'
>>>>>>> 297c8b250bee0ab5b5d00d9b0c429d73325baa25

def read_data(filename):
    raw_time = []
    raw_chat = []
    for i in range (1,11):
        fname = filename + str(i) + ".csv"
        f = open(fname, 'r', encoding='euc-kr')
        raw = csv.reader(f)
        for line in raw:
            raw_time.append(line[1])
            raw_chat.append(line[2])
        f.close()
    return raw_time, raw_chat

def laugh_trans(raw_chat):
    trans_chat = []
<<<<<<< HEAD
    #이걸 먼저 ㅋ 단위로 분리하고....
    #근데 생각을 해봐라 ㅋ이 아니라 ㄱ이 들어갈 수도 있고 ㅁ이 들어갈 수도 있으니까 걍 모든 낱글자에 대해 처리를 해야하지 않냐?
    #-> 일단 낱글자에 대한 빈도수를 확인해보자
    for chat in raw_chat:
        laugh_len = chat.count("ㅋ")
        if laugh_len: #ㅋ이 있다면
            #ㅋ이 3개 미만이면 소거, 3개 이상이면 3개로 통일
            chat = chat.replace("ㅋ","",laugh_len-3)
        laugh_len = chat.count("ㅋ")
        if laugh_len > 3:
            print(chat)

=======
    for chat in raw_chat:
        laugh_len = chat.count("ㅋ")
        if laugh_len:
            #ㅋ이 3개 미만이면 소거, 3개 이상이면 3개로 통일
            chat = chat.replace("ㅋ","",laugh_len-3)
>>>>>>> 297c8b250bee0ab5b5d00d9b0c429d73325baa25
        trans_chat.append(chat)
    return trans_chat

def laugh_check(trans_chat):
    sentence_cnt = 0 #ㅋ이 들어간 문장의 수
    avg_prob = 0
    list_single = []

    for chat in trans_chat:
        laugh_len = chat.count("ㅋ")
        if laugh_len:
            sentence_cnt += 1 #ㅋ이 들어간 문장의 개수
            list_single.append(laugh_len) #한 문장 내에서의 ㅋ의 개수
            avg_prob += (laugh_len/len(chat))

    np_single = np.array(list_single)

    print("전체 chat 데이터 개수: " + str(len(trans_chat)) + "\n")
    print("전체 chat 중 ㅋ이 들어간 chat의 평균 비율: " + str(round(sentence_cnt / len(trans_chat) * 100, 3)) + "%\n")
    print("한 chat 내에서 ㅋ이 차지하는 평균 비율: " + str(round((avg_prob / sentence_cnt) * 100, 3)) + '%\n')
    print("ㅋ이 들어간 chat 중 ㅋ의 최소 길이: " + str(np.min(np_single)) + '\n')
    print("ㅋ이 들어간 chat 중 ㅋ의 최대 길이: " + str(np.max(np_single)) + '\n')
    print("ㅋ이 들어간 chat 중 ㅋ의 길이의 분산: " + str(round(np.var(np_single), 3)) + '\n')
    print("ㅋ이 들어간 chat 중 ㅋ의 평균 길이: " + str(round(np.mean(np_single),3)) + '\n')
    print("ㅋ이 들어간 chat 중 ㅋ의 길이의 중앙값: " + str(np.median(np_single)) + '\n')
    print("ㅋ이 들어간 chat 중 ㅋ의 길이의 최빈값(상위3개): " + str(Counter(np_single).most_common()[:3])+'\n') #상위 3개 확인

raw_time, raw_chat = read_data(file_name)
trans_chat = laugh_trans(raw_chat)
laugh_check(trans_chat)
<<<<<<< HEAD
print(trans_chat)
=======
>>>>>>> 297c8b250bee0ab5b5d00d9b0c429d73325baa25

word_extractor = WordExtractor(
    min_frequency=20,
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
) #여기서는 Cohesion Score 사용

<<<<<<< HEAD
word_extractor.train(trans_chat)
words = word_extractor.extract()
#print("word extraction 길이: ",len(words), " \n결과: ")
#print(words)
#words_score = {word : score.cohesion_forward for word, score in words.items()}
#tokenizer = LTokenizer(scores=words_score)

import math

def word_score(score):
    return (score.cohesion_forward * math.exp(score.right_branching_entropy))

print('단어   (빈도수, cohesion, branching entropy)\n')
for word, score in sorted(words.items(), key=lambda x:word_score(x[1]), reverse=True)[:10]:
    print('%s     (%d, %.3f, %.3f)' % (
            word,
            score.leftside_frequency,
            score.cohesion_forward,
            score.right_branching_entropy
            )
         )
=======
word_extractor.train(raw_chat)
words = word_extractor.extract()
print("word extraction 길이: ",len(words), " \n결과: ")
print(words)
#words_score = {word : score.cohesion_forward for word, score in words.items()}
#tokenizer = LTokenizer(scores=words_score)
>>>>>>> 297c8b250bee0ab5b5d00d9b0c429d73325baa25
