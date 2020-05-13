'''
컴퓨터정보통신공학전공 175704 박지연
ㅋ의 길이 = 3을 기준으로 ㅋ의 길이를 통일하고자 함
길이 3 미만일 경우 ㅋ 시퀀스 소거
길이 3 이상일 경우 ㅋ 시퀀스의 길이 3으로 통일
'''

import math, glob
import csv
from soynlp.word import WordExtractor

def read_data():
    raw_time = []
    raw_chat = []
    for i in glob.glob("preprocessed/*.csv"):
        f = open(i,"r",encoding='utf-8')
        raw = csv.reader(f)
        for line in raw:
            raw_time.append(line[0])
            raw_chat.append(line[1])
        f.close()
    return raw_time, raw_chat

word_extractor = WordExtractor(
    min_frequency=200,
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
) #여기서는 Cohesion Score 사용

raw_time, raw_chat = read_data()
word_extractor.train(raw_chat)
words = word_extractor.extract()

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
