'''
컴퓨터정보통신공학전공 175704 박지연
ㅋ의 길이 = 3을 기준으로 ㅋ의 길이를 통일하고자 함
길이 3 미만일 경우 ㅋ 시퀀스 소거
길이 3 이상일 경우 ㅋ 시퀀스의 길이 3으로 통일
'''

import math
import csv
from soynlp.word import WordExtractor

file_name = './data/non_highlight/79match1/399807785_'

def read_data(filename):
    raw_time = []
    raw_chat = []
    for i in range (1,28):
        fname = filename + str(i) + ".csv"
        f = open(fname, 'r', encoding='euc-kr')
        raw = csv.reader(f)
        for line in raw:
            raw_time.append(line[1])
            raw_chat.append(line[2])
        f.close()
    return raw_time, raw_chat


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
    return trans_raw

raw_time, raw_chat = read_data(file_name)
trans_chat = laugh_trans(raw_chat)
print(trans_chat)

word_extractor = WordExtractor(
    min_frequency=20,
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
) #여기서는 Cohesion Score 사용

word_extractor.train(trans_chat)
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
