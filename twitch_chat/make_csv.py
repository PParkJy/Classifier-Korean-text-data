import csv
from soynlp.normalizer import *
import glob

'''
1. 전처리한 데이터에 시간, 라벨을 붙여주고 다시 하나의 csv로 저장해줍시다.
    -> 라벨 = 0 -> 비하이라이트, 1 = 하이라이트
2. 
3.
4. 전체 데이터 중 3은 테스트, 7은 학습으로 써주자
'''

file_name = './data/highlight/79match1/'

def read_data():
    raw_time = []
    raw_chat = []
    for i in glob.glob(file_name+"*.csv"):
        f = open(i, 'r', encoding='euc-kr')
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
        else:
            trans_raw.append(chat)
    return trans_raw

raw_time, raw_chat = read_data()
pre_chat = laugh_trans(raw_chat)
save_data = []

label = 0
if file_name.split('/')[2] == "highlight":
    label = 1
for i in range(len(raw_time)):
    save_data.append([raw_time[i], pre_chat[i], label])

save_csv = open('preprocessed/'+ str(file_name.split('/')[3]) + "_" + str(file_name.split('/')[2])+".csv","w", encoding='utf-8', newline="")
wr = csv.writer(save_csv)
for i in save_data:
    wr.writerow(i)

'''
#In windows, csv.writerow() has an issue that add the empty line behind each line. So, set the newline="" option.
f = open('test.csv', 'w', encoding='utf-8', newline="")
wr = csv.writer(f)
wr.writerow([1, "Alice", True])
wr.writerow([2, "Bob", False])
f.close()
'''




