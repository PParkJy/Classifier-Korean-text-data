import csv

raw = "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ챌린저스 개꿀잼 ㅋ"
raw2 = list(raw)
laugh_cnt = 0
raw_len = len(raw)

'''
list.index의 문제점
-> 같은 글자가 반복해서 있으면 반환하는 인덱스들이 가장 첫번째 요소의 인덱스로만 반환
-> 개짜증난다... 이거 때문에 1시간은 날린듯
'''

for i in raw:
    if i == "ㅋ":
        if laugh_cnt > 3:
            raw2[laugh_cnt-1] = ""
            print(laugh_cnt)
        laugh_cnt += 1
print(raw2)

'''
#ㅋ이 존재하는 곳의 원소를 바꾸면 되는 거 아닌가?????
for i in range(len(raw)):
    if s_raw[i] == "":
         em_cnt += 1
    elif s_raw[i] != "":
        if em_cnt < 3:
            temp = raw.replace("ㅋ","",em_cnt)
            print("temp1",temp)
        else:
            temp = raw.replace("ㅋ","",em_cnt-3)
            print("temp2",temp)
        em_cnt = 0
        raw = temp
    print("raw",raw)
    print(em_cnt)
'''



#In windows, csv.writerow() has an issue that add the empty line behind each line. So, set the newline="" option.
#f = open('test.csv', 'w', encoding='utf-8', newline=""#)
#wr = csv.writer(f)
#wr.writerow([1, "Alice", True])
#wr.writerow([2, "Bob", False])
#f.close()