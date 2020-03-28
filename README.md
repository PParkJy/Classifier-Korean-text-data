# Classifier-Korean-text-data
한국어 텍스트 데이터에 대한 이진 분류기 만들기

**1. 데이터셋**

Naver Sentiment Movie Corpus(NSMC)

&nbsp;&nbsp; : https://github.com/e9t/nsmc/  
&nbsp;&nbsp; : 네이버 영화 리뷰 데이터를 긍정(0)/부정(1)으로 라벨링한 균형적인 텍스트 데이터
  
  
**2. 데이터 토큰화(Tokenization)**  

soynlp  

&nbsp;&nbsp; : https://github.com/lovit/soynlp  
&nbsp;&nbsp; : 단어 추출(Extract) 및 토큰화 사용  
&nbsp;&nbsp; : 통계적으로 단어의 경계를 나누고 이를 기반으로 토큰화
  
  
**3. 데이터 전처리**  

BOW(Bag Of Words)  

&nbsp;&nbsp; : 단어들의 순서는 전혀 고려하지 않고 단어의 출혀 빈도에만 집중하는 텍스트 데이터의 수치화 표현 방법
&nbsp;&nbsp; : 각 단어에 고유 인덱스를 부여하고 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터(Vector)를 생성
&nbsp;&nbsp; : NLTK에서 제공하느 기능 사용
  
  
**4. 학습 및 평가**  

MLP 사용, epoch 10, batch size 500 기준  

&nbsp;&nbsp; : loss = 0.80683..
&nbsp;&nbsp; : accuracy = 0.795  
&nbsp;&nbsp; : 현재 상황에서 loss가 높으므로 학습을 진행하면 더 좋은 성능이 나올 것이라 판단됨
