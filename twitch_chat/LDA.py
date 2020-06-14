'''
topic modeling with Latent Direchlet Allocation
x axis = ?
y axis = ?
'''

import csv
import tomotopy as tp
import pyLDAvis
import math
from soynlp.word import WordExtractor
from soynlp.utils import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2
from soynlp.normalizer import *
from soynlp.tokenizer import MaxScoreTokenizer
import numpy as np

csvname = "./data/399807785_all.csv"

def read_csv(csvname):
    raw_time = []
    raw_chat = []
    f = open(csvname, 'r')
    raw = csv.reader(f)
    for line in raw:
        raw_time.append(str(line[1]).encode('utf-8').decode('utf-8'))
        raw_chat.append(emoticon_normalize(str(line[2]).encode('utf-8').decode('utf-8'), num_repeats=2))
    f.close()
    return raw_time, raw_chat

def word_score(score):
    #cohesion_forward * right_branching_entropy
    #주어진 글자가 유기적으로 연결되어 함께 자주 나타나며(cohesion_forward)
    #그 단어의 우측에 다양한 조사, 어미, 단어가 등장하여 단어 우측의 branching entropy가 높음 (right_branching_entropy)
    return (score.cohesion_forward * math.exp(score.right_branching_entropy))

#=================read csv data======================
raw_time, raw_chat = read_csv(csvname)
print(len(raw_time))
cnt = 0
for i in raw_chat:
    if "채팅방" in i:
        del raw_chat[cnt]
        del raw_time[cnt]
    cnt = cnt +1
print(len(raw_chat))
#=================Korean preprocessing===================
#====tokenize with soynlp======

#train word boundary
word_extractor = WordExtractor(min_frequency=100, min_cohesion_forward=0.05, min_right_branching_entropy=0.0)
#train = count the frequency of substrings
word_extractor.train(raw_chat)
#select word
words = word_extractor.extract()
print("\n추출된 단어의 개수:", len(words), "\n")

'''
#print all extracted word
for key, value in words.items():
    print(key, value)
'''

#print('단어   (빈도수, cohesion, branching entropy)\n')
#print the top n-th word
n = 30
extracted_words = []
for word, score in sorted(words.items(), key=lambda x:word_score(x[1]), reverse=True)[:n]:
    '''
    print('%s     (%d, %.3f, %.3f)' % (
            word, 
            score.leftside_frequency, 
            score.cohesion_forward,
            score.right_branching_entropy
            )
         )
    '''
    extracted_words.append(word)

cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
tokenizer = MaxScoreTokenizer(scores=cohesion_score)

#=================LDA trian strat========================
#Generate LDAModel
#k = the number of topic
#alpha = ?
#eta = ?
#min_cf = min frequency
model = tp.LDAModel(k=10, alpha=0.1, eta=0.01, min_cf=5)

for i in raw_chat:
    model.add_doc(tokenizer.tokenize(i))

#check the number of words, vocabulary
#prepare the train
model.train(0)
print('Total docs:', len(model.docs))
print('Total words:', model.num_words)
print('Vocab size:', model.num_vocabs)

#200times training
for i in range(200):
    print('Iteration {}\tLL per word: {}'.format(i, model.ll_per_word))
    model.train(1)
 
#print the trained topic
for i in range(model.k):
    #print the top10
    res = model.get_topic_words(i, top_n=5) #top_n = the number of word
    print('Topic #{}'.format(i), end='\t')
    print(', '.join(w for w, p in res))

#LDA visualization
topic_term_dists = np.stack([model.get_topic_word_dist(k) for k in range(model.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in model.docs])
doc_lengths = np.array([len(doc.words) for doc in model.docs])
vocab = list(model.used_vocabs)
term_frequency = model.used_vocab_freq

prepared_data = pyLDAvis.prepare(
    topic_term_dists, 
    doc_topic_dists, 
    doc_lengths, 
    vocab, 
    term_frequency
)

pyLDAvis.save_html(prepared_data, 'ldavis.html')
