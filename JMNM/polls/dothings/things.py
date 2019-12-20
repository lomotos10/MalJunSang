def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        # data = data[1:]
    return data

train_data = read_data('polls/dothings/data/data_train.txt')
test_data = read_data('polls/dothings/data/data_test.txt')

from konlpy.tag import Okt

okt = Okt()
print(okt.pos(u'이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))

import json
import os
from pprint import pprint

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('polls/dothings/data/train_docs.json'):
    with open('polls/dothings/data/train_docs.json') as f:
        train_docs = json.load(f)
    with open('polls/dothings/data/test_docs.json') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[0]), row[1]) for row in train_data]
    test_docs = [(tokenize(row[0]), row[1]) for row in test_data]
    # JSON 파일로 저장
    with open('polls/dothings/data/train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('polls/dothings/data/test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")

# 예쁘게(?) 출력하기 위해서 pprint 라이브러리 사용
pprint(train_docs[0])

tokens = [t for d in train_docs for t in d[0]]

import nltk
text = nltk.Text(tokens, name='NMSC')

selected_words = [f[0] for f in text.vocab().most_common(10000)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]

import numpy as np

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')

y_test = (y_test-1) / 4
y_train = (y_train-1) / 4

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import numpy as np
from sklearn.metrics import roc_auc_score

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(7861,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

model.fit(x_train, y_train, epochs=100, batch_size=512)

results = model.evaluate(x_test, y_test)

def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    return score

predict_pos_neg("무시하지 마라 이놈아 ")
predict_pos_neg("내가 나가기 싫은게 아니라 어쩔 수 없었움... 진짜...")
predict_pos_neg("예측 가능? 나는 불가능 ")
predict_pos_neg("남의 물건을 이렇게 함부로 써도 되는거...? 나 진짜 너무 당황스럽다...")
predict_pos_neg("남는 시간을 쪼개서 하는 모습.. 므시땅~")
predict_pos_neg("구라도 정도껏쳐야지 ㅋㅋㅋ 누가 믿어")
