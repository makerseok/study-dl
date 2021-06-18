import pandas as pd
import numpy as np
import pickle

from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 데이터 로드
df = pd.read_csv('./data/raw_11535_2021-06-18.csv')

# 중복값 제거
df.drop_duplicates(subset=['title'], inplace=True)
df.reset_index(drop=True, inplace=True)

# feature, target 분리
X = df.drop('category', axis=1)
Y = df[['category']]

# one-hot encoding
ohe = OneHotEncoder(sparse=False)
onehot_y = ohe.fit_transform(Y)
label = ohe.categories_
print(label)
print(onehot_y)

# encoder 저장
with open('./data/category_encoder.pickle', 'wb') as f:
    pickle.dump(ohe, f)

# 데이터 형태소 분석
okt = Okt()
X['summary'] = X['summary'].apply(okt.morphs)

# stopword 로드
stopwords = pd.read_csv('../datasets/stopwords.csv', index_col=0)
print(stopwords)

# 함수로 만들어 기존 데이터에 apply해 stopword 제거
def delete_stopwords(lst):
    words = []
    for word in lst:
        if word not in list(stopwords['stopword']) and len(word) > 1:
            words.append(word)
    return ' '.join(words)

X['summary'] = X['summary'].apply(delete_stopwords)

print(X.head())
