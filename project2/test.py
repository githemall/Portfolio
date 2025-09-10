import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 훈련용 데이터와 테스트용 데이터를 불러옵니다.
# '\t'는 데이터가 탭(tab)으로 구분되어 있다는 의미입니다.
train_df = pd.read_csv('projects/Portfolio/project2/ratings_train.txt', sep='\t')
test_df = pd.read_csv('projects/Portfolio/project2/ratings_test.txt', sep='\t')

# 데이터가 어떻게 생겼는지 상위 5개만 출력해봅니다.
print(train_df.head())

# 훈련용 데이터의 전체적인 정보를 확인합니다.
print(train_df.info())

# 긍정(1) 리뷰와 부정(0) 리뷰가 각각 몇 개인지 확인합니다.
print(train_df['label'].value_counts())

# 1. 결측치 확인: 'document' 열에 비어있는 값이 있는지 확인합니다.
print('결측치 확인 전:', train_df.isnull().sum())

# 2. 결측치 제거: 비어있는 행(row)을 제거합니다.
train_df = train_df.dropna(subset=['document'])
print('결측치 확인 후:', train_df.isnull().sum())


# 3. 중복 데이터 확인: 중복된 리뷰가 있는지 확인합니다.
print('중복 데이터 개수:', train_df['document'].nunique(), '/', len(train_df))

# 4. 중복 데이터 제거: 중복된 리뷰를 제거합니다.
train_df.drop_duplicates(subset=['document'], inplace=True)
print('중복 제거 후 개수:', len(train_df))

# 각 리뷰의 길이를 계산해서 새로운 'length' 열을 추가합니다.
train_df['length'] = train_df['document'].apply(len)

# 리뷰 길이의 통계 정보를 확인합니다. (평균, 표준편차, 최대/최소 등)
print(train_df['length'].describe())

# 리뷰 길이 분포를 히스토그램으로 시각화합니다.
plt.hist(train_df['length'], bins=50)
plt.xlabel('length of review')
plt.ylabel('number of review')
plt.show()