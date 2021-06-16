import pandas as pd

# dataframe print 이쁘게
pd.set_option('display.unicode.east_asian_width', True)

# 데이터 로드
df_0 = pd.read_csv('./crawling_data/naver_news_titles_210616_0.csv', index_col=0)
df_1 = pd.read_csv('./crawling_data/naver_news_titles_210616_1.csv')
df_2 = pd.read_csv('./crawling_data/naver_news_titles_210616_2.csv', index_col=0)
df_3 = pd.read_csv('./crawling_data/naver_news_titles_210616_3.csv', index_col=0)
df_4 = pd.read_csv('./crawling_data/naver_news_titles_210616_4.csv', index_col=0)
df_4.drop('Unnamed: 0.1', axis=1, inplace=True)
df_4.columns = ['title', 'category']
df_5 = pd.read_csv('./crawling_data/naver_news_titles_210616_5.csv', index_col=0)
df_headline = pd.read_csv('./crawling_data/naver_news_titles_210616_headline.csv', index_col=0)

df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_headline], axis=0, ignore_index=True)

print(df.head())
print(df.shape)

df.to_csv('./crawling_data/naver_news_titles_210616.csv')