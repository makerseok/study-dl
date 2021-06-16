from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
import re

# 크롬 드라이버 옵션 설정
options = webdriver.ChromeOptions()
#options.add_argument('headless')
options.add_argument('--no-sandbox')  # 리눅스 환경에서 필요
options.add_argument('disable-gpu')  # 리눅스 환경에서 필요
options.add_argument('--disable-dev-shm-usage')  # 안주면 에러남
options.add_argument('lang=ko_KR')  # 언어 설정
driver = webdriver.Chrome('../chromedriver', options=options)  # 크롬 드라이버 실행
driver.implicitly_wait(10)  # 드라이버가 페이지 로드를 기다리는 시간
url = 'https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page=1'

category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']
page_num = [334, 423, 400, 87, 128, 74]
df_title = pd.DataFrame()

for l in range(1,2):
    df_section_title = pd.DataFrame()
    for k in range(101,201):
        url = 'https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=10{}#&date=%2000:00:00&page={}'.format(l,k)
        driver.get(url)  # 해당 url로 크롬 브라우저 이동
        time.sleep(0.5)
        title_list = []
        for j in range(1,5):
            for i in range(1,6):
                try:
                    title = driver.find_element_by_xpath(  # xpath로 요소 찾음
                        '//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[last()]/a'.format(j,i)
                    ).text  # 찾은 요소의 텍스트만 저장
                    title = (re.compile('[^가-힣|a-z|A-Z]').sub(' ', title))  # 특수문자, 문장부호, 숫자 제거
                    print(title)
                    title_list.append(title)
                except NoSuchElementException:
                    print('NoSuchElementException')

    # 수집한 제목들로 데이터프레임 생성
    df_section_title = pd.DataFrame(title_list)
    df_section_title['category'] = category[l]
    df_title = pd.concat([df_title, df_section_title], axis=0,
                         ignore_index=True)
driver.close()
df_title.head(30)

# 데이터프레임 저장
df_title.to_csv('./crawling_data/naver_news_titles_20210615_1_100.csv')

# //*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a
# //*[@id="section_body"]/ul[1]/li[2]/dl/dt[2]/a
# //*[@id="section_body"]/ul[1]/li[3]/dl/dt[2]/a
# //*[@id="section_body"]/ul[1]/li[4]/dl/dt[2]/a
# //*[@id="section_body"]/ul[1]/li[5]/dl/dt[2]/a
# //*[@id="section_body"]/ul[2]/li[1]/dl/dt[2]/a
# //*[@id="section_body"]/ul[2]/li[2]/dl/dt[2]/a


