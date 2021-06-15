from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
import re

# 크롬 드라이버 옵션 설정
options = webdriver.ChromeOptions()
# options.add_argument('headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable-gpu')
options.add_argument('lang=ko_KR')

# 드라이버 경로
chromedriver = '../chromedriver.exe'

# selenium webdriver에 크롬 드라이버 연동
driver = webdriver.Chrome(chromedriver, options=options)
driver.implicitly_wait(10)

url = 'https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=101#&date=%2000:00:00&page=1'

# 뉴스 카테고리
category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']

def get_news_title(section_id, section_name, end_num):
    # 크롤링한 페이지 저장할 데이터프레임
    url = f'https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1={section_id}#&date=%2000:00:00&page='
    df_section_titles = pd.DataFrame()
    for i in range(1, end_num + 1):
        try:
            driver.get(url + str(i))
            titles = driver.find_elements_by_xpath('//*[@id="section_body"]/ul/li/dl/dt[2]/a')

            titles = [re.compile('[^가-힣|a-z|A-Z ]+').sub(' ', title.text) for title in titles]
            df = pd.DataFrame(titles, columns=['title'])
            df_section_titles = pd.concat([df_section_titles, df], ignore_index=True)

            current_page = driver.find_element_by_xpath('//*[@id="paging"]/strong').text

            nextpage_btn = driver.find_element_by_xpath('//*[@id="paging"]/strong/following-sibling::a[1]')
            nextpage_btn.click()

            print(current_page, len(df), '/', end=' ')
            if i % 50 == 0:
                print('\n')
        except NoSuchElementException:
            print('NoSuchElementException')

    print('\n')
    df_section_titles['category'] = section_name
    return df_section_titles


end_nums = [334, 423, 400, 87, 128, 74]  # 각 카테고리별 가져올 페이지 수
df_titles = pd.DataFrame()
# for i, cat in enumerate(zip(category, end_nums)):
#     i += 100
#     print(i, cat[0], cat[1])  # 섹션 ID, 섹션 이름, 페이지 수
#     df_titles = pd.concat([df_titles, get_news_title(i, cat[0], cat[1])])
i = 1
cat = (category[1], end_nums[1])
i += 100
print(i, cat[0], cat[1])  # 섹션 ID, 섹션 이름, 페이지 수
df_titles = pd.concat([df_titles, get_news_title(i, cat[0], cat[1])])
df_titles.to_csv(f'./crawling_data/naver_eco_titles_allpage_{time.strftime("%Y-%m-%d", time.localtime(time.time()))}.csv')