from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
import pandas as pd
import re

# 크롬 드라이버 옵션 설정
options = webdriver.ChromeOptions()
# options.add_argument('headless')  # 브라우저 안보임 / 잘 동작하는지 확인한 후 마지막에 실행할 것
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


def get_news_title(section_id: 'target section url number',
                   section_name: 'target section name',
                   start_num: 'target section first page number',
                   end_num: 'target section last page number',
                   ):
    # 크롤링한 페이지 저장할 데이터프레임
    url = f'https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1={section_id}#&date=%2000:00:00&page='
    df_section_titles = pd.DataFrame()

    # 인수로 받은 마지막 페이지까지 크롤링
    for i in range(start_num, end_num + 1):
        try:
            driver.get(url + str(i))  # 해당 url로 크롬 브라우저 이동

            # xpath로 요소 찾음
            titles = driver.find_elements_by_xpath('//*[@id="section_body"]/ul/li/dl/dt[last()]/a')
            # 찾은 요소의 텍스트에서 특수문자, 문장부호, 숫자 제거
            titles = [re.compile('[^가-힣|a-z|A-Z ]+').sub(' ', title.text) for title in titles]
            df = pd.DataFrame(titles, columns=['title'])

            # 페이지별 찾은 제목들을 한곳에 저장
            df_section_titles = pd.concat([df_section_titles, df], ignore_index=True)

            # 현재 페이지 번호
            current_page = driver.find_element_by_xpath('//*[@id="paging"]/strong').text
            # 다음 페이지 버튼 클릭
            nextpage_btn = driver.find_element_by_xpath('//*[@id="paging"]/strong/following-sibling::a[1]')
            nextpage_btn.click()

            # 진행상황 확인
            print(current_page, len(df), '/', end=' ')
            if i % 50 == 0:
                print('\n')

        # 해당 요소가 없을 시 발생하는 에러 핸들링
        except NoSuchElementException:
            print('NoSuchElementException')

    # 만든 데이터프레임에 label 추가 후 리턴
    print('\n')
    df_section_titles['category'] = section_name
    return df_section_titles


# 뉴스 카테고리
category = ['Politics', 'Economic', 'Social', 'Culture', 'World', 'IT']
# 각 카테고리별 가져올 페이지 수
end_nums = [334, 423, 400, 87, 128, 74]
# 크롤링한 데이터 저장할 데이터프레임
df_titles = pd.DataFrame()

# 모든 카테고리 크롤링
# for i, cat in enumerate(zip(category, end_nums)):
#     i += 100
#     print(i, cat[0], cat[1])  # 섹션 ID, 섹션 이름, 페이지 수
#     df_titles = pd.concat([df_titles, get_news_title(i, cat[0], 1, cat[1])])

# 단일 카테고리 크롤링
i = 1
cat = (category[1], end_nums[1])
i += 100
print(i, cat[0], cat[1])  # 섹션 ID, 섹션 이름, 페이지 수
df_titles = pd.concat([df_titles, get_news_title(i, cat[0], 1, cat[1])])

# 크롤링한 데이터 저장
df_titles.to_csv(
    f'./crawling_data/naver_eco_titles_allpage_{time.strftime("%Y-%m-%d", time.localtime(time.time()))}.csv',
    index=False)

# 크롬 드라이버 종료
driver.close()
