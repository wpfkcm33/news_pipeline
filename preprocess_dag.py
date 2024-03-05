# 표준 라이브러리
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from dotenv import load_dotenv
# 서드파티 라이브러리
import psycopg2
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from transformers import AutoTokenizer
import pandas as pd
from sqlalchemy.types import Boolean
import pytz
import boto3

# Airflow 관련 임포트
from airflow import DAG
from airflow.hooks.postgres_hook import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

load_dotenv()
kst = pytz.timezone('Asia/Seoul')
train_script_path = '/opt/airflow/scripts/bert_finetune_text_classification.py'

# 기본 인자 정의
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 25),
    # 'end_date': datetime(2024, 3, 1),
    'email': ['your_email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

category_map = {
    '100': '정치',
    '101': '경제',
    '102': '사회',
    '103': '생활문화',
    '104': '세계',
    '105': 'IT과학',
    '106': '연예',
    '107': '스포츠'
}

def clean_title(text):
    """뉴스 제목에서 불필요한 패턴과 문자를 제거합니다."""
    text = re.sub(r'\[.*?\]', '', text)  # 대괄호 안의 텍스트 제거
    text = re.sub(r'[\'"“”‘’]', '', text)  # 모든 종류의 따옴표(일반, 특수) 제거
    return text.strip()  # 양쪽 공백 제거

def get_category_from_url(url):
    # URL에서 sid 값을 추출
    sid = url.split('sid=')[1] if 'sid=' in url else None
    return category_map.get(sid, '기타')  # sid가 매핑 딕셔너리에 없는 경우 '기타' 카테고리 반환

# 크롤링 함수
def crawl_data(execution_date):
    # 변수 초기화 및 설정
    query = Variable.get("news_query", default_var="뉴스")
    execution_date = datetime.strptime(execution_date, '%Y-%m-%d')
    execution_date = execution_date.astimezone(kst)
    crawl_start_date = (execution_date - timedelta(days=1)).strftime('%Y.%m.%d')
    end_date = crawl_start_date
    sort_type = "0"
    max_page = 100
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0'
    }
    
    # 데이터베이스 연결 설정
    hook = PostgresHook(postgres_conn_id="postgre_newsdata")
    conn = hook.get_conn()
    cur = conn.cursor()
    
    # 필터링 대상 단어 리스트
    filter_words = ["톱뉴스", "헤드라인", "클로징", "조간", "석간", "오프닝", "주요뉴스"]

    for page in range(1, max_page + 1):
        start = 1 + (page - 1) * 10
        url = f"https://search.naver.com/search.naver?where=news&query={query}&sm=tab_opt&sort={sort_type}&photo=0&field=0&reporter_article=&pd=3&ds={crawl_start_date}&de={end_date}&docid=&nso=so:r,p:from{crawl_start_date}to{end_date},a:all&mynews=0&refresh_start=0&related=0&start={start}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        for article in soup.find_all('div', {'class': 'news_area'}):
            raw_title = article.find('a', {'class': 'news_tit'}).get('title')

            # 원본 제목에서 필터링 대상 단어가 포함되어 있는지 확인
            if any(word in raw_title for word in filter_words):
                continue  # 필터링 대상 단어가 포함되어 있다면 나머지 코드를 실행하지 않고 다음 반복으로 넘어감

            # 필터링 조건에 해당하지 않는 경우에만 제목을 정제
            title = clean_title(raw_title)
            link = article.find('a', {'class': 'news_tit'}).get('href')
            summary = article.find('a', {'class': 'api_txt_lines dsc_txt_wrap'}).get_text()
            naver_url = ""
            press_company = article.find('a', {'class': 'info press'}).get_text(strip=True) if article.find('a', {'class': 'info press'}) else "출처를 찾을 수 없습니다."
            for a_tag in article.find_all('a', {'class': 'info'}):
                if a_tag.get('href').startswith('https://n.news.naver.com'):
                    naver_url = a_tag.get('href')
                    break

            # naver_url 중복 확인
            cur.execute("SELECT EXISTS(SELECT 1 FROM raw_data WHERE naver_link = %s)", (naver_url,))
            is_duplicate = cur.fetchone()[0]

            if naver_url and not is_duplicate:  # naver_url이 존재하고 중복되지 않는 경우만 데이터베이스에 등록
                article_response = requests.get(naver_url, headers=headers)
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                content_area = article_soup.find(id='dic_area')
                news_content = content_area.get_text(strip=True).replace("'", "''") if content_area else "본문을 찾을 수 없습니다."              
                date_text_element = article_soup.find('span', class_='media_end_head_info_datestamp_time')
                date_text = date_text_element.get('data-date-time') if date_text_element else None
                category = get_category_from_url(naver_url)

                # 데이터베이스에 저장
                cur.execute("""
                    INSERT INTO raw_data (date, title, summary, link, naver_link, content, press, category)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (date_text, title, summary, link, naver_url, news_content, press_company, category))
                print(f"Inserted: {title}")

        conn.commit()

    cur.close()
    conn.close()


def preprocess_batch(batch, tokenizer):
    # 배치 데이터에 대한 전처리 수행
    batch['title_tokens'] = batch['title'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    batch['content_tokens'] = batch['content'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    return batch

    return batch
# 데이터 전처리 함수
def preprocess_data():
    # PostgreSQL 연결 엔진 생성
    database_url = os.getenv('DATABASE_URL')
    engine = create_engine(database_url)

    # 데이터베이스에서 크롤링한 데이터 로딩
    sql_query = "SELECT * FROM raw_data"
    df = pd.read_sql(sql_query, engine)
    
    # 중복 제거 (고유 식별자 'id'를 기준으로)
    df = df.drop_duplicates(subset=['id'])

    # 카테고리 필터링을 위한 레이블 매핑
    label_mapping = {
        '0': 'IT과학',
        '1': '경제',
        '2': '사회',
        '3': '생활문화',
        '4': '세계',
        '5': '스포츠',
        '6': '정치'
    }

    # 매핑된 카테고리 이름을 리스트로 변환
    category_names = list(label_mapping.values())
    
    # 해당 카테고리에 속하는 레코드만 필터링
    df_filtered = df[df['category'].isin(category_names)]

    # 토크나이저 초기화 및 배치 크기 정의
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    batch_size = 100

    # 배치로 나누어 처리
    for start in range(0, len(df_filtered), batch_size):
        end = start + batch_size
        batch = df_filtered[start:end]
        
        # 배치에 대해 전처리 수행
        processed_batch = preprocess_batch(batch, tokenizer)

        # 'preprocessed_data' 테이블에서 이미 존재하는 'naver_link'을 조회
        existing_links = pd.read_sql("SELECT naver_link FROM preprocessed_data", engine)['naver_link'].tolist()

        # 중복되지 않는 레코드만 필터링
        filtered_batch = processed_batch[~processed_batch['naver_link'].isin(existing_links)]

        # 필터링된 데이터를 'preprocessed_data' 테이블에 삽입
        filtered_batch.to_sql('preprocessed_data', engine, if_exists='append', index=False)



def store_and_upload_data(**kwargs):
    current_time_str = datetime.now(kst).strftime('%Y%m%d%H%M%S')
    database_url = os.getenv('DATABASE_URL')
    engine = create_engine(database_url)
    s3_client = boto3.client('s3')
    bucket_name = 'ldw-news-train'  # S3 버킷 이름


    # 1단계: preprocessed_data 테이블에서 데이터 추출
    query = """
    SELECT id, category,title,content
    FROM preprocessed_data;
    """
    df = pd.read_sql(query, engine)

    # 2단계: 추출된 데이터를 training_data 테이블에 저장 (예시로, training_data 테이블이 이미 존재한다고 가정)
    # 'if_exists' 옵션을 'replace'로 설정하여, 기존 테이블을 새 데이터로 대체
    df.to_sql('training_data', engine, if_exists='replace', index=False)

    # 3단계: 데이터를 CSV 파일로 로컬에 저장
    # CSV 파일명에 현재 시간을 포함하여, 각 파일이 유니크하게 관리될 수 있도록 함
    file_name = f"training_data_{current_time_str}.csv"
    df.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    # 4단계: 로컬에 저장된 CSV 파일을 S3에 업로드
    s3_client.upload_file(file_name, bucket_name, file_name)
    
    

    
# DAG 정의
with DAG(
    'data_processing_pipeline',
    default_args=default_args,
    description='An example pipeline that crawls, preprocesses, and stores news data',
    schedule_interval='@daily',
    catchup=True, #backfill 사용시 리소스 사용주의
) as dag:
    
    crawl_task = PythonOperator(
        task_id='crawl_data',
        python_callable=crawl_data,
        op_kwargs={'execution_date': '{{ ds }}'},
        dag=dag,
    )

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        dag=dag,
    )
    
    store_and_upload_data_task = PythonOperator(
    task_id='store_and_upload_data',
    python_callable=store_and_upload_data,
    dag=dag,
    )

    crawl_task >> preprocess_data_task >> store_and_upload_data_task
