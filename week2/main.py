import os.path
import re

import pandas as pd
import numpy as np
import sklearn
import sys
from pandas.io.common import file_exists
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = '../data/tech_docs.csv'

# 기능1 데이터 불러오기 (CSV 파일 불러와 DataFrame 반환)
def load_data(data_path):
    # print("+++++ 기능1 - 데이터 불러오기 +++++")
    # file 존재 여부 확인
    file_exists_flag = os.path.exists(data_path)
    # 파일 있는 경우, DataFrame 반환
    if file_exists_flag:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        print("===== 데이터 로드 확인 ======")
        print("데이터 로드 완료: {row_num} x {col_num}".format(row_num=df.shape[0], col_num=df.shape[1]))
        return df
    else:
        print("파일이 없습니다. 프로그램을 종료합니다.")
        sys.exit()

# 기능2 데이터 구조 확인(행/열 수, 컬러명, 자료형, 상위 5행 출력)
def explore_structure():

    print("===== 기능2 - 데이터 구조 확인 =====")
    df = load_data(DATA_PATH)
    print("===== 데이터 구조 확인 ======")
    print("행 수: {row_num}, 열 수: {col_num}".format(row_num=df.shape[0], col_num=df.shape[1]))

    print("===== 데이터 컬럼명과 자료형 확인 =====")
    for i in range(len(df.columns)):

        col_name = df.columns[i]
        col_type = df.dtypes.iloc[i]
        print("컬럼명: {col_name}, 자료형: {col_type}".format(col_name=col_name, col_type=col_type))

    print("===== 상위 5개 행 출력 =====")
    print(df.head(5))

    print("===== info 함수 출력 =====")
    df.info()


# 카테고리별 문서 수 및 평균 단어수 계산/출력
def show_category_distribution():
    print("+++++ 기능3 - 카테고리 분포 확인 +++++")
    # 문서 로드
    df = load_data(DATA_PATH)
    # 카테고리 리스트
    cat_lst = df["category"].unique()

    # ## 문1) 카테고리별 문서 수
    cnt_df = df["category"].value_counts().to_frame(name="문서수")
    p_df = df["category"].value_counts(normalize=True).to_frame("문서비율(%)") * 100
    res_df = pd.concat([cnt_df, p_df], axis=1)
    print("===== 카테고리별 문서 수 =====")
    print(res_df)

    ## 문2) 카테고리별 평균 단어 수 (딕셔너리 활용)
    for i in range(len(df)):
        df.loc[i,"word_cnt"] = len(df.loc[i,"content"].split())

    word_avg_dict = {}
    for cat in cat_lst:
        word_avg_dict[cat] = df[df["category"] == cat]["word_cnt"].mean()

    print("===== 카테고리별 평균 단어 수 (딕셔너리 반복문 출력) =====")
    for cat, word_avg in word_avg_dict.items():
        print("{cat} - 평균 단어 수: {word_cnt}".format(cat=cat, word_cnt=word_avg))




# 컬럼별 결측치 수/비율 계산 및 심각도 출력
def check_missing():
    print("+++++ 기능4 - 결측치 현황 파악 +++++")
    # 데이터 로드
    df = load_data(DATA_PATH)

    # 결측치 수(isnull()의 True 값 SUM)
    sum_df = df.isnull().sum().to_frame("결측치 수") # True(결측치) 개수 구하기
    # 결측치 비율
    mean_df = df.isnull().mean().to_frame("결측치 비율")

    res_df = pd.concat([sum_df, mean_df], axis=1) #결측치 수, 결측치 비율

    column_lst = df.columns.tolist()
    for col in column_lst:
        # 결측치 여부
        if res_df.loc[col, "결측치 수"] != 0: res_df["결측치 여부"] = True
        else: res_df["결측치 여부"] = False
        # 심각도
        missing_rate = res_df.loc[col, "결측치 비율"]
        if missing_rate < 5 * 0.01: res_df.loc[col, "심각도"] = "낮음"
        elif missing_rate >= 5 * 0.01 and missing_rate < 20 * 0.01: res_df.loc[col, "심각도"] = "주의"
        else: res_df.loc[col, "심각도"] = "높음"

    print("===== 컬럼별 결측치 존재 여부 확인 =====")
    missing_cols = res_df[res_df["결측치 여부"] == True].index.tolist()
    no_missing_cols = res_df[res_df["결측치 여부"] == False].index.tolist()
    print("결측치 있는 컬럼: {missing_cols}".format(missing_cols=missing_cols))
    print("결측치 없는 컬럼: {no_missing_cols}".format(no_missing_cols=no_missing_cols))

    print("===== 컬럼별 결측치 현황 출력 =====")
    print(res_df[["결측치 수", "결측치 비율", "심각도"]])


def numpy_doc_stats():
    print("+++++ 기능5 - 통계량 계산 +++++")

    raw_df = load_data(DATA_PATH)

    # "content"행에 결측치 존재 -> 제거
    df = raw_df.dropna(subset=["content"])
    # "content" 단어 수 NumPy 배열
    np_array = np.array(df["content"].str.split().str.len())
    # 통계량
    res_mean = np.mean(np_array)
    res_std = np.std(np_array, ddof=1)
    res_median = np.median(np_array)
    res_min = np.min(np_array)
    res_max = np.max(np_array)
    res_under_50 = np_array[np_array < 50]

    print("===== content 컬럼 단어 수 - 통계량(NumPy 계산) =====")
    print("평균: ", res_mean)
    print("표준편차: ", res_std)
    print("중앙값: ", res_median)
    print("최솟값: ", res_min)
    print("최댓값: ", res_max)
    print("50 미만 값만 추출: ", res_under_50)


    print("===== content 컬럼 단어 수 - 통계량(describe함수) =====")
    pd_stats = pd.Series(np_array).describe()
    print(pd_stats)

    print("===== 통계량 비교(NumPy vs. describe()) =====")
    stats_df = pd.DataFrame (index = ['평균','표준편차','중앙값','최솟값','최댓값']
                             , columns = ['NumPy(직접계산)', 'Pandas(describe사용)'])
    stats_df["NumPy(직접계산)"] = [res_mean, res_std, res_median, res_min, res_max]
    stats_df["Pandas(describe사용)"] = pd_stats.loc[['mean', 'std', '50%', 'min', 'max']].to_list()
    print(stats_df)

def preprocess(text):

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def cosine_similarity_numpy(a, b):
    # 리스트 -> NumPy배열 변환 : 벡터 연산
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    # 벡터 내적 (A · B)
    dot_prod = np.dot(a, b)
    # 벡터 크기(norm) 및 벡터 norm의 곱 (|A| X |B|)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    norms_prod = a_norm * b_norm

    if norms_prod == 0:
        return 0.0
    else:
        cos_sim = dot_prod / norms_prod
        return cos_sim


def keyword_search(question:str, df:pd.DataFrame, top_k:int)->pd.DataFrame:
    """키워드 기반 Baseline 검색
       Args:
           question: 질문 문자열, df: raw df, top_k: 상위 K개 지정
       Returns:
           점수 높은 순으로 Top-K 문서 데이터프레임
    """
    # 질문 전처리
    # question_clean = preprocess(question)
    # question_set = set(question_clean)
    question_set = set(question.split())

    df["score"] = df["content_clean"].str.split().apply(set).apply(lambda x: x & question_set).apply(len)

    res_df = df.sort_values("score", ascending=False)
    # print(res_df[["title", "score"]])

    return res_df[["doc_id", "title", "category", "score"]].head(top_k)


def build_tfidf(df):
    """TF-IDF 벡터 행렬 생성
       Args: DataFrame (df: pd.DataFrame - 데이터 전처리 완료 ("content_clean" 컬럼))
       Returns: (행렬, vectorizer)
    """
    # 문서를 벡터 행렬로 변환 (TF-IDF: 흔한 단어의 가중치를 낮추고 희귀하지만 중요한 단어의 가중치를 높임)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=2, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["content_clean"])

    print(f"TF-IDF 행렬 크기: {tfidf_matrix.shape} | 사용된 단어 수: {tfidf_matrix.shape[1]}") #인덱스 0: 전체 문서(행) 수, 인덱스1: 고유 단어(열)

    return tfidf_matrix, tfidf_vectorizer

def tfidf_search(question, df, tfidf_matrix, tfidf_vectorizer, top_k) -> pd.DataFrame:

    # 질문: 전처리 -> 희소 행렬 -> NumPy배열(유사도 비교 arg)
    question_clean = preprocess(question)
    # 문장을 리스트에 담아서 vectorizer에 입력 (vectorizer는 여러 문서 입력을 기대함. 문장 1개를 리스트에 넣어서 전달)
    question_tfidf = tfidf_vectorizer.transform([question_clean]) # 결과: (1, 401)

    # args: tfidf_matrix - (60, 401) -> CSR 희소 행렬(Sparse Matrix) 형태, 메모리 효율 위해 0이 아닌 값들의 위치만 기억하고 압축한 상태.
    # tfidf_matrix를 toarray()로 60행 401열로 변환
    tfidf_array = tfidf_matrix.toarray()
    question_array = question_tfidf.toarray().flatten() # (401,)

    similarity_lst = []
    for row in tfidf_array:
        score = cosine_similarity_numpy(row, question_array)
        similarity_lst.append(score)

    df["similarity"] = similarity_lst

    # 상위 k개 (argsort()[::-1][:k] 사용
    top_k_indices = np.array(similarity_lst).argsort()[::-1][:top_k]
    res_df = df[["doc_id", "title", "category", "similarity"]].iloc[top_k_indices]

    return res_df



# 함수 호출
def main():
    # explore_structure()
    # show_category_distribution()
    # check_missing()
    # numpy_doc_stats()

    # df: 전처리 완료
    raw_df = load_data(DATA_PATH)
    #
    # dropna(): 결측치 행 삭제 / reset_index(): 결측치 행 삭제되면서 인덱스 중간 번호가 비는 문제 해결
    df = raw_df.dropna(subset=["content"]).reset_index(drop=True)
    df["content_clean"] = df["content"].apply(preprocess)
    print("전처리 완료: content_clean 컬럼 생성")

    # 코사인 유사도 계산
    # cosine_similarity_numpy(a, b)

    # 질문과 유사한 문서 검색
    # question = "gradient descent"
    question = "how does gradient descent work in machine learning"
    top_k = 3

    # Baseline 검색
    baseline_res = keyword_search(question, df, top_k)

    # TF-IDF 벡터 행렬 생성
    tfidf_matrix, tfidf_vectorizer = build_tfidf(df)

    # TF-IDF Top-k 검색
    tfidf_res = tfidf_search(question, df, tfidf_matrix, tfidf_vectorizer, top_k)

    # Baseline 검색 결과
    print("질문 : ", question)
    print("=== Keyword Baseline ===")
    print(baseline_res)
    # TF-IDF 검색 결과
    print("=== TF-IDF 검색 결과 ===")
    print(tfidf_res)

    # 결과 분석
    # + Baseline 검색은 "gradient", "descent"가 존재하는 문서를 검색해서 단어 교집합을 score로 구함.
    # -> "gradient", "descent" 단어의 빈도수, 문서에서 "gradient descent"를 어느 정도의 비중으로 다루는지와 무관하게 두 단어가 존재하면 score가 2가 최댓값이 됨.
    # -> "gradient descent"를 중요하게 다루지 않은 문서도 score 최댓값(2)일 수 있고, 이런 문서가 여러 개인 경우 Top 3 결과가 무의미함(관련 없는 문서도 상위).

    # + TF-IDF 검색은 단어빈도수에 따른 가중치를 계산('the'같은 단어는 낮게, 중요한 'gradient' 단어는 높게)하여 검색어와 유사도 구함.
    # -> 컴퓨터 관련 문서들 중에서 관련도가 높은 문서를 찾을 경우, 컴퓨터 문서에서 흔한 단어인 "algorithm"도 가중치를 낮추고 "gradient", "descent"와 관련성 있는 문서 도출 가능함.
    # -> Baseline 검색의 score가 교집합 개수(정수)라서 문서들 간 차별화가 없었다면, smilarity는 척도는 0~1사이의 실수로 문서간 유사도 정렬(랭킹)이 가능해서 관련 높은 문서 검색 가능함

if __name__ == "__main__":
    main()