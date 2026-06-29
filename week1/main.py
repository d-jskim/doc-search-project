import os.path
import pandas as pd
import numpy as np
import sklearn
import sys
from pandas.io.common import file_exists

DATA_PATH = '../data/tech_docs.csv'

# 기능1 데이터 불러오기 (CSV 파일 불러와 DataFrame 반환)
def load_data(data_path):
    print("+++++ 기능1 - 데이터 불러오기 +++++")
    # file 존재 여부 확인
    file_exists_flag = os.path.exists(data_path)
    # 파일 있는 경우, DataFrame 반환
    if file_exists_flag:
        df = pd.read_csv('../data/tech_docs.csv', encoding="utf-8-sig")
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

    # 문서 로드
    df = load_data(DATA_PATH)
    # 카테고리 리스트
    cat_lst = df["category"].unique()

    # ## 문1) 카테고리별 문서 수
    cat_df = pd.DataFrame({"category": cat_lst})
    cnt_df = df["category"].value_counts().to_frame(name="문서수")
    p_df = df["category"].value_counts(normalize=True).to_frame("문서비율(%)") * 100

    res_df = pd.concat([cnt_df, p_df], axis=1)
    print("===== 카테고리별 문서 수 =====")
    print(res_df)


    ## 문2) 카테고리별 평균 단어 수 (딕셔너리 활용)
    word_cnt_dict = {}
    for cat in cat_lst:
        total_word_cnt = 0
        df_cat = df[df["category"] == cat]
        for i in range (len(df_cat)):
            text = df_cat["content"].iloc[i]
            word_cnt = len(text.split())
            total_word_cnt += word_cnt
            avg_word_cnt = total_word_cnt / res_df.loc["Python","문서수"]

            word_cnt_dict[cat] = avg_word_cnt

    print("===== 카테고리별 평균 단어 수 (딕셔너리 반복문 출력) =====")
    for cat, avg_word_cnt in word_cnt_dict.items():
        print("{cat} - 평균 단어 수: {word_cnt}".format(cat=cat, word_cnt=avg_word_cnt))




    # 평균값 계산



    # print("===== 카테고리별 평균 단어 수 =====")



# 컬럼별 결측치 수/비율 계산 및 심각도 출력
# def check_missing()

# NumPy로 문서 길이 통계량 직접 계산/출력
# def numpy_doc_stats()

# 함수 호출
def main():
    # load_data(DATA_PATH)
    # explore_structure()
    show_category_distribution()

if __name__ == "__main__":
    main()