import os.path
import pandas as pd
import numpy as np
import sklearn
import sys
from pandas.io.common import file_exists

DATA_PATH = '../data/tech_docs.csv'

# 기능1 데이터 불러오기 (CSV 파일 불러와 DataFrame 반환)
def load_data(data_path):
    print("===== 기능1 - 데이터 불러오기 =====")
    # file 존재 여부 확인
    file_exists_flag = os.path.exists(data_path)
    # 파일 있는 경우, DataFrame 반환
    if file_exists_flag:
        df = pd.read_csv('../data/tech_docs.csv', encoding="utf-8-sig")
        print("데이터 로드 완료: {row_num} x {col_num}".format(row_num=df.shape[0], col_num=df.shape[1]))
        return df
    else:
        print("파일이 없습니다. 프로그램을 종료합니다.")
        sys.exit()

# 기능2 데이터 구조 확인(행/열 수, 컬러명, 자료형, 상위 5행 출력)
def explore_structure():

    print("===== 기능2 - 데이터 구조 확인 =====")
    df = load_data(DATA_PATH)
    print("행 수: {row_num}, 열 수: {col_num}".format(row_num=df.shape[0], col_num=df.shape[1]))



    for i in range(len(df.columns)):

        col_name = df.columns[i]
        col_type = df.dtypes.iloc[i]
        print(col_name, col_type)
        print("컬럼명: {col_name}, 자료형: {col_type}".format(col_name=col_name, col_type=col_type))

# 카테고리별 문서 수 및 평균 단어수 계산/출력
# def show_category_distribution():


# 컬럼별 결측치 수/비율 계산 및 심각도 출력
# def check_missing()

# NumPy로 문서 길이 통계량 직접 계산/출력
# def numpy_doc_stats()

# 함수 호출
def main():
    # load_data(DATA_PATH)
    explore_structure()

if __name__ == "__main__":
    main()