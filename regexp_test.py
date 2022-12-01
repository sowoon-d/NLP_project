import common
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import time
import pickle
import itertools
import yaml
from gensim.models import FastText
import pandas as pd
import copy
from datetime import datetime
from functools import partial
import zipfile
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
import pyspark.sql.functions as fn

if __name__ == '__main__':
    start = time.time()
    today_time = datetime.today().strftime("%Y/%m/%d %H:%M:%S")
    print('start time : ', today_time)

    spark_conf = SparkConf().setAppName("swpark") \
        .set("spark.driver.maxResultSize", "8g") \
        .set("spark.executor.heartbeatInterval", "3500s") \
        .set("spark.network.timeout", "3600s") \
        .set('spark.driver.extraClassPath', 'tibero6-jdbc.jar')
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .enableHiveSupport() \
        .config(conf=spark_conf) \
        .appName("test") \
        .getOrCreate()
    data = [('▶T/G가니쉬 매칭 상단부 갭큰경향SPEC : 1.7ACT : 2.0~2.1▶O/S램프 매칭부 단차 꺼짐 RH -0.7'),
         ("(FBL-01)[시작1호차('19 6/3)]▶ 헤드램프가니시 측하단 매칭부 내부 노출 및 외관상 안좋음"),
         ("[품질평가1차]▷하이브리드 정보 관련 문제 - 유럽 1. 에너지 흐름도 및 배터리 SOC 표시 간헐적 "
          "안됨 (유럽) 2. 하이브리드 메뉴 표시 중 P → R → P단 변속 시 이전 하이브리드 화면이 아닌 "
          "홈 화면으로 이동됨 (유럽) 3. 홈 위젯 하이브리드 연비, 전기 모터 사용량 숫자 컬러 통일 필요 "
          "(내수) 4. 에너지 흐름도 라인 컬러 AVN과 클러스터간 매칭안됨 (내수) "
          "5. AVN 하이브리드 연비 그래프 리셋 시 팝업 문구 의미 불합리 (유럽) - "
          "'Factory setting successfully restored'[품질평가2차]1. 배터리 "
          "SOC간헐적 표시 안됨(유럽)3,4,5,번 문제점")]
    df = spark.createDataFrame(data, StringType()).toDF('content')

    df = df.withColumn('reg_content', fn.regexp_replace('content', '(\(|\[|\{|\<)', ' \( '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\)|\]|\}|/>)', '\ ) '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\,|\?|\!)', ' , '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\.)', ' . '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content',
                                                    '[^가-힣a-zA-Z0-9|/|~|%|^|_|→|←|↔|↑|↓|↕|±|℃|℉|Ø|Φ|\']',
                                                    ' '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\n+|\t+|\r+|\s+)', ' '))

    df_pandas = df.toPandas()

    for i,v in df_pandas.iterrows():
        print(v['content'])
        print(v['reg_content'])



    spark.stop()
