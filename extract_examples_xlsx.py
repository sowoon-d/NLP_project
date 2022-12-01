import common
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import time
import numpy as np
import pickle
import itertools
import yaml
from gensim.models import FastText
import pandas as pd

start = time.time()
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

df = spark.read.csv('data/mst+result+org.csv', header=True, encoding='UTF-8')
# df = df.limit(5000)
print(df.count())

# 2. 전처리 (특수기호)
df = common.preprocessing(df)
# print(df.show())
df.cache()
# time_2 = time.time()
# print('Process 2 done. ', round(time_2 - time_1, 3), 's')

# 3. 전처리 (띄어쓰기)
df = df.withColumn('spaced', common.get_spaced_result(df['reg_content']))
# time_3 = time.time()
# print('Process 3 done. ', round(time_3 - time_2, 3), 's')
# print(df.show())

df = df.toPandas()
# df_pandas = common.toPandas(df, 100)


# del df['reg_content']
# df.rename(columns={'spaced': 'content_2', 'content': 'content_1'}, inplace=True)
# print(df.head())
df2 = pd.DataFrame()
df2['source'] = range(len(df))
df2['content_1'] = df['content']
df2['content_2'] = df['spaced']
df2 = pd.wide_to_long(df2, stubnames='content_', i=['source'], j='number')
# print(df2.head())
df2 = df2.sort_index(axis=0)
# print(df2.head())
df2.to_excel('result/preprocessing_result.xlsx', encoding='utf-8', index=False)


with open('./result/nouns_.pkl', 'rb') as f:
    nouns_ = pickle.load(f)

print(len(df))
print(len(nouns_))

df3 = pd.DataFrame()
df3['content_1'] = df['spaced']
df3['content_2'] = nouns_
df3['source'] = range(len(df3))
df3 = pd.wide_to_long(df3, stubnames='content_', i=['source'], j='number')
# print(df3.head())
df3 = df3.sort_index(axis=0)
# print(df3.head())
df3.to_excel('result/noun_extractor_result.xlsx', encoding='utf-8', index=False)
