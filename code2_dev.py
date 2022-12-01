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

# 1. 데이터 로드
# sql_pdis = "SELECT a.prob_cont, a.cause_cont, b.content FROM pdis_prob_hist a " \
#       "LEFT OUTER JOIN pdis_file_hist_tika b ON a.prob_id = b.prob_id"
# df_pdis = common.get_sparkSession_query(spark, sql_pdis)
# sql_aims = "select a.prob_cont, a.cause_cont, b.content from aims_prob_hist a " \
#            "FULL OUTER JOIN aims_files_tika b ON a.prob_id = b.prob_id;"
# df_aims = common.get_sparkSession_query(spark, sql_aims)
# sql_acms = "select a.prob_cont, a.cause_cont, b.content from acms_prob_hist a " \
#            "FULL OUTER JOIN acms_files_tika b ON a.prob_id = b.prob_id;"
# df_acms = common.get_sparkSession_query(spark, sql_acms)
# df = common.unionAll([df_pdis, df_aims, df_acms])

df = spark.read.csv('data/mst+result+org.csv', header=True, encoding='UTF-8')
# df = df.limit(10000)
print(df.count())
with open('./result/noun_scores.pkl', 'rb') as f:
    noun_scores = pickle.load(f)
with open('./result/preprocessed_content.pkl', 'rb') as f:
    result = pickle.load(f)
time_1 = time.time()
print('Process 1 done. ', round(time_1 - start, 3), 's')

# 2. noun_extract
nouns_ = common.get_noun_extractor(noun_scores, result)

# 엑셀 파일로 추출할 데이터 개수
df_len_limit = 1000

tmp_df = pd.DataFrame()
tmp_df['content_0'] = result[:df_len_limit]
tmp_df['content_1'] = nouns_[:df_len_limit]

time_2 = time.time()
print('Process 2 done. ', round(time_2 - time_1, 3), 's')

# 3. 불용어 read - db connection
sql_spam = '( select SPAM_KWRD as word from T_SPAM_KWRD ) t0'
df_spam = common.get_dictionary_words(spark, sql_spam)

spam_list = list(df_spam.select('WORD').toPandas()['WORD'])

time_3 = time.time()
print('Process 3 done. ', round(time_3 - time_2, 3), 's')


# 4. postprocessing
df_nouns = pd.DataFrame()
df_nouns['nouns'] = copy.deepcopy(nouns_)

df_post = common.postprocessing(df_nouns[:df_len_limit], spam_list)

time_4 = time.time()
print('Process 4 done. ', round(time_4 - time_3, 3), 's')


tmp_df['content_2'] = df_post['nouns'][:df_len_limit]
tmp_df['source'] = range(len(tmp_df))

tmp_df = pd.wide_to_long(tmp_df, stubnames='content_', i=['source'], j='number')
tmp_df = tmp_df.sort_index(axis=0)

tmp_df.to_excel('result/df_post.xlsx', encoding='utf-8', index=False)

# with open('result/nouns_.pkl', 'wb') as f:
#     pickle.dump(nouns_, f)


spark.stop()

end = time.time()
print('total time : ', time.strftime("%H:%M:%S", time.gmtime(end - start)))