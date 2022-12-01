import common
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import time
import pickle
import pandas as pd

start = time.time()
spark_conf = SparkConf().setAppName("swpark") \
    .set("spark.driver.maxResultSize", "10g") \
    .set("spark.executor.instances", "4") \
    .set("spark.executor.cores", "4") \
    .set("spark.executor.heartbeatInterval", "4000s") \
    .set("spark.network.timeout", "4100s") \
    .set('spark.driver.extraClassPath', 'tibero6-jdbc.jar')
spark = SparkSession \
    .builder \
    .master("local[*]") \
    .enableHiveSupport() \
    .config(conf=spark_conf) \
    .appName("test") \
    .getOrCreate()

# 1. 데이터 로드
# df = common.get_sparkSession_query(spark)
df = spark.read.csv('data/mst+result+org.csv', header=True, encoding='UTF-8')
# df = df.limit(400000)
print(df.count())
time_1 = time.time()
print('Process 1 done. ', round(time_1 - start, 3), 's')

# 2. 전처리 (특수기호)
df = common.preprocessing(df)
# print(df.show())
# df.cache()
time_2 = time.time()
print('Process 2 done. ', round(time_2 - time_1, 3), 's')

# 3. 전처리 (띄어쓰기)
df = df.withColumn('spaced', common.get_spaced_result(df['reg_content']))
df.cache()
time_3 = time.time()
print('Process 3 done. ', round(time_3 - time_2, 3), 's')
# print(df.show())



# 4. spark df column to python list
# org_content = list(df.select('content').toPandas()['content'])
# reg_content = list(df.select('reg_content').toPandas()['reg_content'])
result = list(df.select('spaced').toPandas()['spaced'])

# # 엑셀 파일로 추출할 데이터 개수
# df_len_limit = 1000
#
# tmp_df = pd.DataFrame()
# tmp_df['content_0'] = org_content[:df_len_limit]
# tmp_df['content_1'] = reg_content[:df_len_limit]
# tmp_df['content_2'] = result[:df_len_limit]
# tmp_df['source'] = range(len(tmp_df))
#
# tmp_df = pd.wide_to_long(tmp_df, stubnames='content_', i=['source'], j='number')
# tmp_df = tmp_df.sort_index(axis=0)
#
# tmp_df.to_excel('result/preprocessing_20210122.xlsx', encoding='utf-8', index=False)

time_4 = time.time()
print('Process 4 done. ', round(time_4 - time_3, 3), 's')
#
# 5. noun score 추출
noun_scores = common.get_noun_score_LRNounExtractor_v2(result)
with open('result/noun_scores.pkl', 'wb') as f:
    pickle.dump(noun_scores, f)
time_5 = time.time()
print('Process 5 done. ', round(time_5 - time_4, 3), 's')

# 6. 전처리 된 본문 파일 저장 (newword_extract에 사용)
with open('result/preprocessed_content.pkl', 'wb') as f:
    pickle.dump(result, f)
time_6 = time.time()
print('Process 6 done. ', round(time_6 - time_5, 3), 's')

spark.catalog.clearCache()
spark.stop()

end = time.time()
print('total time : ', time.strftime("%H:%M:%S", time.gmtime(end - start)))


