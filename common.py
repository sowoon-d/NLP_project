from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import pyspark.sql.functions as fn
from pyspark.sql.types import *
from gensim.models import FastText, Word2Vec
import pandas as pd
from soyspacing.countbase import CountSpace
from soyspacing.countbase import RuleDict
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer
from soynlp.hangle import decompose
import re
import pickle
import time
import yaml
from datetime import datetime
import enc_conf as enc
import itertools
import nltk
import functools
import re

spacing_model = CountSpace()
spacing_model.load_model('result/soyspacing_model_20201103.json', json_format=True)

ini_path = 'conf/soyspacing_ft_conf.yaml'

with open(ini_path, encoding='utf-8') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

soy_verbose = conf['soyspacing']['verbose']
soy_mc = conf['soyspacing']['mc']  # min_count
soy_ft = conf['soyspacing']['ft']  # force_abs_threshold
soy_nt = conf['soyspacing']['nt']  # nonspace_threshold
soy_st = conf['soyspacing']['st']  # space_threshold

min_noun_score = conf['soynlp']['min_noun_score']  # min_noun_score
min_noun_frequency = conf['soynlp']['min_noun_frequency']  # min_noun_frequency
min_eojeol_frequency = conf['soynlp']['min_eojeol_frequency']  # min_eojeol_frequency

f.close()

ini_path_soy_rule = 'conf/soyspacing_rule.yaml'

with open(ini_path_soy_rule, encoding='utf-8') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

soy_list = conf['soyspacing']['rule']

f.close()

f = open("ruledict.txt", 'w', encoding='utf-8')
for i in soy_list:
    f.write(i)
    f.write('\n')
f.close()


rule_dict = RuleDict('ruledict.txt')


# get hive data, pyspark
def get_sparkSession_query(spark, sql):
    df = spark.sql(sql)
    df = df.withColumn('content', fn.concat(fn.col('prob_cont'), fn.lit(' ') \
                                            , fn.col('cause_cont'), fn.lit(' ') \
                                            , fn.col('content')))
    df_content = df.select('content')
    # print("df_content_total_row : ", df_content.count())
    return df_content


# 전처리
def preprocessing(df):
    df = df.withColumn('reg_content', fn.regexp_replace('content', 'null', ''))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\\uf0e0|\\u3000|\\x0b)', ' '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\(|\[|\{|\<)', ' \( '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\)|\]|\}|/>)', '\ ) '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\.|\?|\!)', ' . '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\,)', ' , '))
    df = df.withColumn('reg_content', fn.regexp_replace('reg_content', '(\n+|\t+|\r+|\s+)', ' '))

    return df


# 후처리 - pandas
def postprocessing(df, spam_list):
    # 단어 자체 삭제
    pattern1 = re.compile('(\((주|재|합|유|사|복|자|의|협|학)\))|(.+회사)|([0-9]{2,})|(\&.+;)|'
                          '(법인|학교|교회|병원|기독교|공장)|우체국|경찰서|새마을.*금고|'
                          '(한국|서울)|어린이|체육회|대표\s{,1}회의|협동\s{,1}조합|복지|아파트|현대|'
                          '.{1,3}구청|유치원|.*까$')
    loc_spam_list = ['광주','대한','강남','강동','강서','강북','송파','도봉','노원','성북','성동','동대문','전국','국립','은평']

    # 부분 삭제
    pattern2 = re.compile('[^가-힣a-zA-Z0-9|\/]')
    split_target_list = ['및', pattern2]

    # 정규식 조건(pattern1)으로 단어 삭제, loc_spam_list 포함 단어 삭제
    for row_index, value in df.iterrows():
        print(value['nouns'])
        for x in value['nouns'][:]:
            print(x)
            if len(x) <= 1:
                value['nouns'].remove(x)
            elif pattern1.match(x):
                print('!!')
                value['nouns'].remove(x)
                break
            else:
                for loc_spam in loc_spam_list:
                    if loc_spam in x:
                        value['nouns'].remove(x)
                        break

    # 정규식+및 들어간 단어 부분 삭제
    for row_index, value in df.iterrows():
        for x in value['nouns'][:]:
            for s_target in split_target_list:
                tmp_str = re.sub(s_target, '', x)
                value['nouns'][value['nouns'].index(x)] = tmp_str
                x = tmp_str
            if len(x) <= 1:
                value['nouns'].remove(x)

    # spam_list과 비교하여 exact_match 단어 삭제
    for row_index, value in df.iterrows():
        for x in value['nouns'][:]:
            if len(x) <= 1:
                value['nouns'].remove(x)
            else:
                for r_target in spam_list:
                    if r_target == x:
                        value['nouns'].remove(x)
                        break

    return df


def get_noun_score_LRNounExtractor_v2(sents):
    noun_extractor = LRNounExtractor_v2(verbose=False, extract_compound=True)
    noun_scores = noun_extractor.train_extract(sents, min_noun_score=min_noun_score,
                                               min_noun_frequency=min_noun_frequency,
                                               min_eojeol_frequency=min_eojeol_frequency)
    # noun_scores = {noun: score.score for noun, score in noun_scores.items()}

    return noun_scores


def get_noun_extractor(noun_scores, result):
    lmatch_tokenizer = NounLMatchTokenizer(nouns=noun_scores)
    # nouns_ = []
    # for sent in result:
    #     nouns_.append(lmatch_tokenizer.tokenize(sent))
    nouns_ = list(map(lambda x: lmatch_tokenizer.tokenize(x), result))

    return nouns_





@fn.udf(StringType())
def get_spaced_result(content):

    # yaml로 받아온 hyperparameter 사용
    sent_corrected, tags = spacing_model.correct(
        doc=content,
        verbose=soy_verbose,
        force_abs_threshold=soy_ft,
        nonspace_threshold=soy_nt,
        space_threshold=soy_st,
        min_count=soy_mc, rules=rule_dict)
    text = re.sub('\s+', ' ', sent_corrected)
    text = text.strip()

    return text


def _map_to_pandas(rdds):
    return [pd.DataFrame(list(rdds))]


def toPandas(df, n_partitions=None):
    if n_partitions is not None: df = df.repartition(n_partitions)
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    return df_pand


# 자음 모음 분리
def jamo_sentence(sent):
    doublespace_pattern = re.compile('\s+')

    def transform(char):
        if char == ' ':
            return char
        cjj = decompose(char)
        if cjj == None:
            return char
        elif len(cjj) == 1:
            return cjj
        cjj_ = ''.join(c if c != ' ' else '-' for c in cjj)
        return cjj_

    sent_ = ''.join(transform(char) for char in sent)
    sent_ = doublespace_pattern.sub(' ', sent_)
    return sent_


# 자음 모음 합치기
def decode(s):
    from soynlp.hangle import compose
    p = re.compile('[a-zA-Z]+')

    def process(t):
        t_ = t.replace('-', ' ')
        i = 0
        recovered = ''
        while i < len(t_):
            if p.match(t_[i]) != None:
                recovered = recovered + t_[i]
                i = i + 1
            else:
                recovered_ = compose(t_[i], t_[i + 1], t_[i + 2])
                recovered = recovered + recovered_
                i = i + 3
        return recovered

    return ' '.join(process(t) for t in s.split())


# tibero db connectiopn
def get_dictionary_words(spark, sql):
    decrypted_user, decrypted_password, decrypted_url, jdbc_driver, JAVA_HOME, class_name = enc.read_db_conf()

    df = spark.read \
        .format("jdbc") \
        .option('url', decrypted_url) \
        .option('dbtable', sql) \
        .option('user', decrypted_user) \
        .option('password', decrypted_password) \
        .option('driver', 'com.tmax.tibero.jdbc.TbDriver') \
        .load()

    # df_pandas = toPandas(df, 100)

    return df


# dict merge
def merge(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


# fasttext, fasttext_jamo 유사어 추출
def get_most_similar_word(word, ft_model, ft_model_jamo):
    ft_sim_words = dict((x, y) for x, y in ft_model.wv.similar_by_word(word, 10))
    ft_sim_jamo_words = dict((decode(x), y) for x, y in ft_model_jamo.wv.similar_by_word(jamo_sentence(word), 10))



    return dict(sorted(merge(ft_sim_words, ft_sim_jamo_words).items(), key=lambda x: x[1], reverse=True)[:5])


def bigram_join(tok_list):
    text = nltk.ngrams(tok_list,2)
    return list(map(lambda x: x[0] + ' ' + x[1], text))


def trigram_join(tok_list):
    text = nltk.ngrams(tok_list,3)
    return list(map(lambda x: x[0] + ' ' + x[1] + ' ' + x[2], text))


def quadgram_join(tok_list):
    text = nltk.ngrams(tok_list,4)
    return list(map(lambda x: x[0] + ' ' + x[1] + ' ' + x[2] + ' ' + x[3], text))


def pentagram_join(tok_list):
    text = nltk.ngrams(tok_list,5)
    return list(map(lambda x: x[0] + ' ' + x[1] + ' ' + x[2] + ' ' + x[3] + ' ' + x[4], text))


def join_gram(tok_list):
    bigram_list = bigram_join(tok_list)
    trigram_list = trigram_join(tok_list)
    quadgram_list = quadgram_join(tok_list)
    pentagram_list = pentagram_join(tok_list)

    total_word_list = pentagram_list + quadgram_list + trigram_list + bigram_list

    return total_word_list

def unionAll(dfs):
    return functools.reduce(lambda df1,df2: df1.union(df2.select(df1.columns)), dfs)

if __name__ == '__main__':
    start = time.time()
    spark_conf = SparkConf().setAppName("swpark") \
        .set("spark.driver.maxResultSize", "8g") \
        .set("spark.executor.heartbeatInterval", "3500s") \
        .set("spark.network.timeout", "3600s") \
        .set('spark.driver.extraClassPath', 'C:/Users/H2009134/PycharmProjects/cft2_1/tibero6-jdbc.jar')
    spark = SparkSession \
        .builder \
        .master("local[*]") \
        .enableHiveSupport() \
        .config(conf=spark_conf) \
        .appName("test") \
        .getOrCreate()
    today = datetime.today().strftime('%Y%m%d')

    # ini_path = 'conf/soyspacing_ft_conf.yaml'
    #
    # with open(ini_path, encoding='utf-8') as f:
    #     conf = yaml.load(f, Loader=yaml.FullLoader)
    #
    # soy_verbose = conf['soyspacing']['verbose']
    # soy_mc = conf['soyspacing']['mc']  # min_count
    # soy_ft = conf['soyspacing']['ft']  # force_abs_threshold
    # soy_nt = conf['soyspacing']['nt']  # nonspace_threshold
    # soy_st = conf['soyspacing']['st']  # space_threshold
    #
    # ft_size = conf['fasttext']['size']
    # ft_window = conf['fasttext']['window']
    # ft_mc = conf['fasttext']['min_count']
    # ft_sg = conf['fasttext']['sg']
    #
    # ft_size_jamo = conf['fasttext_jamo']['size']
    # ft_window_jamo = conf['fasttext_jamo']['window']
    # ft_mc_jamo = conf['fasttext_jamo']['min_count']
    # ft_sg_jamo = conf['fasttext_jamo']['sg']

    # 1. 데이터 로드
    # 나중되면 기존 + 신규 불러와서 합치고, soyspacing은 합친 것으로 하고, noun 추출은 신규 데이터만
    df = spark.read.csv('data/mst+result+org.csv', header=True, encoding='UTF-8')
    df = df.limit(5000)

    time_1 = time.time()
    print('Process 1 done. ', round(time_1 - start, 3), 's')

    # 2. 전처리 (특수기호)
    df = preprocessing(df)
    print(df.show())
    df.cache()
    time_2 = time.time()
    print('Process 2 done. ', round(time_2 - time_1, 3), 's')

    # 3. 전처리 (띄어쓰기) - 신규 데이터
    spacing_model = CountSpace()
    spacing_model.load_model('result/soyspacing_model_20201103.json', json_format=True)

    df = df.withColumn('spaced', get_spaced_result(df['reg_content']))
    time_3 = time.time()
    print('Process 3 done. ', round(time_3 - time_2, 3), 's')

    # 3-1. Pandas dataframe으로 변형
    df_pandas = toPandas(df, 100)

    time_3_1 = time.time()
    print('Process 3-1 done. ', round(time_3_1 - time_3, 3), 's')

    # 4. 띄어쓰기 된 신규 데이터 txt 파일 저장
    result = df_pandas['spaced'].tolist()

    with open("result/spaced_content.txt", "w", encoding='utf-8') as f:
        f.write('\n'.join(result))
    time_4 = time.time()
    print('Process 4 done. ', round(time_4 - time_3_1, 3), 's')

    # 5. 그 txt 파일로 띄어쓰기 모델 train, save
    spacing_model.train('result/spaced_content.txt')
    spacing_model.save_model('result/soyspacing_model_' + today + '.json', json_format=True)
    time_5 = time.time()
    print('\nProcess 5 done. ', round(time_5 - time_4, 3), 's')

    # 6. noun score 추출
    noun_scores = get_noun_score_LRNounExtractor_v2(result)
    with open('result/noun_scores.txt', 'wb') as f:
        pickle.dump(noun_scores, f)
    time_6 = time.time()
    print('Process 6 done. ', round(time_6 - time_5, 3), 's')

    # 7. noun 추출
    nouns_ = get_noun_extractor(noun_scores, result)
    # print('nouns >> ', nouns_)
    time_7 = time.time()
    print('Process 7 done. ', round(time_7 - time_6, 3), 's')

    # 7-1. nouns_를 1차원 리스트로 만들어서 n-gram 키워드 추출하고, 본문과 비교하는 부분 추가
    nouns_total = [n for nn in nouns_ for n in nn]

    ngram_list = join_gram(nouns_total)
    df_ngram = df_pandas['content'].str.extractall(pat='(' + "|".join(ngram_list) + ')')

    total_kwrd_list = df_ngram[0].tolist()
    total_kwrd_list = list(set(total_kwrd_list))  # 중복 제거
    print('N-gram word Count :', len(total_kwrd_list))
    time_7_1 = time.time()
    print('Process 7-1 done. ', round(time_7_1 - time_7, 3), 's')

    # 8. fasttext train, save model

    ini_path = 'conf/soyspacing_ft_conf.yaml'

    with open(ini_path, encoding='utf-8') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    ft_size = conf['fasttext']['size']
    ft_window = conf['fasttext']['window']
    ft_mc = conf['fasttext']['min_count']
    ft_sg = conf['fasttext']['sg']
    ft_size_jamo = conf['fasttext_jamo']['size']
    ft_window_jamo = conf['fasttext_jamo']['window']
    ft_mc_jamo = conf['fasttext_jamo']['min_count']
    ft_sg_jamo = conf['fasttext_jamo']['sg']

    ft_model = FastText(nouns_, size=ft_size, window=ft_window, min_count=ft_mc, workers=4, sg=ft_sg, sample=1e-3)
    ft_model.save('result/model_fasttext_' + today + '.bin')
    time_8 = time.time()
    print('Process 8 done. ', round(time_8 - time_7_1, 3), 's')

    # 9. fasttext_jamo train, save model
    noun_list_jamo = []
    for words in nouns_:
        tmp_list = []
        for word in words:
            tmp_list.append(jamo_sentence(word))
        noun_list_jamo.append(tmp_list)

    ft_model_jamo = FastText(noun_list_jamo, size=ft_size_jamo, window=ft_window_jamo, min_count=ft_mc_jamo, workers=4,
                             sg=ft_sg_jamo, sample=1e-3)
    ft_model_jamo.save('result/model_fasttext_jamo_' + today + '.bin')
    time_9 = time.time()
    print('Process 9 done. ', round(time_9 - time_8, 3), 's')

    # 10. 기존 사전 단어 read - db connection
    sql_ner = '( select INSTANCE as word from t_ner_instance ) t1'
    df_ner = get_dictionary_words(spark, sql_ner)
    print('INSTANCE Count :', len(df_ner))

    sql_nlp = '( select KEY_WORD as word from T_NLP_DIC ) t2'
    df_nlp = get_dictionary_words(spark, sql_nlp)
    print('NLP_DIC Count :', len(df_nlp))

    sql_sim = '( SELECT SIMILAR_WORD as word FROM t_synonym ) t3'
    df_sim = get_dictionary_words(spark, sql_sim)
    print('SIMILAR_WORD Count :', len(df_sim))

    sql_syn = '( SELECT SYNONYM_WORD as word FROM t_synonym ) t4'
    df_syn = get_dictionary_words(spark, sql_syn)
    print('SYNONYM_WORD Count :', len(df_syn))

    df_userDic = pd.concat([df_nlp, df_ner, df_sim, df_syn], ignore_index=True)
    print('CONCAT Data Count :', len(df_userDic))
    df_userDic.drop_duplicates(inplace=True)
    print('After Drop Duplicate :', len(df_userDic))
    print(df_userDic.head())

    spark.stop()
    time_10 = time.time()
    print('Process 10 done. ', round(time_10 - time_9, 3), 's')

    # 11. 신조어(사전에 없는 단어) 추출
    noun_list = list(itertools.chain(*nouns_))
    df_new_word = pd.DataFrame(data=noun_list, columns=['WORD'])
    df_new_word.drop_duplicates(inplace=True)

    df_t = pd.concat([df_userDic, df_new_word])
    df_t.drop_duplicates(['WORD'], keep=False, inplace=True)
    df_t.dropna(inplace=True)
    time_11 = time.time()
    print('Process 11 done. ', round(time_11 - time_10, 3), 's')

    # 12. fasttext, fasttext_jamo 유사어 추출
    df_t['simillar_word'] = df_t['WORD'].apply(lambda x: get_most_similar_word(x))
    df_t.to_csv('result/df_t.csv', encoding='utf-8', index=False)

    ft_model.init_sims(replace=True)  # 학습이 완료 되면 필요없는 메모리를 unload
    ft_model_jamo.init_sims(replace=True)
    time_12 = time.time()
    print('Process 12 done. ', round(time_12 - time_11, 3), 's')

    # 13. DB insert
    end = time.time()
    print('total time : ', time.strftime("%H:%M:%S", time.gmtime(end - start)))
