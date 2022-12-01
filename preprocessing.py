#####################################################################################
# preprocessing func list
# rm_special_symbol(df) : 특수기호 제거 및 더블스페이스, 개행 제거
# apply_kospacing(df) : kospacing multiprocessing
# change_hanja(df) : change hanja

#####################################################################################


import pandas as pd
import numpy as np
from multiprocessing import Pool
from pykospacing import spacing
import hanja
from soyspacing.countbase import CountSpace


# 특수기호 제거 및 더블스페이스, 개행 제거
def rm_special_symbol(df):
    df['content'] = df['content'].str.replace(pat=r'[^가-힣A-Za-z0-9]', repl=r' ',
                                                    regex=True)  # replace all special symbols to space
    df['content'] = df['content'].str.replace(pat=r'[\n+]', repl=r' ', regex=True)
    df['content'] = df['content'].str.replace(pat=r'[\s\s+]', repl=r' ',
                                                          regex=True)  # replace multiple spaces with a single space
    return df


# kospacing multiprocessing
def spacing_processing(df):
    df['content'] = df['content'].apply(spacing)
    return df

def parallelize_dataframe(df,func,n_cores=4):
    df_split = np.array_split(df,n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))

    pool.close()
    pool.join()
    return df

def apply_kospacing(df):
    df = parallelize_dataframe(df, spacing_processing)
    return df


# change hanja
def change_hanja(df):
    df['content'] = df['content'].apply(lambda x: hanja.translate(x, 'substitution'))
    return df



def apply_soyspacing(df):
    model = CountSpace()
    model.load_model('result/sent_model.json', json_format=True)

    # verbose = False
    # mc = 10  # min_count
    # ft = 0.3  # force_abs_threshold
    # nt = -0.3  # nonspace_threshold
    # st = 0.3  # space_threshold

    # with parameters
    # sent_corrected, tags = model.correct(
    #     doc=df,
    #     verbose=verbose,
    #     force_abs_threshold=ft,
    #     nonspace_threshold=nt,
    #     space_threshold=st,
    #     min_count=mc)

    # without parameters
    df['content'] = df['content'].apply(lambda x: model.correct(x)[0])
    return df

