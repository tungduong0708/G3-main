import pandas as pd
import re
import ast
from tqdm import tqdm

df_raw = pd.read_csv('./data/im2gps3k/im2gps3k_places365.csv')

# zs_df = pd.read_csv('./data/im2gps3k/0_llm_predict_results_rag.csv')
# rag_5_df = pd.read_csv('./data/im2gps3k/5_llm_predict_results_rag.csv')
# rag_10_df = pd.read_csv('./data/im2gps3k/10_llm_predict_results_rag.csv')
# rag_15_df = pd.read_csv('./data/im2gps3k/15_llm_predict_results_rag.csv')

zs_df_1 = pd.read_csv('./results/llm_predict_results_zs_1.csv')
zs_df_2 = pd.read_csv('./results/llm_predict_results_zs_2.csv')
zs_df_3 = pd.read_csv('./results/llm_predict_results_zs_3.csv')
zs_df_4 = pd.read_csv('./results/llm_predict_results_zs_4.csv')
zs_df_5 = pd.read_csv('./results/llm_predict_results_zs_5.csv')

pattern = r'[-+]?\d+\.\d+'

for i in tqdm(range(zs_df_1.shape[0])):
    response = zs_df_1.loc[i, 'response']
    response = ast.literal_eval(response)
    for idx, content in enumerate(response):
        idx = 0
        try:
            match = re.findall(pattern, content)
            latitude = match[0]
            longitude = match[1]
            df_raw.loc[i, f'zs_{idx}_latitude'] = latitude
            df_raw.loc[i, f'zs_{idx}_longitude'] = longitude
        except:
            df_raw.loc[i, f'zs_{idx}_latitude'] = '0.0'
            df_raw.loc[i, f'zs_{idx}_longitude'] = '0.0'

for i in tqdm(range(zs_df_2.shape[0])):
    response = zs_df_2.loc[i, 'response']
    response = ast.literal_eval(response)
    for idx, content in enumerate(response):
        idx = 1
        try:
            match = re.findall(pattern, content)
            latitude = match[0]
            longitude = match[1]
            df_raw.loc[i, f'zs_{idx}_latitude'] = latitude
            df_raw.loc[i, f'zs_{idx}_longitude'] = longitude
        except:
            df_raw.loc[i, f'zs_{idx}_latitude'] = '0.0'
            df_raw.loc[i, f'zs_{idx}_longitude'] = '0.0'

for i in tqdm(range(zs_df_3.shape[0])):
    response = zs_df_3.loc[i, 'response']
    response = ast.literal_eval(response)
    for idx, content in enumerate(response):
        idx = 2
        try:
            match = re.findall(pattern, content)
            latitude = match[0]
            longitude = match[1]
            df_raw.loc[i, f'zs_{idx}_latitude'] = latitude
            df_raw.loc[i, f'zs_{idx}_longitude'] = longitude
        except:
            df_raw.loc[i, f'zs_{idx}_latitude'] = '0.0'
            df_raw.loc[i, f'zs_{idx}_longitude'] = '0.0'

for i in tqdm(range(zs_df_4.shape[0])):
    response = zs_df_4.loc[i, 'response']
    response = ast.literal_eval(response)
    for idx, content in enumerate(response):
        idx = 3
        try:
            match = re.findall(pattern, content)
            latitude = match[0]
            longitude = match[1]
            df_raw.loc[i, f'zs_{idx}_latitude'] = latitude
            df_raw.loc[i, f'zs_{idx}_longitude'] = longitude
        except:
            df_raw.loc[i, f'zs_{idx}_latitude'] = '0.0'
            df_raw.loc[i, f'zs_{idx}_longitude'] = '0.0'

for i in tqdm(range(zs_df_5.shape[0])):
    response = zs_df_5.loc[i, 'response']
    response = ast.literal_eval(response)
    for idx, content in enumerate(response):
        idx = 4
        try:
            match = re.findall(pattern, content)
            latitude = match[0]
            longitude = match[1]
            df_raw.loc[i, f'zs_{idx}_latitude'] = latitude
            df_raw.loc[i, f'zs_{idx}_longitude'] = longitude
        except:
            df_raw.loc[i, f'zs_{idx}_latitude'] = '0.0'
            df_raw.loc[i, f'zs_{idx}_longitude'] = '0.0'

# for i in tqdm(range(df_raw.shape[0])):
#     response = rag_5_df.loc[i, 'rag_response']
#     response = ast.literal_eval(response)
#     for idx, content in enumerate(response):
#         try:
#             match = re.findall(pattern, content)
#             latitude = match[0]
#             longitude = match[1]
#             df_raw.loc[i, f'5_rag_{idx}_latitude'] = latitude
#             df_raw.loc[i, f'5_rag_{idx}_longitude'] = longitude
#         except:
#             df_raw.loc[i, f'5_rag_{idx}_latitude'] = '0.0'
#             df_raw.loc[i, f'5_rag_{idx}_longitude'] = '0.0'

# for i in tqdm(range(df_raw.shape[0])):
#     response = rag_10_df.loc[i, 'rag_response']
#     response = ast.literal_eval(response)
#     for idx, content in enumerate(response):
#         try:
#             match = re.findall(pattern, content)
#             latitude = match[0]
#             longitude = match[1]
#             df_raw.loc[i, f'10_rag_{idx}_latitude'] = latitude
#             df_raw.loc[i, f'10_rag_{idx}_longitude'] = longitude
#         except:
#             df_raw.loc[i, f'10_rag_{idx}_latitude'] = '0.0'
#             df_raw.loc[i, f'10_rag_{idx}_longitude'] = '0.0'

# for i in tqdm(range(df_raw.shape[0])):
#     response = rag_15_df.loc[i, 'rag_response']
#     response = ast.literal_eval(response)
#     for idx, content in enumerate(response):
#         try:
#             match = re.findall(pattern, content)
#             latitude = match[0]
#             longitude = match[1]
#             df_raw.loc[i, f'15_rag_{idx}_latitude'] = latitude
#             df_raw.loc[i, f'15_rag_{idx}_longitude'] = longitude
#         except:
#             df_raw.loc[i, f'15_rag_{idx}_latitude'] = '0.0'
#             df_raw.loc[i, f'15_rag_{idx}_longitude'] = '0.0'

df_raw.to_csv('./data/im2gps3k/im2gps3k_prediction.csv', index=False)
