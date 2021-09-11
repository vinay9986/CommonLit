import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


def handle_cate_NA(df, columns_to_ignore=[]):
    temp = copy.deepcopy(df)
    cate_cols = list(set(temp.select_dtypes('object').columns.tolist()) - set(columns_to_ignore))
    for col in cate_cols:
        if temp[col].isna().sum() > 0:
            column_name = 'NA_POS_'+col
            col_values = ['Y' if pd.isna(value[1]) else 'N' for value in df[col].items()]
            temp[col].fillna(value='ABS', inplace=True)
            temp[column_name] = col_values
    return temp

def handle_cont_NA(df, method='mean'):
    action = ''.join(c.lower() for c in method if not c.isspace())
    temp = copy.deepcopy(df)
    num_cols = temp.select_dtypes(include='number')
    for col in num_cols:
        if temp[col].isna().sum() > 0:
            column_name = 'NA_POS_'+col
            col_values = ['Y' if pd.isna(value[1]) else 'N' for value in df[col].items()]
            #value_if_true if condition else value_if_false
            fill_value = np.mean(temp[col]) if 'mean' == action else np.median(temp[col])
            temp[col].fillna(value = fill_value, inplace=True)
            temp[column_name] = col_values
    return temp

def train_pca(df, list_of_columns, column_prefix):
    temp = copy.deepcopy(df)
    x = temp.loc[:, list_of_columns].values
    ss = StandardScaler().fit(x)
    x = ss.transform(x)
    pca = PCA(n_components=2)
    pca.fit(x)
    principalComponents = pca.transform(x)
    print(column_prefix, pca.explained_variance_ratio_)
    principalDf = pd.DataFrame(data = principalComponents, columns = [column_prefix+'_1', column_prefix+'_2'])
#     temp.drop(columns=list_of_columns, axis=1, inplace=True)
    temp = pd.concat([temp, principalDf], axis = 1)
    result_dict = { 'pca': pca, 'ss': ss, 'list_of_columns': list_of_columns, 'column_prefix': column_prefix } 
    return result_dict, temp

def apply_pca(trained_pca, df):
    temp = copy.deepcopy(df)
    x = temp.loc[:, trained_pca.get('list_of_columns')].values
    x = trained_pca.get('ss').transform(x)
    principalComponents = trained_pca.get('pca').transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = [trained_pca.get('column_prefix')+'_1', trained_pca.get('column_prefix')+'_2'])
#     temp.drop(columns=trained_pca.get('list_of_columns'), axis=1, inplace=True)
    temp = pd.concat([temp, principalDf], axis = 1)
    return temp