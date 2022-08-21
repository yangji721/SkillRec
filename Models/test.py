import sys
sys.path.append("../../")
sys.path.append("../")
from CONFIG import HOME_PATH
import pandas as pd
from Utils.JobReader import read_jd, n_skill
from scipy.sparse import coo_matrix


if __name__ == "__main__":
    datalst = [['Google',10],['Runoob',12],['Wiki',13],['Google',10],['Runoob',12],['Google',10],[10, 'Google']]
    df_mat = pd.DataFrame(datalst, columns=['node_1', 'node_2'])
    df_mat['count'] = 1
    df_mat = df_mat.groupby(['node_1', 'node_2']).sum().reset_index()

    df_zero = pd.DataFrame([(ind, 2-ind, 1) for ind in range(5)], columns=['node_1', 'node_2', 'all_count'])

    df_zero2 = pd.DataFrame([(ind, ind+1) for ind in range(4)], columns=['node_1', 'all_count'])

    df_merge = pd.merge(df_zero, df_zero2, on='node_1')
    df_merge['count'] = df_merge[['all_count_x', 'all_count_y']].apply(lambda x: 1.0 * x[0] / x[1], axis=1)
    df_mat = df_merge[['node_1', 'node_2', 'count']]

    df_mat = df_mat[df_mat['count'] > 0.3]
    df_mat.drop('count', axis=1)
    
    a1_lst = df_mat['node_1'].tolist() + df_mat['node_2'].tolist()
    a2_lst = df_mat['node_2'].tolist() + df_mat['node_1'].tolist()
    df = pd.DataFrame()
    df['node_1'] = a1_lst
    df['node_2'] = a2_lst
    df = df.drop_duplicates()
    print(df)
    lst = [[] for u in range(100)]
    for a1, a2 in df[['node_1', 'node_2']].values.tolist():
        lst[a1].append(a2)
    print(lst)