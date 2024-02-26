import dill
import numpy as np
import pandas as pd

"""You need to execute this program first (after processing)"""

# dataset = 'mimic4'
dataset = 'mimic3'

record = dill.load(open(f'../data/{dataset}/output/records_final.pkl', 'rb'))
voc = dill.load(open(f'../data/{dataset}/output/voc_final.pkl', 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
voc_size = (
    len(diag_voc.idx2word),
    len(pro_voc.idx2word),
    len(med_voc.idx2word)
)

matrix1 = np.zeros([voc_size[0], voc_size[2]])
matrix2 = np.zeros([voc_size[1], voc_size[2]])

diag_count = np.zeros(voc_size[0])
proc_count = np.zeros(voc_size[1])

for patient in record:
    for adm in patient:
        for med in adm[2]:
            for diag in adm[0]:
                matrix1[diag][med] += 1
            for proc in adm[1]:
                matrix2[proc][med] += 1
        for diag in adm[0]:
            diag_count[diag] += 1
        for proc in adm[1]:
            proc_count[proc] += 1

for i in range(matrix1.shape[0]):
    matrix1[i, :] /= diag_count[i]

for i in range(matrix2.shape[0]):
    matrix2[i, :] /= proc_count[i]

effect_df1 = pd.DataFrame(0.0, index=[f"Diag_{i}" for i in range(voc_size[0])],
                         columns=[f"Med_{j}" for j in range(voc_size[2])])
effect_df1.iloc[:, :] = matrix1

effect_df2 = pd.DataFrame(0.0, index=[f"Proc_{i}" for i in range(voc_size[1])],
                         columns=[f"Med_{j}" for j in range(voc_size[2])])
effect_df2.iloc[:, :] = matrix2

dill.dump(effect_df1, open(f'../data/{dataset}/graphs/Diag_Med_relevance.pkl', 'wb'))
dill.dump(effect_df2, open(f'../data/{dataset}/graphs/Proc_Med_relevance.pkl', 'wb'))
