import pandas as pd
import random
from Genie import *

from multiprocessing import freeze_support
from copy import deepcopy


def find_mask(A):
    mask = np.std(A, axis=0) > 0
    return mask


def read_diagnosis(fname, sample=1):
    E = pd.read_csv(fname)
    print (f'There are {len(E)} unfiltered rows in the MIMIC3 dataset.')

    Diseases = []
    Patients = {}
    for i in range(len(E)):
        if random.uniform(0, 1.0) > sample:
            continue

        sid = E.iloc[i]['SUBJECT_ID']
        did = E.iloc[i]['ICD9_CODE']

        if did not in Diseases:
            Diseases.append(did)

        if sid not in Patients.keys():
            Patients[sid] = []

        Patients[sid].append(did)

    D = {'SUBJECT_ID': []}
    for i in range(len(Diseases)):
        D[Diseases[i]] = []

    for sid in Patients.keys():
        D['SUBJECT_ID'].append(sid)
        for disease in Diseases:
            if disease in Patients[sid]:
                D[disease].append(1)
            else:
                D[disease].append(0)

    D = pd.DataFrame(D)
    D.to_csv('/Users/sr0215/Python/Clinical/Bayes/ICD9_Data_Pivot_filtered2.csv')


if __name__ == '__main__':
    freeze_support()

    read_diagnosis('/Users/sr0215/Python/Clinical/Bayes/DIAGNOSES_ICD.csv')
    D = pd.read_csv('/Users/sr0215/Python/Clinical/Bayes/ICD9_Data_Pivot_filtered2.csv')

    # # Remove all columns with nans or zeros
    # D = D.loc[D.sum(axis=1).ne(0) & D.notna().all(axis=1),
    #           D.sum(axis=0).ne(0) & D.notna().all(axis=0)]

    # Remove all columns with nans or zeros
    D = D.loc[:, (D != 0).any(axis=0) & D.notna().any(axis=0)]
    print (len(D.columns))
    print (f'There are {len(D)} filtered (patient) rows in MIMIC3.')
    exit(1)

    Diseases = D.columns.tolist()[2:]
    Mapping = {Diseases[i]: i for i in range(len(Diseases))}
    # print (Mapping)

    # Input the prevalence matrix to GENIE
    A = D.iloc[:, 2:].values

    '''
    # Remove all diseases with 0 standard deviation
    mask_full = find_mask(deepcopy(A))
    mask_half = find_mask(deepcopy(A[:int(A.shape[0] / 2), :]))
    mask_all = [bool(mask_full[i] & mask_half[i]) for i in range(len(mask_full))]
    print (mask_all)

    A = deepcopy(A[:, mask_all])
    Diseases = [d for d, m in zip(Diseases, mask_all) if m]
    print (f'There are {len(Diseases)} diseases in our universe')
    print (A.shape)
    '''

    # GENIE comes here.
    # VIM3_half = GENIE3(deepcopy(A[:int(A.shape[0] / 2), :]),
    #                    nthreads=1, gene_names=Diseases, ntrees=200)
    VIM3 = GENIE3(A, nthreads=1, gene_names=Diseases, ntrees=200)
    # print (VIM3.shape)
    # print (VIM3[:3, :3])

    pickle.dump([A, Diseases, None, VIM3], open('/Users/sr0215/Python/Clinical/Bayes/Refinement/VIM3.p', 'wb'))

    '''
    # Optional
    Inverse_mapping = {Mapping[key]: key for key in Mapping.keys()}
    print (Inverse_mapping)
    '''
