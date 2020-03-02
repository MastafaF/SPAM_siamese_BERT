import pandas as pd
import random
import os
import sys
import numpy as np
import pickle


# export SIAMESE_BERT='/Users/foufamastafa/Documents/micro_projects/sentence_BERT_cosine/anomaly_detection_SPAM_nonSPAM'
assert os.environ.get('SIAMESE_BERT'), 'Please set the environment variable VAE_DETECTOR'
SIAMESE_BERT = os.environ['SIAMESE_BERT']
DATA_PATH = SIAMESE_BERT + "/data/"
sys.path.append(SIAMESE_BERT + '/data/')

random.seed(1995)

df = pd.read_csv(DATA_PATH + "/df_spam_XLM_en_2048_embed.csv")

# # is_spam = 1
# df_SPAM = df.groupby('is_spam').get_group(1).loc[:, 'message_cleaned']
# df_nonSPAM = df.groupby('is_spam').get_group(0).loc[:, 'message_cleaned']
# print("====== TYPE ========")
# print(type(df_SPAM))
#
# df_SPAM = pd.DataFrame(df_SPAM)
# df_nonSPAM = pd.DataFrame(df_nonSPAM)
#
# df_SPAM['label_SPAM'] = 1
# df_nonSPAM['label_nonSPAM'] = 0


# Merge those two dataframes
# df_SPAM['key'] = 1
# df_nonSPAM['key'] = 1
# #
# print(df_SPAM.sample(2))
# df_merge = pd.merge(df_SPAM, df_nonSPAM, on='key').drop('key',axis=1)
# print(df_SPAM.shape)
# print(df_nonSPAM.shape)
# print(df_merge.shape)
# assert df_merge.shape[0] == df_SPAM.shape[0]*df_nonSPAM.shape[0], "Not desired result"
# %%
def get_pairs(df_nonSPAM, df_SPAM, N_pairs_nonSPAM):
    # df_merge_2: nonSPAM, nonSPAM (fraction)
    # Strategy 1
    # Take one portion of the data
    # Dublicate it
    # As for df_merge_1, build all possible pairs from it
    # Let k be the fraction of points we take from df_nonSPAM
    # If df_nonSPAM of shape n1
    # Then we expect to have n1**2/k**2 pairs
    # We can force the number of pairs we desire
    # And from it, we deduce k
    # Indeed we have n1**2/k**2 = N_pairs_nonSPAM
    # So k = np.sqrt(n1**2/N)
    n1 = df_nonSPAM.shape[0]
    k = np.sqrt(n1 ** 2 / N_pairs_nonSPAM)
    df_fraction_nonSPAM = df_nonSPAM.sample(frac=1 / k, random_state=1995)  # we have n1/k elements from this
    df_fraction_nonSPAM['key'] = 1
    df_merge_2 = pd.merge(df_fraction_nonSPAM, df_fraction_nonSPAM, on='key').drop('key', axis=1)
    df_merge_2.drop('label_nonSPAM_y', axis = 1, inplace = True)
    df_merge_2.rename(columns = {'label_nonSPAM_x':'label'}, inplace = True)

    print(df_merge_2.columns)
    # df_merge_1 : nonSPAM, SPAM
    df_SPAM['key'] = 1
    df_nonSPAM['key'] = 1
    # Instead of merging df_nonSPAM with df_SPAM which results in many pairs
    # We can merge df_fraction_nonSPAM
    df_merge_1 = pd.merge(df_fraction_nonSPAM, df_SPAM, on='key').drop('key', axis=1)
    # For a pair (nonSPAM, SPAM) the label that we give is the one from SPAM ie 1
    df_merge_1.drop('label_nonSPAM', axis=1, inplace = True)
    df_merge_1.rename(columns = {'label_SPAM': 'label'}, inplace=True)

    # Concatenate both dataframe
    df_concat = pd.concat([df_merge_1, df_merge_2], axis=0)
    df_concat = df_concat.sample(frac = 1)
    print("Fraction of nonSPAM data {}%".format((1 / k) * 100))
    # print(df_concat.sample(2))
    return df_concat

if __name__ == "__main__":
    # 1/ Get training data and test data

    # For good comparison, we use the indices that we stored previously in a dictionary and used for
    # experiments in our paper. (from sentence_ROBERTA experiments)
    with open(DATA_PATH + "/storage_indices_train_test.dic", "rb") as f:
        storage_indices = pickle.load(f)

    df_train, df_test = df[df.index.isin(storage_indices['train'])], df[df.index.isin(storage_indices['test'])]
    print(storage_indices.keys())
    print(df_train.shape, df_test.shape)

    # is_spam = 1
    def prepare_for_fairs(df):
        # is_spam = 1
        df_SPAM = df.groupby('is_spam').get_group(1).loc[:, 'message_cleaned']
        df_nonSPAM = df.groupby('is_spam').get_group(0).loc[:, 'message_cleaned']

        df_SPAM = pd.DataFrame(df_SPAM)
        df_nonSPAM = pd.DataFrame(df_nonSPAM)

        df_SPAM['label_SPAM'] = 1
        df_nonSPAM['label_nonSPAM'] = 0
        return df_nonSPAM, df_SPAM

    df_nonSPAM_train, df_SPAM_train = prepare_for_fairs(df_train)
    df_nonSPAM_test, df_SPAM_test = prepare_for_fairs(df_test)


    # 2/ Get pairs_train , pairs_test
    df_concat_train = get_pairs(df_nonSPAM_train, df_SPAM_train, N_pairs_nonSPAM=10000)
    df_concat_test = get_pairs(df_nonSPAM_test, df_SPAM_test, N_pairs_nonSPAM=10000)

    # 3/ Save them in data directory
    df_concat_train.to_csv(SIAMESE_BERT + "/data/train/pairs_ham10K_spam75K.tsv", sep='\t', index=False)
    df_concat_test.to_csv(SIAMESE_BERT + "/data/test/pairs_ham10K_spam75K.tsv", sep='\t', index=False)

    # df_concat = get_pairs(df_nonSPAM, df_SPAM, N_pairs_nonSPAM=10000)
    # df_concat.to_csv(SIAMESE_BERT+"/data/pairs_ham10K_spam75K.tsv", sep = '\t', index = False)
    # # print(df_concat.loc[:, ['label_SPAM', 'label_nonSPAM']].sample(5))
    # print(df_concat.sample(2))
    # print(df_concat.columns)
    # # print(df_concat[df_concat.label_nonSPAM.isnan])
    # print(df_concat.label.value_counts())