import pandas as pd
import random
import os
import sys
import numpy as np
import pickle


# export SIAMESE_BERT='/Users/foufamastafa/Documents/micro_projects/sentence_BERT_cosine/anomaly_detection_SPAM_nonSPAM'
assert os.environ.get('SIAMESE_BERT'), 'Please set the environment variable SIAMESE_BERT'
SIAMESE_BERT = os.environ['SIAMESE_BERT']
DATA_PATH = SIAMESE_BERT + "/data/"
sys.path.append(SIAMESE_BERT + '/data/')

import argparse

parser = argparse.ArgumentParser(description='Evaluating Siamese BERT on extremely-skewed dataset. ')

parser.add_argument('--nb_reference', type=int, default=1,
    help='Strategy used to compare test set with N reference normal observations. We strategy'
         'in {1,3} ')
parser.add_argument('--nb_pairs_nonSPAM', type=int, default=1e3,
    help='Number of pairs of nonSPAM used for comparison in the training set ')

parser.add_argument('--percentage_anomaly', type=float, default=100,
    help='Percentage of anomalies kept in the training set. This ratio is in percentage')


args = parser.parse_args()
NB_REFERENCE_NORMAL = args.nb_reference
N_pairs_nonSPAM = args.nb_pairs_nonSPAM
PERCENTAGE_ANOMALY = args.percentage_anomaly


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

        if PERCENTAGE_ANOMALY != 100: # If the percentage of desired anomalies is != 100%

            print("We take {}% of anomalies".format(PERCENTAGE_ANOMALY))
            print("Original number of anomalies {}: ".format(df_SPAM.shape[0]))
            # Then take only a fraction of anomalies = SPAM here
            ratio_anomaly = PERCENTAGE_ANOMALY/100
            nb_normal = df_nonSPAM.shape[0]
            # We know ratio_anomaly = nb_normal/(nb_normal + nb_expected_anomaly)
            # We deduce from that:
            nb_expected_anomaly = ratio_anomaly*nb_normal/(1-ratio_anomaly)
            df_SPAM = df_SPAM.sample(n = int(nb_expected_anomaly) , random_state = 1995)
            print("Sampled number of anomalies {}: ".format(df_SPAM.shape[0]))

        return df_nonSPAM, df_SPAM

    # For training, it works well but we made mistake in df_test
    # We need ot keep the indices intact when focusing on df_test
    df_nonSPAM_train, df_SPAM_train = prepare_for_fairs(df_train)
    # df_nonSPAM_test, df_SPAM_test = prepare_for_fairs(df_test)


    # 2/ Get pairs_train , pairs_test
    df_concat_train = get_pairs(df_nonSPAM_train, df_SPAM_train, N_pairs_nonSPAM=N_pairs_nonSPAM)
    # df_concat_test = get_pairs(df_nonSPAM_test, df_SPAM_test, N_pairs_nonSPAM=10000)

    # 3/ Save them in data directory
    df_concat_train.to_csv(SIAMESE_BERT + "/data/train/pairs_ham10K_spam75K.tsv", sep='\t', index=False)
    # df_concat_test.to_csv(SIAMESE_BERT + "/data/test/pairs_ham10K_spam75K.tsv", sep='\t', index=False)

    # df_concat = get_pairs(df_nonSPAM, df_SPAM, N_pairs_nonSPAM=10000)
    # df_concat.to_csv(SIAMESE_BERT+"/data/pairs_ham10K_spam75K.tsv", sep = '\t', index = False)
    # # print(df_concat.loc[:, ['label_SPAM', 'label_nonSPAM']].sample(5))
    # print(df_concat.sample(2))
    # print(df_concat.columns)
    # # print(df_concat[df_concat.label_nonSPAM.isnan])
    # print(df_concat.label.value_counts())

    file_indices_train_test = DATA_PATH + "/storage_indices_train_test.dic"
    # Get dictionary with indices from train/test set
    with open(file_indices_train_test, "rb") as f:
        storage_indices = pickle.load(f)

    df_test = df[df.index.isin(storage_indices['test'])]
    df_test = df_test.loc[:, ["message_cleaned", "is_spam"]]

    # For test set
    # Strategy 1 : only one reference observation for each x_test_obs
    if NB_REFERENCE_NORMAL == 1:
        """
        @TODO: 
    
        1/ Take reference nonSPAM from TRAINING !!!
        2/ For each observation, compare with each of the 3 reference observations 
    
        With this implementation, what we do is that we chose 3 random reference nonSPAM observations from training set
        We assign them to every test observation only once! So for every observation we have compare(x_obs, random(reference_nonSPAM))
    
        In the future, we want: most_common[ (x_obs, reference_nonSPAM(1)), (x_obs, reference_nonSPAM(2)), (x_obs, reference_nonSPAM(3)) ] 
        """
        # Prepare test data

        # file_indices_train_test = DATA_PATH + "/storage_indices_train_test.dic"
        # # Get dictionary with indices from train/test set
        # with open(file_indices_train_test, "rb") as f:
        #     storage_indices = pickle.load(f)

        # df = pd.read_csv("./data/df_spam_XLM_en_2048_embed.csv")
        # df_test = df[df.index.isin(storage_indices['test'])]
        # print(df_test.columns)

        # Step 2: We chose to take 3 nonSPAM representant as comparison for now
        # For each x_new , we do compare(x_new, x_nonSPAM(1)), compare(x_new, x_nonSPAM(2)) compare(x_new, x_nonSPAM(3))
        # Then we have label_1, label_2, label_3
        # label(x_new) = most_common_label(label_1, label_2, label_3)
        # Get 3 random nonSPAM representant
        df_test_sample_nonSPAM = df_test.groupby('is_spam').get_group(0).loc[:4, 'message_cleaned']  # 3 representant
        arr_nonSPAM_repr = np.array(df_test_sample_nonSPAM.values)
        # print (arr_nonSPAM_repr.shape[0])
        N_rep = df_test.shape[0] // arr_nonSPAM_repr.shape[0]
        # expand array of reference representants of nonSPAM
        arr_nonSPAM_repr_expand = np.tile(arr_nonSPAM_repr, N_rep)
        while arr_nonSPAM_repr_expand.shape[0] != df_test.shape[0]:
            arr_nonSPAM_repr_expand = np.append(arr_nonSPAM_repr_expand, [arr_nonSPAM_repr_expand[0]], axis=0)
        assert arr_nonSPAM_repr_expand.shape[0] == df_test.shape[
            0], "The reference nonSPAM texts does not match the test dataframe"

        # Concatenate df_test with arr_nonSPAM_repr_expand
        """
        @TODO: normally we should map 0 --> 1 and 1 --> 0 in this labelling because the model predicts the exact opposite from Pearson Correlation = -1 
    
        In the future I will do in the train set : label_true(nonSPAM, nonSPAM) = 1 and label_true(nonSPAM, SPAM) = 0
        But now, label_true(nonSPAM, nonSPAM) = 1 and label_true(nonSPAM, SPAM) = 0
    
        ------------------------------------------
        Let's test now with: 
        The current labelling where (SPAM, nonSPAM_reference ) will be label(SPAM) = 1 
        And (nonSPAM, SPAM) = 0 
        """
        df_test = df_test.loc[:, ['message_cleaned', "is_spam"]]
        df_test = pd.DataFrame(df_test)
        df_test['reference_nonSPAM'] = arr_nonSPAM_repr_expand
        # print(df_test.sample(2))

        df_test = df_test.reset_index(drop=True)
        df_test.to_csv(DATA_PATH + "/test/pairs_ham10K_spam75K.tsv", sep="\t")

    if NB_REFERENCE_NORMAL == 3:
        """
        In the following,

        We do exactly the same as before except that now we consider 3 reference comparisons instead of 1 
        and we get the most common label as our predicted_label
        """

        # Step 2: We chose to take 3 nonSPAM representant as comparison for now
        # For each x_new , we do compare(x_new, x_nonSPAM(1)), compare(x_new, x_nonSPAM(2)) compare(x_new, x_nonSPAM(3))
        # Then we have label_1, label_2, label_3
        # label(x_new) = most_common_label(label_1, label_2, label_3)
        # Get 3 random nonSPAM representant

        # @TODO: IN the future, you can get 3 representant from the TRAINING DATA not the TEST DATA as we do now.
        df_test_sample_nonSPAM = df_test.groupby('is_spam').get_group(0).loc[:4,
                                 'message_cleaned']  # get 3 representant
        arr_nonSPAM_repr = np.array(df_test_sample_nonSPAM.values)


        # We want [x_reference_normal_1 for _ in range(N_test_obs)] , [x_reference_normal_2 for _ in range(N_test_obs)], [x_reference_normal_3 for _ in range(N_test_obs)]
        N_test_obs = df_test.shape[0]
        ref_1_arr, ref_2_arr, ref_3_arr = [arr_nonSPAM_repr[0] for _ in range(N_test_obs)], [arr_nonSPAM_repr[1] for _
                                                                                             in range(N_test_obs)], [
                                              arr_nonSPAM_repr[2] for _ in range(N_test_obs)]
        ref_arr_tot = ref_1_arr + ref_2_arr + ref_3_arr  # concatenate above arrays

        # We extend df_test 3 times : [df_test, df_test, df_test]
        df_test_expand = pd.concat([df_test] * 3)  # Keep the index intact
        # Add a new columb called 'reference_obs_normal' with reference observations (from nonSPAM in this case)
        df_test_expand['reference_nonSPAM'] = ref_arr_tot

        df_test_expand.to_csv(DATA_PATH + "/test/pairs_ham10K_spam75K.tsv", sep="\t")

