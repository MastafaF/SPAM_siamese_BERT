"""
The system trains BERT on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import sys
sys.path.append("./")
sys.path.append("./utils")

from utils.EmbeddingSimilarityEvaluator import EmbeddingSimilarityEvaluatorNew
from SPAM_Reader import *
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

import argparse

parser = argparse.ArgumentParser(description='Evaluating Siamese BERT on extremely-skewed dataset. ')

parser.add_argument('--nb_reference', type=int, default=1,
    help='Strategy used to compare test set with N reference normal observations. We strategy'
         'in {1,3} ')

parser.add_argument('--epochs_train', type=int, default=1,
    help='Number of epochs to train the model ')

args = parser.parse_args()
NB_REFERENCE_NORMAL = args.nb_reference
NB_EPOCHS = args.epochs_train

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
# @TODO: load model corresponding to CamemBERT
model_name = 'bert-base-uncased'
batch_size = 32
# Data in French from Flaubert github
parent_data_folder = './data/'
spam_reader = SPAMReader(parent_data_folder)  # after
# sts_reader = STSDataReader('../datasets/stsbenchmark')
train_num_labels = spam_reader.get_num_labels()
model_save_path = 'output/train_SPAM' + model_name + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# Use BERT for mapping tokens to embeddings
word_embedding_model = models.BERT(model_name)
print(type(word_embedding_model))

#####################################################################
######### Focus on transfer learning on French NLI dataset #############
#########################################################################
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Convert the dataset to a DataLoader ready for training
# logging.info("Read AllNLI train dataset")
# train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
# train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
# train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

logging.info("Read french NLI train dataset")
# get_examples just get $split as parameter
train_data = SentencesDataset(spam_reader.get_examples('train'), model=model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

print(type(train_data))



logging.info("Read SPAM test dataset")
# test dataset contains: 5010 rows
# valid dataset contains: 2490
# # cf $wc - l valid.x2 ==> 2490
# dev_data = SentencesDataset(examples=sts_reader.get_examples('test'), model=model)
# dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
# evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

dev_data = SentencesDataset(spam_reader.get_examples('test'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

# Configure the training
# @TODO: add parameters num_epochs
num_epochs = NB_EPOCHS

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=8000,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on test data
#
##############################################################################
import sys
sys.path.append("./utils")
from EmbeddingSimilarityEvaluator import EmbeddingSimilarityEvaluatorNew
from SPAM_Reader import SPAMReader

# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# from SPAM_Reader import *
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from torch.utils.data import DataLoader

parent_data_folder = './data/'
spam_reader = SPAMReader(parent_data_folder)  # after

batch_size = 32

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=spam_reader.get_examples("test"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluatorNew(test_dataloader)
similarity, labels = model.evaluate(evaluator)

"""
Bon normalement, 

si x > 0, alors les points sont similaires alors on a (nonSPAM, nonSPAM) alors on assigne 1 
si x < 0, alors les points sont dissimilaires alors on a (SPAM, nonSPAM) alors on assigne 0 

Mais avec notre implementation source, on avait dit label_true(SPAM, nonSPAM) = label_true(SPAM) = 1 (c'etait simple a coder en fait car on preserve le label du SPAM)
et label_true
"""


def threshold(x):
    if x > 0:
        return 0
    else:
        return 1

if NB_REFERENCE_NORMAL == 3:
    labels_pred = [threshold(dot_product) for sublist in labels for dot_product in
                   sublist]  # if positive value, they are similar, if negative they are dissimilar

    df_test_expand['labels_pred'] = labels_pred


    def get_most_common_labels_to_df_NEW(df_comparisons):
        """
        Example:
        ----------------
        df_comparions:
            labels_pred
          1       1
          1       2
          2       1
          1       2

        output: pandas.DataFrame
              labels_pred
          1                 2
          2                 1

        Explanation:
        ----------------
        most_common(label_11, label_12, label_13) = most_common(1, 2, 2) = 2 for the observation with index = 1
        most_common(label_21) = most_common(1) = 1 for the observation with index = 2
        """
        print(
            "========== Estimating the labels by taking the most common labels from the comparisons to reference observations ==============")

        df_res = df_comparisons.groupby(df_comparisons.index).labels_pred.agg(pd.Series.mode)
        df_res = pd.DataFrame(df_res)
        # df_res.rename(columns = {'labels_pred': 'estimated_labels'}, inplace = True)
        return df_res


    # Getting most common labels in df_res
    df_res = get_most_common_labels_to_df_NEW(df_test_expand)
    # classification report with sklearn comparing labels and df_test.is_spam
    from sklearn.metrics import classification_report
    target_names = ['nonSPAM', 'SPAM']
    classification_report_df = classification_report(df_test.is_spam, df_res.labels_pred, target_names=target_names)
    print(classification_report_df)

    classification_report_df.to_csv("./output/classification_report_3ref.tsv", sep = "\t")

if NB_REFERENCE_NORMAL == 1:
    labels_pred = [threshold(dot_product) for sublist in labels for dot_product in
                   sublist]  # if positive value, they are similar, if negative they are dissimilar


    target_names = ['nonSPAM', 'SPAM']
    classification_report_df = classification_report(df_test.is_spam, labels_pred, target_names=target_names)
    print(classification_report_df)
    classification_report_df.to_csv("./output/classification_report_1ref.tsv", sep="\t")