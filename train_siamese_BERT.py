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
parent_data_folder = './data/train/'
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
num_epochs = 1

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

# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# from SPAM_Reader import *
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from torch.utils.data import DataLoader

parent_data_folder = './data/test/'
spam_reader = SPAMReader(parent_data_folder)  # after

batch_size = 32


# model_save_path = "./output/train_SPAMbert-base-uncased-2020-03-01_10-31-29"
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


labels_pred = [threshold(dot_product) for sublist in labels for dot_product in sublist] # if positive value, they are similar, if negative they are dissimilar

# @TODO
# classification report with sklearn comparing labels and df_test.is_spam

# Error in Embedding NEW: @TODO: check how to get predictions
target_names = ['nonSPAM', 'SPAM']
classification_report_df = classification_report(df_test.is_spam, labels_pred, target_names=target_names)
print(classification_report_df)

classification_report_df.to_csv("./output/classification_report_test.csv")