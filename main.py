#!/usr/bin/python3

import os
import gc
import random
import warnings
import argparse
import pandas as pd
import numpy as np
from math import log
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score, precision_score, recall_score, f1_score
from keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU
from keras.models import Model
from keras.layers.merge import concatenate
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LayerNormalization
)
from keras.callbacks import EarlyStopping
from utils import load_data_split, print_performance
from nns import create_model
from query import *

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='1,2'


################################### Variable Parser #######################################

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, help='tcr or epi')
parser.add_argument('--active_learning', type=bool, default=False, help='True for use case a: reduce annotation cost, False for use case b:reduce redundancy')
parser.add_argument('--query_strategy', type=str, default='random_sampling')
parser.add_argument('--train_strategy', type=str, default='retrain', help='finetune or retrain')
parser.add_argument('--query_balanced', type=str, default='unbalanced', help='balanced or unbalanced')
parser.add_argument('--query_n', type=int, default=12003, help='this is 5%')
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--run', type=int)
parser.add_argument('--gpu', type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


######################### ActiveTCR use case a: Reducing annotation cost #####################

## Train an initial model using 10% data
def ActiveTCR_UC2(
    initTrain, pool_pos, pool_neg,
    X1_init, X2_init, y_init,                  # train_dataset,
    X1_pool_neg, X2_pool_neg, y_pool_neg,      # pool_negatives,
    X1_pool_pos, X2_pool_pos, y_pool_pos,      # pool_positives,
    X1_test, X2_test, y_test,                  # test_dataset,
    query_strategy,
    epochs,
    patience,
    num_iterations=8,
    sampling_size=24006,                       # 10% of Training data increment
):

    # Creating lists for storing metrics
    losses, val_losses, accuracies, val_accuracies = [], [], [], []

    model = create_model()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=keras.metrics.BinaryAccuracy())

    # Defining checkpoints.
    # The checkpoint callback is reused throughout the training since it only saves the best overall model.
    model_path = f'./models/{args.train_strategy}_{args.query_balanced}/{query_strategy}'
    logs_path = f'./logs/{args.train_strategy}_{args.query_balanced}/{query_strategy}'
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
        
    exp_name = args.split + '_seed_' + str(args.seed) + '_run_' + str(args.run)
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(model_path, exp_name + ".h5"),
        save_best_only=True, verbose=1,
    )
    early_stopping = keras.callbacks.EarlyStopping(patience=patience, verbose=1)

    print(f"Starting to train with {len(X1_init)} samples")
    # Initial fit with a small subset of the training set
    
    #### Save intermediate queried samples #######
    queried_samples = initTrain.assign(group = 0)
    queried_samples.to_pickle(os.path.join(logs_path, "QS_" + exp_name + ".pkl"))
    
    history = model.fit(
        [X1_init, X2_init], y_init, 
        batch_size = 32,
        epochs=epochs,
        validation_split=0.20,
        callbacks=[checkpoint, early_stopping],
        verbose=0,
    )
    
    ## Update variable name. This will be used in 20% loop iterations...
    X1_query = X1_init
    X2_query = X2_init
    y_query = y_init
    
    # Appending history
    losses, val_losses, accuracies, val_accuracies = append_history(
        losses, val_losses, accuracies, val_accuracies, history
    )
    
    model = keras.models.load_model(os.path.join(model_path, exp_name + ".h5"))
    
    y_pred = model.predict([X1_test, X2_test])
    print_performance(y_test, y_pred)
    
    ## 20% training loop...
    for iteration in range(1, num_iterations+1):
        
        if not args.active_learning:
            QS = query_strategies(model)

            if args.query_balanced == 'balanced':
                ## First loc ind for pos, then random generate balanced TCRs per epitope
                if query_strategy == 'random_sampling':
                    ind_pos = QS.random_sampling(X1_pool_pos, X2_pool_pos) # locating indexs for pool_pos
                elif query_strategy == 'entropy_sampling':        
                    ind_pos = QS.entropy_sampling(X1_pool_pos, X2_pool_pos) # locating indexs for pool_pos
                elif query_strategy == 'entropy_and_random_pooling_sampling':        
                    ind_pos = QS.entropy_and_random_pooling_sampling(X1_pool_pos, X2_pool_pos) # locating indexs for pool_pos
                elif query_strategy == 'entropy_and_random_pooling_and_dropout_sampling':
                    ind_pos = QS.entropy_and_random_pooling_and_dropout_sampling(X1_pool_pos, X2_pool_pos) 
                elif query_strategy == 'leastY_sampling':
                    ind_pos = QS.leastY_sampling(X1_pool_pos, X2_pool_pos) 

                # 2. Get index for pool_neg
                temp_pos = pool_pos.loc[ind_pos].value_counts(['epi']).reset_index(name='count')
                ind_neg = []

                for i in range(len(temp_pos)):
                    epi, count = temp_pos['epi'][i], temp_pos['count'][i]
                    # Search this epitope in negative and check their indexes list
                    ind_neg_per_epi = pool_neg.index[pool_neg['epi']==epi].tolist()
                    if len(ind_neg_per_epi) >= count:
                        ind_neg.extend(random.sample(ind_neg_per_epi, k=count))
                    else:
                        ind_neg.extend(ind_neg_per_epi)

            elif args.query_balanced == 'unbalanced':
                ## loc ind_pos and ind_neg at the same time. Cannot ganrantee for each epitope, it has 1:1 pos neg
                if query_strategy == 'random_sampling':
                    ind_pos = QS.random_sampling(X1_pool_pos, X2_pool_pos) # locating indexs for pool_pos
                    ind_neg = QS.random_sampling(X1_pool_neg, X2_pool_neg)
                elif query_strategy == 'entropy_sampling':        
                    ind_pos = QS.entropy_sampling(X1_pool_pos, X2_pool_pos) # locating indexs for pool_pos
                    ind_neg = QS.entropy_sampling(X1_pool_neg, X2_pool_neg)
                elif query_strategy == 'entropy_and_random_pooling_sampling':        
                    ind_pos = QS.entropy_and_random_pooling_sampling(X1_pool_pos, X2_pool_pos) # locating indexs for pool_pos
                    ind_neg = QS.entropy_and_random_pooling_sampling(X1_pool_neg, X2_pool_neg)
                elif query_strategy == 'entropy_and_random_pooling_and_dropout_sampling':
                    ind_pos = QS.entropy_and_random_pooling_and_dropout_sampling(X1_pool_pos, X2_pool_pos) 
                    ind_neg = QS.entropy_and_random_pooling_and_dropout_sampling(X1_pool_neg, X2_pool_neg)
                elif query_strategy == 'leastY_sampling':
                    ind_pos = QS.leastY_sampling(X1_pool_pos, X2_pool_pos) 
                    ind_neg = QS.mostY_sampling(X1_pool_neg, X2_pool_neg) 


            if args.train_strategy == 'finetune':
                ## Add and remove 
                # add pool_pos to train
                X1_query = np.concatenate((X1_pool_pos[ind_pos], X1_pool_neg[ind_neg]), axis=0)
                X2_query = np.concatenate((X2_pool_pos[ind_pos], X2_pool_neg[ind_neg]), axis=0)
                y_query = np.concatenate((y_pool_pos[ind_pos],y_pool_neg[ind_neg]), axis=0)
            elif args.train_strategy == 'retrain':
                X1_query = np.concatenate((X1_query, X1_pool_pos[ind_pos], X1_pool_neg[ind_neg]), axis=0)
                X2_query = np.concatenate((X2_query, X2_pool_pos[ind_pos], X2_pool_neg[ind_neg]), axis=0)
                y_query = np.concatenate((y_query, y_pool_pos[ind_pos],y_pool_neg[ind_neg]), axis=0)

            # update by removing ind from pool_pos
            X1_pool_pos = np.delete(X1_pool_pos, ind_pos, axis = 0)
            X2_pool_pos = np.delete(X2_pool_pos, ind_pos, axis = 0)
            y_pool_pos  = np.delete(y_pool_pos, ind_pos, axis = 0)

            X1_pool_neg = np.delete(X1_pool_neg, ind_neg, axis = 0)
            X2_pool_neg = np.delete(X2_pool_neg, ind_neg, axis = 0)
            y_pool_neg  = np.delete(y_pool_neg, ind_neg, axis = 0)

            print(f"Starting training with {len(X1_query)} samples")
            print("-" * 100)

            #### Save all intermediate queried samples with group id
            #### Save the queried_samples for post analysis. Very important!!!
            # 1. create a new df for these queried samples
            samples_pos_tmp = pool_pos.loc[ind_pos]
            samples_neg_tmp = pool_neg.loc[ind_neg]

            # 2. Drop these queried samples from the pool_pos and pool_neg
            pool_pos = pool_pos.drop(ind_pos)
            pool_neg = pool_neg.drop(ind_neg)

            # 3. Reset index for both the newly created query df and pool_pos and pool_neg
            samples_pos_tmp = samples_pos_tmp.reset_index(drop=True)
            samples_neg_tmp = samples_neg_tmp.reset_index(drop=True)
            pool_pos = pool_pos.reset_index(drop=True)
            pool_neg = pool_neg.reset_index(drop=True)

            # 4. Create a new group_id column for the queried samples
            new_queried_samples = pd.concat([samples_pos_tmp, samples_neg_tmp], ignore_index=True)
            new_queried_samples = new_queried_samples.assign(group = iteration)

            # 5. Concatenate the initial training data and queried samples
            queried_samples = pd.concat([queried_samples, new_queried_samples], ignore_index=True)
            queried_samples.to_pickle(os.path.join(logs_path, "QS_" + exp_name + ".pkl"))
        
        
        else:
            pass


        if args.train_strategy == 'finetune':
            history = model.fit(
                            [X1_query, X2_query], y_query, 
                            batch_size = 32,
                            epochs=epochs,
                            validation_split=0.20,
                            callbacks=[checkpoint, early_stopping],
                            verbose = 0
                        )
        elif args.train_strategy == 'retrain':
            # release memory
            del history
            del model
            gc.collect()
            
            model = create_model()
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=keras.metrics.BinaryAccuracy())
            
            history = model.fit(
                            [X1_query, X2_query], y_query, 
                            batch_size = 32,
                            epochs=epochs,
                            validation_split=0.20,
                            callbacks=early_stopping,
                            verbose = 0
                        )
        

        # Appending the history
        losses, val_losses, accuracies, val_accuracies = append_history(
            losses, val_losses, accuracies, val_accuracies, history
        )

        # Loading the best model from this training loop
#         model = keras.models.load_model(os.path.join(model_path, exp_name + ".h5"))
        y_pred = model.predict([X1_test, X2_test])
        print_performance(y_test, y_pred)

    print("-" * 100)
    print(
        "Test set evaluation: ",
        
        model.evaluate([X1_test, X2_test], y_test, verbose=0, return_dict=True),
    )
    print("-" * 100)

    return model


def ActiveTCR_UC1(
    initTrain, unlabelPool,
    X1_init, X2_init, y_init,                  # train_dataset,
    X1_pool, X2_pool, y_pool,      
    X1_test, X2_test, y_test,                  # test_dataset,
    query_strategy,
    epochs,
    patience,
    num_iterations=8,
    sampling_size=24006,                       # 10% of Training data increment
):

    # Creating lists for storing metrics
    losses, val_losses, accuracies, val_accuracies = [], [], [], []

    model = create_model()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=keras.metrics.BinaryAccuracy())

    # Defining checkpoints.
    # The checkpoint callback is reused throughout the training since it only saves the best overall model.
    model_path = f'./models/{args.train_strategy}_{args.query_balanced}/{query_strategy}'
    logs_path = f'./logs/{args.train_strategy}_{args.query_balanced}/{query_strategy}'
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
        
    if not args.active_learning:
        exp_name = args.split + '_seed_' + str(args.seed) + '_run_' + str(args.run)
    else:
        exp_name = 'AL_' + args.split + '_seed_' + str(args.seed) + '_run_' + str(args.run)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(model_path, exp_name + ".h5"),
        save_best_only=True, verbose=1
    )
    
    early_stopping = keras.callbacks.EarlyStopping(patience=patience, verbose=1)

    print(f"Starting to train with {len(X1_init)} samples")
    # Initial fit with a small subset of the training set
    
    #### Save intermediate queried samples #######
    queried_samples = initTrain.assign(group = 0)
    queried_samples.to_pickle(os.path.join(logs_path, "QS_" + exp_name + ".pkl"))
    
    history = model.fit(
        [X1_init, X2_init], y_init, 
        batch_size = 32,
        epochs=epochs,
        validation_split=0.20,
        callbacks=[checkpoint, early_stopping],
        verbose=0,
    )
    
    ## Update variable name. This will be used in 20% loop iterations...
    X1_query = X1_init
    X2_query = X2_init
    y_query = y_init
    
    # Appending history
    losses, val_losses, accuracies, val_accuracies = append_history(
        losses, val_losses, accuracies, val_accuracies, history
    )
    
    model = keras.models.load_model(os.path.join(model_path, exp_name + ".h5"))
    
    y_pred = model.predict([X1_test, X2_test])
    print_performance(y_test, y_pred)
    
    ## 20% training loop...
    for iteration in range(1, num_iterations+1):
        
        QS = query_strategies(model, n_samples=12003*2)
    
        if query_strategy == 'random_sampling':
            ind = QS.random_sampling(X1_pool, X2_pool)  # locating indexs for pool_pos
        elif query_strategy == 'entropy_sampling':        
            ind = QS.entropy_sampling(X1_pool, X2_pool) # locating indexs for pool_pos
        elif query_strategy == 'entropy_and_random_pooling_sampling':        
            ind = QS.entropy_and_random_pooling_sampling(X1_pool, X2_pool) # locating indexs for pool_pos
        elif query_strategy == 'entropy_and_random_pooling_and_dropout_sampling':
            ind = QS.entropy_and_random_pooling_and_dropout_sampling(X1_pool, X2_pool)  
        elif query_strategy == 'dropout_sampling_and_random_pooling_sampling':
            ind = QS.dropout_sampling_and_random_pooling_sampling(X1_pool, X2_pool)  
        

        if args.train_strategy == 'finetune':
            ## Add and remove 
            # add pool_pos to train
            X1_query = X1_pool[ind]
            X2_query = X2_pool[ind]
            y_query = y_pool[ind]
        elif args.train_strategy == 'retrain':
            X1_query = np.concatenate((X1_query, X1_pool[ind]), axis=0)
            X2_query = np.concatenate((X2_query, X2_pool[ind]), axis=0)
            y_query = np.concatenate((y_query, y_pool[ind]), axis=0)

        # update by removing ind from pool_pos
        X1_pool = np.delete(X1_pool, ind, axis = 0)
        X2_pool = np.delete(X2_pool, ind, axis = 0)
        y_pool  = np.delete(y_pool, ind, axis = 0)


        print(f"Starting training with {len(X1_query)} samples")
        print("-" * 100)

        #### Save all intermediate queried samples with group id
        #### Save the queried_samples for post analysis. Very important!!!
        # 1. create a new df for these queried samples
        samples_tmp = unlabelPool.loc[ind]

        # 2. Drop these queried samples from the pool_pos and pool_neg
        unlabelPool = unlabelPool.drop(ind)

        # 3. Reset index for both the newly created query df and pool_pos and pool_neg
        samples_tmp = samples_tmp.reset_index(drop=True)
        unlabelPool = unlabelPool.reset_index(drop=True)

        # 4. Create a new group_id column for the queried samples
        new_queried_samples = samples_tmp.assign(group = iteration)

        # 5. Concatenate the initial training data and queried samples
        queried_samples = pd.concat([queried_samples, new_queried_samples], ignore_index=True)
        queried_samples.to_pickle(os.path.join(logs_path, "QS_" + exp_name + ".pkl"))
        

        if args.train_strategy == 'finetune':
            history = model.fit(
                            [X1_query, X2_query], y_query, 
                            batch_size = 32,
                            epochs=epochs,
                            validation_split=0.20,
                            callbacks=[checkpoint, early_stopping],
                            verbose = 0
                        )
        elif args.train_strategy == 'retrain':
            # release memory
            del history
            del model
            gc.collect()
            
            model = create_model()
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=keras.metrics.BinaryAccuracy())
            
            history = model.fit(
                            [X1_query, X2_query], y_query, 
                            batch_size = 32,
                            epochs=epochs,
                            validation_split=0.20,
                            callbacks=early_stopping,
                            verbose = 0
                        )
        

        # Appending the history
        losses, val_losses, accuracies, val_accuracies = append_history(
            losses, val_losses, accuracies, val_accuracies, history
        )

        # Loading the best model from this training loop
#         model = keras.models.load_model(os.path.join(model_path, exp_name + ".h5"))
        y_pred = model.predict([X1_test, X2_test])
        print_performance(y_test, y_pred)

    print("-" * 100)
    print(
        "Test set evaluation: ",
        
        model.evaluate([X1_test, X2_test], y_test, verbose=0, return_dict=True),
    )
    
    print("-" * 100)

    return model



def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["binary_accuracy"]
    val_accuracy = val_accuracy + history.history["val_binary_accuracy"]
    return losses, val_losses, accuracy, val_accuracy



######################################## main code starts here. Reading input TCR-epitope pairs #####################
dat = pd.read_pickle("/mnt/disk07/user/pzhang84/data/tcr_repertoires_healthy_samples/combined_dataset_repTCRs/catELMo_4_layers_1024.pkl")

split = args.split
seed = args.seed
X1_train, X2_train, y_train, X1_test, X2_test, y_test, testData, trainData = load_data_split(dat,split, seed)


### Separate data into init training, final testing, and unlabeled pools
## Keep testing data unchanged. Prepare initTrain and unlabelled pool data
# 1. Randomly sample 10% of trainData and make it as initTrain. And make the rest 90% data pool data.
initTrain = trainData.sample(frac=0.1)
unlabelPool = trainData.drop(initTrain.index)

initTrain = initTrain.reset_index(drop=True)
unlabelPool = unlabelPool.reset_index(drop=True)

## ActiveTCR Use Case b
if not args.active_learning:
    # Separate postive pool and negative pools
    pool_pos = unlabelPool.loc[unlabelPool['binding'] == 1].reset_index(drop=True)
    pool_neg = unlabelPool.loc[unlabelPool['binding'] == 0].reset_index(drop=True)

    # 2. Convert initTrain and pool_pos into numpy
    X1_init = np.array(initTrain.tcr_embeds.to_list())
    X2_init = np.array(initTrain.epi_embeds.to_list())
    y_init = np.array(initTrain.binding.to_list())

    X1_pool_pos = np.array(pool_pos.tcr_embeds.to_list())
    X2_pool_pos = np.array(pool_pos.epi_embeds.to_list())
    y_pool_pos = np.array(pool_pos.binding.to_list())

    X1_pool_neg = np.array(pool_neg.tcr_embeds.to_list())
    X2_pool_neg = np.array(pool_neg.epi_embeds.to_list())
    y_pool_neg = np.array(pool_neg.binding.to_list())


    ## Train the model
    active_learning_model = ActiveTCR_UC2(
        initTrain, pool_pos, pool_neg,
        X1_init, X2_init, y_init,                  # train_dataset,
        X1_pool_neg, X2_pool_neg, y_pool_neg,      # pool_negatives,
        X1_pool_pos, X2_pool_pos, y_pool_pos,      # pool_positives,
        X1_test, X2_test, y_test,
        query_strategy = args.query_strategy,
        epochs = args.epochs,
        patience = args.patience,
    )
    

## ActiveTCR Use Case a
else:
    X1_init = np.array(initTrain.tcr_embeds.to_list())
    X2_init = np.array(initTrain.epi_embeds.to_list())
    y_init = np.array(initTrain.binding.to_list())

    X1_pool = np.array(unlabelPool.tcr_embeds.to_list())
    X2_pool = np.array(unlabelPool.epi_embeds.to_list())
    y_pool = np.array(unlabelPool.binding.to_list())

    ## Train the model
    active_learning_model = ActiveTCR_UC1(
        initTrain, unlabelPool,
        X1_init, X2_init, y_init,            
        X1_pool, X2_pool, y_pool,     
        X1_test, X2_test, y_test,
        query_strategy = args.query_strategy,
        epochs = args.epochs,
        patience = args.patience,
    )