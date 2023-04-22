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

def create_model():
    inputA = Input(shape=(1024,))
    inputB = Input(shape=(1024,))

    x = Dense(2048,kernel_initializer = 'he_uniform')(inputA)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = tf.nn.silu(x)
    x = Model(inputs=inputA, outputs=x)
    
    y = Dense(2048,kernel_initializer = 'he_uniform')(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = tf.nn.silu(y)
    y = Model(inputs=inputB, outputs=y)
    combined = concatenate([x.output, y.output])
    
    z = Dense(1024)(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    z = tf.nn.silu(z)
    z = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.summary()
    return model