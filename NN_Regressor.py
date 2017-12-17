#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:12:24 2017

__author__ =  "Mohsin Hasan Khan"
"""

#Import libraries
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint

class EM_NNRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, embed_cols=None, dense_cols=None, embed_dims=None, 
                 text_embed_cols=None, text_embed_seq_lens=None, 
                 text_embed_dims=None, text_embed_tokenizers=None,
                 num_layers=2, multiprocess=False,
                layer_activations=None, layer_dims=None,layer_dropouts=None, epochs=20, batchsize=32,
                optimizer_kwargs=None, val_size=0.1, verbose=1, seed=1):
        
        self.embed_cols = embed_cols
        self.dense_cols = dense_cols
        self.embed_dims = embed_dims
        self.text_embed_cols = text_embed_cols
        self.text_embed_dims = text_embed_dims
        self.text_embed_tokenizers = text_embed_tokenizers
        self.text_embed_seq_lens = text_embed_seq_lens
        self.dense_dims = None
        self.num_layers = num_layers
        self.layer_dims = layer_dims
        self.layer_activations = layer_activations
        self.layer_dropouts = layer_dropouts
        self.epochs = epochs
        self.batchsize = batchsize
        self.optimizer_kwargs = optimizer_kwargs
        self.val_size = val_size
        self.verbose = verbose
        self.multiprocess = multiprocess
        self.seed = seed
        self.model = None
        if self.dense_cols:
            self.dense_dims = len(self.dense_cols)
            
    def _splitX(self, X):
        X_splits = []
        
        if self.embed_cols:
            for col in self.embed_cols :
                X_splits.append(X[col].values.reshape(X.shape[0], -1))
                
        if self.text_embed_cols:
            for i, (col, tok) in enumerate(zip(self.text_embed_cols, self.text_embed_tokenizers)):
                max_len = self.text_embed_seq_lens[i]
                input_text = X[col].astype(str)
                x_train = tok.texts_to_sequences(input_text)
                print(np.mean([len(l) for l in x_train]))
                x_train = sequence.pad_sequences(x_train, maxlen=max_len)
                X_splits.append(np.array(x_train).reshape(X.shape[0], -1))
                
        if self.dense_cols:
            X_splits.append(X[self.dense_cols].values.reshape(X.shape[0], -1))
            
        return X_splits
    
    
    def _build_model(self):
        model_inputs = []
        model_layers = []
        
        if self.embed_cols:
            for col, dim in zip(self.embed_cols, self.embed_dims):
                x1 = Input( shape=(1,), name=col)
                model_inputs.append(x1)
                x1 = Embedding(input_dim=dim[0], output_dim=dim[1], )(x1)
                #x1 = Dropout(0.1)(x1)
                x1 = Reshape(target_shape=(dim[1],))(x1)
                model_layers.append(x1)
                
        if self.text_embed_cols:
            for col, dim, seq_len in zip(self.text_embed_cols, 
                                                self.text_embed_dims, 
                                                self.text_embed_seq_lens):
                x3 = Input( shape=(seq_len,))
                model_inputs.append(x3)
                x3 = Embedding(input_dim=dim[0], output_dim=dim[1], input_length=seq_len)(x3)
                x3 = GlobalAveragePooling1D()(x3)
                x3 = Reshape(target_shape=(dim[1],))(x3)
                model_layers.append(x3)
                
        if self.dense_cols:
            x2 = Input( shape=(self.dense_dims, ), name='dense_cols')
            model_inputs.append(x2)
            model_layers.append(x2)
        print(model_layers)
        x = concatenate(model_layers)
        
        if self.num_layers > 0:
            for dim, drops in zip(self.layer_dims, self.layer_dropouts):
                x = BatchNormalization()(x)
                x = Dropout(rate=drops)(x)
                x = Dense(dim, kernel_initializer='he_normal')(x)
                x = PReLU()(x)
        
        x = BatchNormalization()(x)
        x = Dropout(0.01)(x)
        output = Dense(1, activation='linear', kernel_initializer='normal')(x)
        
        model = Model(inputs=model_inputs, outputs=output)
        #print(model.summary())
        adam = SGD(lr=0.01, decay=1e-07, nesterov=True)
        model.compile(optimizer=adam, loss='mean_squared_error' )
        
        return model 
    
    
    def fit(self, X, y):
        self.model = self._build_model()
        if self.val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=self.seed)
            print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
            
            callbacks= [ModelCheckpoint("embed_NN_"+str(self.seed)+".check", save_best_only=True, verbose=1)]
            if self.multiprocess == False:
                self.model.fit(self._splitX(X_train), y_train, batch_size=self.batchsize, epochs=self.epochs,
                               verbose=self.verbose,
                              validation_data=(self._splitX(X_val), y_val), shuffle=True,
                              callbacks=callbacks)
            else:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size, random_state=1)

        else:
            self.model.fit(self._splitX(X), y, batch_size=self.batchsize, epochs=self.epochs,
               verbose=self.verbose, shuffle=True)

        
        return self
    
    def predict(self, X, y=None):
        
        if self.model:
            model = load_model("embed_NN_"+str(self.seed)+".check")
            y_hat = model.predict(self._splitX(X))
        else:
            raise ValueError("Model not fit yet")
            
        return y_hat
