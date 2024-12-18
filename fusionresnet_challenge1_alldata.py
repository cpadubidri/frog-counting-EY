from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Concatenate, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers



class ResNetFusionModel(tf.keras.Model):
    def __init__(self, input_shape=(256,256, 3), train_resnet=True):
        super(ResNetFusionModel, self).__init__()
        self.resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        self.resnet_model.trainable = train_resnet
        initializer = tf.keras.initializers.HeNormal()
        self.numeric_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256,kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256,kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(128,kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(128,kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(128,kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
])
        self.resnet_fc_layer = Dense(128,kernel_initializer=initializer)
        self.bn = BatchNormalization()
        self.act = LeakyReLU()
        self.act = Activation('relu')
        self.numeric_fc_layer = Dense(128, activation='relu')
        self.concat_layer = Concatenate()
        self.output_layer = Dense(1, activation='sigmoid')
    def call(self, inputs):
        image_input, numeric_input = inputs
        resnet_output = self.resnet_model(image_input)
        #print(resnet_output.shape)
        resnet_output = self.resnet_fc_layer(resnet_output)
        #print(resnet_output.shape)
        #resnet_output = self.resnet_fc_layer(resnet_output)
        resnet_output = self.bn(resnet_output)
        #print(resnet_output.shape)
        resnet_output = self.act(resnet_output)
        #print(resnet_output.shape)
        resnet_output = Flatten()(resnet_output)
        #print(resnet_output.shape)
        
        numeric_output = self.numeric_model(numeric_input)
        #print(numeric_output.shape)
        numeric_output = self.numeric_fc_layer(numeric_output)
        #print(numeric_output.shape)
        numeric_output = Flatten()(numeric_output)
        #print(numeric_output.shape)
        fusion_output = self.concat_layer([resnet_output, numeric_output])
        #print(fusion_output.shape)
        
        output = self.output_layer(fusion_output)
        #print(output.shape)
        #output = self.act2(output)
        return output