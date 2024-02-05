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
        
        #self.resnet_model_nir = ResNet50(weights = 'imagenet', include_top=False, input_shape=nir_input_shape)
        #self.resnet_model_nir.trainable = train_resnet
        self.numeric_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Activation('relu'),
        ])
        self.resnet_fc_layer = Dense(128)
        self.bn = BatchNormalization()
        self.act = LeakyReLU()
        
        self.nir_fc_layer = Dense(128)
        self.nir_bn = BatchNormalization()
        self.nir_act = LeakyReLU()
        
        # self.act = Activation('relu')
        self.numeric_fc_layer = Dense(128, activation='relu')
        self.concat_layer = Concatenate()
        self.output_layer = Dense(1, activation='sigmoid')
        self.act2 = LeakyReLU()

    def call(self, inputs):
        image_input, nir_input, numeric_input = inputs
        
        # resnet_output = self.resnet_model(image_input)
        # resnet_output = self.resnet_fc_layer(resnet_output)
        # resnet_output = self.bn(resnet_output)
        # resnet_output = self.act(resnet_output)
        # resnet_output = Flatten()(resnet_output)
        
        # nir_output = self.resnet_model(nir_input)
        # nir_output = self.nir_fc_layer(nir_output)
        # nir_output = self.nir_bn(nir_output)
        # nir_output = self.nir_act(nir_output)
        # nir_output = Flatten()(nir_output)

        numeric_output = self.numeric_model(numeric_input)
        numeric_output = self.numeric_fc_layer(numeric_output)
        numeric_output = Flatten()(numeric_output)

        # fusion_output = self.concat_layer([resnet_output, nir_output, numeric_output])
        # output = self.output_layer(fusion_output)
        # output = self.act2(output)
        output = numeric_output

        return output
