
from tensorflow import keras
import os
import sys
import random
import math
import numpy as np
import cv2
import pandas as pd
import imageio.v2 as imageio
from scipy.ndimage.filters import gaussian_filter 
from sklearn.preprocessing import MinMaxScaler, Normalizer
import imgaug.augmenters as iaa

class DataFeeder(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.imagepath = path
        self.csvpath =  os.path.join(path,'updated_reduced_final_bl_duplicated.csv')
        self.covariatesDf = pd.read_csv(self.csvpath)
        self.batch_size = batch_size
        self.image_size = image_size
        self.covscale = MinMaxScaler()
        # self.on_epoch_end()
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Affine(rotate=(-10, 10)),  # random rotations
            iaa.GaussianBlur(sigma=(0, 0.5)),  # random Gaussian blur
        ])
    
    def __load__(self, filepath, augment=False):
        image = imageio.imread(filepath)
        image = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        if augment:
            image = self.augmenter(image=image)
        image = cv2.resize(image, (self.image_size, self.image_size))
        if len(image.shape) == 2:  # Check if the image is grayscale
            image = np.expand_dims(image, axis=-1)  # Add channel dimension
            image = np.repeat(image, 3, axis=-1)
        return image
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        images = []
        covariates = []
        labels  = []
        nirs=[]
        processed_ids =set() # keep track of ids
        weights = []
        for id_name in files_batch:
            
                
                
            country = DataFeeder.getCountry(id_name.split('_')[2])
            
            if country in ['costarica_new2']:
                weight = 5.0
            elif country in ['southafrica']:
                weight = 3.0
            else:
                weight = 1.0
            weights.append(weight)
            # print(country)
            imagepath, imagename = DataFeeder.getRgbFile(self.imagepath, country, id_name)
            
            if id_name in processed_ids:
                image = self.__load__(imagepath, augment=True)
            else:
                image = self.__load__(imagepath)
                
            
            #print(image.shape)
            # print(image.max(), image.min())
            images.append(image)
            
            nirpath, nirname = DataFeeder.getNirFile(self.imagepath, country, id_name)
            if id_name in processed_ids:
                nir = self.__load__(nirpath, augment=True)
            else:
                nir = self.__load__(nirpath)
            #print(nir.shape)
            nirs.append(nir)
            processed_ids.add(id_name)
            covariatesrow = self.covariatesDf[self.covariatesDf['id']==id_name]
            covariatesrow = covariatesrow.head(1) 
            cnt = covariatesrow['frog_count'].values[0]
            cnt = 1 if cnt < 250 else 2
            # print(cnt)
            # cnt = 1 if cnt < 10 else (10 if cnt >= 100 else (cnt // 10 + 1))
            # print(covariatesrow['frog_count'].values[0])
            # labels.append(covariatesrow['frog_count'].values[0])
            labels.append(cnt)
            
            #labels = np.array(labels).reshape(-1, 1)
            #print(labels)
            covariate = covariatesrow[["pdsi","pet","ppt","q","soil","tmax","tmin","vap","vpd","ws"]]
            covariate = np.array(covariate)
            # print(covariate.shape)
            covariate = self.normalizeData(covariate)
            covariates.append(covariate)
            
        
        weights = np.array(weights, dtype = np.float32)
        images = np.array(images, dtype=np.float32)
        # print(images.shape)
        labels = np.array(labels, dtype=np.float32)
        labels_array = labels.reshape(-1, 1)
        #print(labels_array)
        
        nirs = np.array(nirs, dtype=np.float32)
        covariates = np.array(covariates, dtype=np.float32)
        # print(covariates.shape)
        
        return [images,nirs,covariates], labels_array, weights
    
    def normalizeData(self, data):
        return self.covscale.fit_transform(data.reshape(-1, 1))

    @staticmethod
    def getCountry(code):
        if code=='aus':
            return 'australia'
        elif code=='sa':
            return 'southafrica'
        else:
            return 'costarica_new2'

    @staticmethod
    def getRgbFile(path, country, id_name):
        filename = id_name.replace("cropped_images", "esri_land")
        return os.path.join(path, country, id_name, filename+'.tif'), filename
    
    def getNirFile (path, country, id_name):
        filename = id_name.replace("cropped_images", "ndvi")
        return os.path.join(path, country, id_name, filename+'.tif'), filename
    
    def on_epoch_end(self):
        random.shuffle(self.ids)
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))


def datafeeder(path, train_split,batch_size, image_size, skipdata='missing.csv'):
    csv_data = pd.read_csv(os.path.join(path,'updated_reduced_final_bl_duplicated.csv'))
    missing_data = pd.read_csv(skipdata)
    missing_data = list(missing_data['0'])
    image_id = list(csv_data['id'])
    image_id = [x for x in image_id if x not in missing_data]
    random.shuffle(image_id)
    # print(len(image_id))

    train_ids = image_id[:int(len(image_id)*train_split)]
    test_ids = image_id[int(len(image_id)*train_split):]
    # print(len(train_ids), len(test_ids))
    train_feeder = DataFeeder(ids=train_ids, path=path, batch_size=batch_size, image_size=image_size)
    test_feeder = DataFeeder(ids=test_ids, path=path, batch_size=batch_size, image_size=image_size)
    
    return train_feeder, test_feeder



if __name__=='__main__':
    # pass
    hyperparameters = {
        "dataset_path":'data',
        "train_split":1,
        "batch_size": 20,
        "input_shape": (512,512, 3),
        "epochs": 10,
        "save_weight_epoch": 1
    }
    train_feeder, test_feeder = datafeeder(path=hyperparameters['dataset_path'], 
                                               train_split=hyperparameters['train_split'], 
                                               batch_size=hyperparameters['batch_size'], 
                                               image_size=hyperparameters['input_shape'][0])


    for i in range(0,20):
        x = train_feeder.__getitem__(i)
        print(x[1])
        # print(x[0][0].shape,x[0][1].shape,x[0][2])
