from data_feeder_v2_rgb_weighted import DataFeeder
#from datafeedCPP import DataFeederSAMP
from fusionresnet_test import ResNetFusionModel
from tensorflow.keras import regularizers
import tensorflow as tf
import pandas as pd
import os
import csv
import numpy as np
from tqdm import tqdm
from numpy import interp
import time
import random
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def regularized_loss(loss_fn, model, l1_factor=0.01, l2_factor=0.01):
    def loss_with_regularization(y_true, y_pred, weights):
        loss_value = loss_fn(y_true, y_pred)
        regularization_loss = tf.reduce_sum([tf.reduce_sum(tf.square(var)) for var in model.trainable_variables]) * l2_factor
        total_loss = tf.reduce_mean(weights*loss_value)+regularization_loss
        return total_loss
    return loss_with_regularization

def adjust_Lr(hyperparameters, epoch, optimizer, start=0.000000001):
    if epoch < hyperparameters["warmup_epochs"]:
        warmup_lr = interp(epoch, [0, hyperparameters["warmup_epochs"] - 1], [start, hyperparameters["LR"]])
        optimizer.learning_rate.assign(warmup_lr)



#def datafeederSAMP(hyperparameters):
    # Create an instance of the data feeder
    #train_data_feeder = DataFeederSAMP(input_shape=hyperparameters["input_shape"], batch_size=hyperparameters["batch_size"])
    #test_data_feeder = DataFeederSAMP(input_shape=hyperparameters["input_shape"], batch_size=hyperparameters["batch_size"])
    #return train_data_feeder, test_data_feeder

def train(hyperparameters, resume_training=False):
    
    
        train_feeder, test_feeder = datafeeder(path=hyperparameters['dataset_path'], 
                                               train_split=hyperparameters['train_split'], 
                                               batch_size=hyperparameters['batch_size'], 
                                               image_size=hyperparameters['input_shape'][0])
        #train_data_feeder, test_data_feeder = datafeederSAMP(hyperparameters)

        overall_feature_rankings_x2=[]
        
        warnings.filterwarnings('ignore', category=UserWarning)
        model = ResNetFusionModel(input_shape=hyperparameters["input_shape"])
        optimizer = tf.keras.optimizers.Adam(learning_rate = hyperparameters["LR"])
        loss = tf.keras.losses.MeanAbsoluteError()
        #loss = tf.keras.losses.MeanSquaredLogarithmicError()
        # loss = tf.keras.losses.MeanSquaredLogarithmicError()
        loss_fn = regularized_loss(loss, model, l1_factor=0.01, l2_factor=0.06)

        train_loss = tf.keras.metrics.Mean()
        train_mse = tf.keras.metrics.MeanSquaredError()
        train_mae = tf.keras.metrics.MeanAbsoluteError()

        test_loss = tf.keras.metrics.Mean()
        test_mse = tf.keras.metrics.MeanSquaredError()
        test_mae = tf.keras.metrics.MeanAbsoluteError()

        weights_dir = "weights_fusion_lc_nume_weighted_1000ep_accuracy_rfe"
        os.makedirs(weights_dir, exist_ok=True)
        
        # Create a checkpoint manager to save checkpoints
        #checkpoint_dir = "checkpoints_aug_weighted_lc"
        #os.makedirs(checkpoint_dir, exist_ok=True)
        #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        #checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=hyperparameters["epochs"])

        log_file_path = "metrics/metrics_fusion_lc_nume_weighted_1000ep_accuracy_rfe.csv"
        if os.path.isfile(log_file_path):
            log_file = open(log_file_path, "a", newline="")
            writer = csv.writer(log_file)
        else:
            log_file = open(log_file_path, "w", newline="")
            writer = csv.writer(log_file)
            writer.writerow(["Epoch", "Train Loss", "Train MSE", "Train MAE", "Test Loss", "Test MSE", "Test MAE", "learning rate", "Batch size"])
        
        # Resume training if requested
        #if resume_training:
            #latest_checkpoint = checkpoint_manager.latest_checkpoint
            #if latest_checkpoint:
                #checkpoint.restore(latest_checkpoint)
                #loaded_epoch = int(latest_checkpoint.split('-')[-1])
                #start_epoch = loaded_epoch 
                #print(f"Resumed training from checkpoint: {latest_checkpoint}")
            #else:
                #print("No checkpoints found. Starting training from scratch.")
                #start_epoch = 1
        #else:
        start_epoch = 1
        

        epochs = hyperparameters["epochs"]

        trainprint = 25

        # Create a directory for TensorBoard logs
        log_dir = "logs_fusion_lc_nume_weighted_1000ep_accuracy_rfe"
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=log_dir)

        for epoch in range(start_epoch, epochs + 1):
            print(f"Epoch {epoch }/{epochs}")
            
            [x1, x2_covariates],y,z = train_feeder.__getitem__(0)
            feature_importances_x2 = []
            #rfe_estimator = RandomForestRegressor()  # You can choose the appropriate estimator
            #rfe_x2 = RFE(estimator=rfe_estimator, n_features_to_select=10)  # Selecting top 5 features
            #x2_covariates_reshaped = x2_covariates.reshape(x2_covariates.shape[0], -1)  # Reshape to 2D
            #rfe_x2.fit(x2_covariates_reshaped, y)  # Use y_batch as the target variable
            # Get the selected features for x2_covariates
            #selected_features_indices_x2 = rfe_x2.support_
            #selected_covariates_x2 = x2_covariates[:, selected_features_indices_x2]
        
            # Get the importance values from the RFE estimator
            #feature_importances_x2.append(rfe_x2.estimator_.feature_importances_)
            #overall_feature_rankings_x2.append(selected_features_indices_x2)
            
            adjust_Lr(hyperparameters, epoch, optimizer)
            train_loss.reset_states()
            
            train_mse.reset_states()
            train_mae.reset_states()
            start_time = time.time()
            batch_num=0
            for [image_batch, numeric_batch], labels_batch, weights in tqdm(train_feeder, desc='Train'):
                batch_num+=1
                image_batch = image_batch.astype(np.float32)
                numeric_batch = numeric_batch.astype(np.float32)
                weights_tensor = tf.constant(weights, dtype= tf.float32)
                rfe_estimator = RandomForestRegressor()  # You can choose the appropriate estimator
                rfe_x2 = RFE(estimator=rfe_estimator, n_features_to_select=10)  # Selecting top 10 features
                x2_covariates_reshaped = x2_covariates.reshape(x2_covariates.shape[0], -1)  # Reshape to 2D
                rfe_x2.fit(x2_covariates_reshaped, y)
                # Train RandomForestRegressor using selected features
                selected_features_indices_x2 = rfe_x2.support_
                selected_covariates_x2 = x2_covariates[:, selected_features_indices_x2]
                selected_covariates_x2_reshaped = selected_covariates_x2.reshape(selected_covariates_x2.shape[0], -1)
                rfe_x2.estimator_.fit(selected_covariates_x2_reshaped, y)
                feature_importances_x2.append(rfe_x2.estimator_.feature_importances_)
                with tf.GradientTape() as tape:
                    
                        
                            
                    logits = model([image_batch, numeric_batch])
                    loss_value = loss_fn(labels_batch, logits, weights_tensor)
                    

                gradients = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients([
                (grad, var) 
                for (grad, var) in zip(gradients, model.trainable_variables) 
                if grad is not None
                ])

                train_loss(loss_value)
                
                train_mse(labels_batch, logits)
                train_mae(labels_batch, logits)
            train_feeder.on_epoch_end()

            for [image_batch, numeric_batch], labels_batch, weights in tqdm(test_feeder, desc='Test'):
                image_batch = image_batch.astype(np.float32)
                numeric_batch = numeric_batch.astype(np.float32)
                weights_tensor = tf.constant(weights, dtype=tf.float32)

                test_logits = model([image_batch, numeric_batch])
                test_loss_value = loss_fn(labels_batch, test_logits, weights_tensor)
                test_loss(test_loss_value)
                
                test_mse(labels_batch, test_logits)
                test_mae(labels_batch, test_logits)

            if epoch % hyperparameters["save_weight_epoch"] == 0:
                #checkpoint_manager.save()
                model_filename = f"resnet_fusion_model_rfe_lc_epoch{epoch + 1}_acc{test_mae.result().numpy():.4f}.h5"
                model.save_weights(os.path.join(weights_dir, model_filename))

            writer.writerow([epoch + 1, train_loss.result().numpy(),  train_mse.result().numpy(), train_mae.result().numpy(),test_loss.result().numpy(),  test_mse.result().numpy(), test_mae.result().numpy(), hyperparameters["LR"], hyperparameters["batch_size"]])
            elapsed_time = time.time() - start_time
            print(f"Time taken: {round(elapsed_time/60, 2)} minutes")
            print(f"Epoch: {epoch}, Train Loss: {train_loss.result()}, Train MSE: {train_mse.result()}, Train MAE: {train_mae.result()}") 
            print(f"Epoch: {epoch}, Test  Loss: {test_loss.result()},  Test MSE:    {test_mse.result()}, Test MAE: {test_mae.result()}")

            # Update TensorBoard logs
            with tf.summary.create_file_writer(log_dir).as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                #tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
                tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
                #tf.summary.scalar('test_accuracy', test_accuracy.result(), step=epoch)

            tensorboard_callback.on_epoch_end(epoch)
            
            # Calculate mean importance value for each feature across all batches
            mean_importance_values = np.mean(feature_importances_x2, axis=0)
            # Create a dictionary to store feature names and their corresponding mean importance values
            feature_importance_dict = {}
            for i, feature_name in enumerate(["pdsi", "pet", "ppt", "q", "soil", "tmax", "tmin", "vap", "vpd", "ws"]):
                feature_importance_dict[feature_name] = mean_importance_values[i]
            # Sort the features based on their mean importance values in descending order
            sorted_features = sorted(feature_importance_dict.keys(), key=lambda x: feature_importance_dict[x], reverse=True)

            # Print the ranked features along with their mean importance values
            print("Ranked Features based on Mean Importance:")
            for feature in sorted_features:
                importance_value = feature_importance_dict[feature]
                print(f"Feature '{feature}': Mean Importance Value = {importance_value}")

        log_file.close()
        
    

if __name__ == '__main__':
    hyperparameters = {
        "input_shape": (512, 512, 3),
        "batch_size": 16,
        "epochs": 1000,
        "save_weight_epoch": 1,
        "dataset_path":'Sentinel_2',
        "LR": 0.0001,
        "warmup_epochs": 5,
        "train_split":0.8
    }

    main_folder = "Sentinel_2"
    countries = ["australia", "costarica_new2", "southafrica"]
    csv_file = "Sentinel_2/updated_reduced_final_bl_duplicated.csv"
    train_split = 0.8
    is_train = True
    resume_training=False

    train(hyperparameters,resume_training=False)