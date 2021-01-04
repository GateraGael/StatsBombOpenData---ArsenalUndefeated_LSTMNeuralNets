import StatsBomb_GetArsenal_Undefeated_json
import pandas as pd
import numpy as np
import sys
import os
import datetime
import matplotlib.pyplot as plt
import math
from contextlib import redirect_stdout

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

"""
# Metrics
"""
from itertools import cycle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

"""
Custom made plot for confusion matrix
"""
import plotting_confusion_matrix as plot_conf
import TensorBoard_PlotCM

pd.set_option("display.max_rows", None, "display.max_columns", None)

"""
Importing Tensorflow in order to use GPU's
"""
import tensorflow as tf


# fix random seed for reproducibility
np.random.seed(7)

# Getting the Original Dataframe into it's object
Arsenal_0304_DataFrame = StatsBomb_GetArsenal_Undefeated_json.main()


# Sorting by Match Date and storing into a it's own Dataframe 
Sorted_by_MatchDate = Arsenal_0304_DataFrame.sort_values(by='match_date')

# Drop the old index column and the match_date column since the dataframe is sorted
Sorted_DF = Sorted_by_MatchDate.reset_index().drop(columns=['match_date', 'index'])

DropedDF = Sorted_DF.drop(columns=['competition_name', 'country_name', 'season_year'])

# Here the Results are for Arsenal results: 1 = win, 2 = loss, 0 = draw
# Hardcoded but there is a possibility this could be done using code but I wanted to be fast
results = [1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1]

MatchResult_df = DropedDF

MatchResult_df['y'] = results #= DropedDF

def normalize_series(given_series):
    values = given_series.values
    values = values.reshape((len(values), 1))
    # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    #print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    # normalize the dataset and print
    normalized = scaler.transform(values)

    return normalized

home_scores_norm = normalize_series(MatchResult_df['home_score'])
away_scores_norm = normalize_series(MatchResult_df['away_score'])

NormalizedDF = MatchResult_df

NormalizedDF['home_score_norm'] = home_scores_norm
NormalizedDF['away_score_norm'] = away_scores_norm


NormalizedDF = NormalizedDF.drop(columns=['home_score', 'away_score'])

# Turning Some Features into Categorical Data
# integer encode
def encode_feature(given_series):
    values = np.array(given_series)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    return integer_encoded


encoded_home_team_name = encode_feature(MatchResult_df['home_team_name'])
encoded_away_team_name = encode_feature(MatchResult_df['away_team_name'])
encoded_ko_time = encode_feature(MatchResult_df['kick_off'])

Encoded_df = NormalizedDF

Encoded_df['home_teams_encoded'] = encoded_home_team_name
Encoded_df['away_teams_encoded'] = encoded_away_team_name
Encoded_df['kick_off_encoded'] = encoded_ko_time


Encoded_df = Encoded_df.drop(columns=['home_team_name', 'away_team_name', 'kick_off'])

dataset = Encoded_df.drop(columns=['y'])
labels = pd.Series(Encoded_df['y'])

dataset = dataset.to_numpy()
labels = labels.to_numpy()


# split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size


train_x, test_x, train_y, test_y = dataset[0:train_size,:], \
                                   dataset[train_size:len(dataset), :], \
                                   labels[0:train_size], \
                                   labels[train_size:len(labels)]

train_y = tf.keras.utils.to_categorical(train_y, num_classes=2)
test_y = tf.keras.utils.to_categorical(test_y, num_classes=2)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
testX = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

#with tf.device('/GPU:0'):
config_dict = {"config1": [
    tf.keras.layers.LSTM(units=2, input_shape=(1, trainX.shape[2]),
                         activation='relu'),
    tf.keras.layers.Dense(units=2, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax')],

    "config2": [tf.keras.layers.LSTM(units=1, input_shape=(1, trainX.shape[2]),
                                     activation='relu'),
                tf.keras.layers.Dense(units=1, input_shape=(trainX.shape[1],),
                                      activation='relu'),
                tf.keras.layers.Dense(units=2, activation='softmax')],

    "config3": [tf.keras.layers.LSTM(units=2, input_shape=(1, trainX.shape[2]),
                                     activation='relu'),
                tf.keras.layers.Dense(units=2, activation='relu'),

                tf.keras.layers.Dense(units=2, activation='softmax')],

    "config4": [tf.keras.layers.LSTM(units=trainX.shape[1], input_shape=(1, trainX.shape[2]),
                                     activation='relu'),
                tf.keras.layers.Dense(units=trainX.shape[1], activation='relu'),
                tf.keras.layers.Dense(units=(trainX.shape[1] * 2), activation='relu'),
                #tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=2, activation='softmax')],

    "config5": [tf.keras.layers.LSTM(units=6, input_shape=(1, trainX.shape[2]),
                                     activation='relu'),
                tf.keras.layers.Dense(units=3, input_shape=(trainX.shape[1],),
                                      activation='relu'),
                tf.keras.layers.Dense(units=8, input_shape=(trainX.shape[1],),
                                      activation='relu'),
                #tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=2, activation='softmax')],

    "config6": [tf.keras.layers.LSTM(units=trainX.shape[1], input_shape=(1, trainX.shape[2]),
                                     activation='relu'),
                tf.keras.layers.Dense(units=(trainX.shape[1] * 2),
                                      input_shape=(trainX.shape[1],),
                                      activation='relu'),
                tf.keras.layers.Dense(units=2, activation='softmax')],

    "config7": [tf.keras.layers.LSTM(units=trainX.shape[1], input_shape=(1, trainX.shape[2]),
                                     activation='relu'),
                tf.keras.layers.Dense(units=(trainX.shape[1] * 2),
                                      activation='relu'),
                #tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=2, activation='softmax')],

    "config8": [tf.keras.layers.LSTM(units=8, input_shape=(1, trainX.shape[2]),
                                     activation='relu'),
                tf.keras.layers.Dense(units=2, activation='softmax')]}

for config_key, layer_config in config_dict.items():
    ModelName = f"LSTM Binary Classifier {config_key}"
    log_dir = 'logs' + '/' + ModelName

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(log_dir)


    def log_confusion_matrix(epoch, logs):

        # Use the model to predict the values from the test_images.
        test_pred_raw = model.predict(testX)

        test_pred = np.argmax(test_pred_raw, axis=1)
        #print(type(test_pred[0]))

        test_pred = np.argmax(test_pred_raw, axis=-1)
        #print(type(test_pred[0]))

        all_y_true = np.argmax(test_y, axis=-1)

        testy = test_y.astype(bool)
        testpred = test_pred.astype(bool)

        # Calculate the confusion matrix using sklearn.metrics

        cm = confusion_matrix(all_y_true, testpred)

        figure = TensorBoard_PlotCM.plot_confusion_matrix(cm, class_names=['Draw', 'Win'])
        cm_image = TensorBoard_PlotCM.plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, profile_batch=10000)
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    # create and fit the LSTM network
    model = tf.keras.Sequential(layer_config, name=str(config_key) )

    EPOCHS = 25
    STEPS_PER_EPOCH = test_size
    #VALIDATION_STEPS = 30

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(trainX, train_y, epochs=EPOCHS, batch_size=1, validation_data=(testX, test_y),
              verbose=1, callbacks=[tensorboard_callback, cm_callback])

    predictions = model.predict(testX, batch_size=1, verbose=2)

    scores = model.evaluate(testX, test_y)

    # record mean and min/max of each set of results
    mean_list, min_list, max_list = list(), list(), list()

    # store mean accuracy
    mean_score = np.mean(scores)
    mean_list.append(mean_score)

    # store min and max relative to the mean
    min_score = np.min(scores)
    min_list.append(mean_score - min_score)
    max_score = np.max(scores)
    max_list.append(max_score - mean_score)

    all_y_true = np.argmax(test_y, axis=-1)
    rounded_predictions = np.argmax(predictions, axis=-1)
    all_y_pred = rounded_predictions

    # Metrics
    output_path = log_dir + './Metrics'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Compute and plot ROC curve
    fpr, tpr, threshold = roc_curve(y_true=all_y_true,
                                    y_score=all_y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC' + ' ' + str(config_key))
    plt.legend(loc="lower right")
    plt.savefig(output_path + '/' + 'ROC-AUC' + ' ' + str(config_key) )
    plt.close()

    # Compute and plot Precision-Recall curve
    prec, rec, threshold_pr = precision_recall_curve(y_true=all_y_true,
                                                     probas_pred=all_y_pred)
    prec_rec_auc = auc(rec, prec)

    plt.figure()
    plt.plot(rec, prec, color='darkorange',
             lw=2, label='Precision-Recall curve (area = %0.2f)' % prec_rec_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall AUC' + ' ' + ' ' + str(config_key))
    plt.legend(loc="lower right")
    plt.savefig(output_path + '/' + 'Precision-Recall AUC' + ' ' + str(config_key) )
    plt.close()

    # Confusion Matrix
    confusion_mat = confusion_matrix(y_true=all_y_true,
                                     y_pred=all_y_pred)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]

    for title, normalize in titles_options:
        disp = plot_conf.plot_confusion_matrix1(confusion_mat,
                                                target_names=['Draw',
                                                              'Win'],
                                                cmap=plt.cm.Blues,
                                                normalize=False,
                                                title=title + str(config_key))

        disp.savefig(output_path + '/' + title + ' ' + str(config_key))
        disp.close()

    tn, fp, fn, tp = confusion_matrix([bool(i) for i in all_y_true],
                                      [bool(i) for i in all_y_pred]).ravel()

    # Other Metrics
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    missed_classes = 1 - accuracy
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    precision = tp / (tp + fp)

    # Program to show various ways to read and
    # write data in a file.
    file1 = open(output_path + '/' + "OtherMetrics" + ' ' + str(config_key) + ".csv", "w")

    OtherMetrics = [f"{config_key} \n",
                    f"Accuracy: {accuracy}  \n",
                    f"Missed Classes: {missed_classes} \n",
                    f"TPR: {tpr}  \n",
                    f"FPR: {fpr}  \n",
                    f"F-1 Score: {f1}  \n",
                    f"Precision: {precision}  \n"]

    file1.writelines(OtherMetrics)
    file1.close()  # to change file access modes

    # write the other metrics on a file
    with open(log_dir + '/' + "ModelSummary" + ' ' + str(config_key)  + ".txt", "w") as f:
        f.write(f"Adam learning_rate:{str(opt)} \n")
        with redirect_stdout(f):
            model.summary()




