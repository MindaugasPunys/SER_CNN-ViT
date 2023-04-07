# Import libraries
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
# import IPython.display as ipd  # To play sound in the notebook
from playsound import playsound

import sys
import warnings
import cv2
from PIL import Image

import keras
from keras import regularizers
from keras.preprocessing import sequence
# from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LeakyReLU, ELU
from keras import callbacks
from keras import optimizers

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other
import librosa
import librosa.display
import json
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import os
import pickle

# ViT libraries
from tensorflow import image as tfi
from keras.layers import Normalization
from keras.layers import Resizing
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from keras.layers import RandomZoom
# Model
from tensorflow.nn import gelu
from keras.models import Model
from keras.layers import Dense
from keras.layers import Layer
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import MultiHeadAttention
from keras.layers import LayerNormalization
from keras.layers import Add
from keras.layers import Flatten
# callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
# compile
from keras.losses import SparseCategoricalCrossentropy as SCCe
from tensorflow_addons.optimizers import AdamW
from keras.metrics import SparseCategoricalAccuracy as Acc
from keras.metrics import SparseTopKCategoricalAccuracy as KAcc

# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)])
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# _________________________________________________________________________________________________

# some global config
GLB_READ_DATA = False
GLB_DISPLAY_DATA = False
GLB_USE_1D_CNN = False  # todo
GLB_MODE = 5 # 1 - 1D CNN create; 2 - 1D CNN load; 3 - 2D CNN create; 4 - 2D CNN load;

# Database files:
TESS = "toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
RAV = "ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
SAVEE = "surrey-audiovisual-expressed-emotion-savee/ALL/"
CREMA = "cremad/AudioWAV/"

ALL_PATH_CSV = "Data_path.csv"
MFCC_PATH_CSV = 'Data_MFCC_path.csv'
# _________________________________________________________________________________________________

def ReadData_SAVEE():
    print("________________________________________ SAVEE read data ________________________________________")
    # Get the data location for SAVEE
    dir_list = os.listdir(SAVEE)

    # parse the filename to get the emotions
    emotion = []
    path = []
    for i in dir_list:
        if i[-8:-6] == '_a':
            emotion.append('angry')
        elif i[-8:-6] == '_d':
            emotion.append('disgust')
        elif i[-8:-6] == '_f':
            emotion.append('fear')
        elif i[-8:-6] == '_h':
            emotion.append('happy')
        elif i[-8:-6] == '_n':
            emotion.append('neutral')
        elif i[-8:-6] == 'sa':
            emotion.append('sad')
        elif i[-8:-6] == 'su':
            emotion.append('surprise')
        else:
            emotion.append('male_error')
        path.append(SAVEE + i)

    # Now check out the label count distribution
    SAVEE_df = pd.DataFrame(emotion, columns=['labels'])
    SAVEE_df['source'] = 'SAVEE'
    SAVEE_df = pd.concat(
        [SAVEE_df, pd.DataFrame(path, columns=['path'])], axis=1)
    print(SAVEE_df.labels.value_counts())
    print("\n")

    if GLB_DISPLAY_DATA:
        # use the well known Librosa library for this task
        fname = SAVEE + 'DC_f11.wav'
        data, sampling_rate = librosa.load(fname)
        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(data, sr=sampling_rate)
        plt.show()
        # Lets play the audio
        playsound(fname)

    return SAVEE_df
def ReadData_RAVDESS():
    print("________________________________________ RAVDESS read data ________________________________________")
    dir_list = os.listdir(RAV)
    dir_list.sort()

    emotion = []
    gender = []
    path = []
    for i in dir_list:
        fname = os.listdir(RAV + i)
        for f in fname:
            part = f.split('.')[0].split('-')
            emotion.append(int(part[2]))
            temp = int(part[6])
            if temp % 2 == 0:
                temp = "female"
            else:
                temp = "male"
            gender.append(temp)
            path.append(RAV + i + '/' + f)

    RAV_df = pd.DataFrame(emotion)
    RAV_df = RAV_df.replace({1: 'neutral', 2: 'neutral', 3: 'happy',
                            4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'})
    RAV_df = pd.concat([pd.DataFrame(gender), RAV_df], axis=1)
    RAV_df.columns = ['gender', 'emotion']
    # RAV_df['labels'] = RAV_df.gender + '_' + RAV_df.emotion
    RAV_df['labels'] = RAV_df.emotion # REMOVED GENDER
    RAV_df['source'] = 'RAVDESS'
    RAV_df = pd.concat([RAV_df, pd.DataFrame(path, columns=['path'])], axis=1)
    RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
    print(RAV_df.labels.value_counts())
    print("\n")

    # Pick a fearful track
    if GLB_DISPLAY_DATA:
        fname = RAV + 'Actor_14/03-01-06-02-02-02-14.wav'
        data, sampling_rate = librosa.load(fname)
        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(data, sr=sampling_rate)
        plt.show()

        # Lets play the audio
        # playsound(fname) # FILE NEME IS TOO LONG??? WHAT???

    dir_list = os.listdir(TESS)
    dir_list.sort()
    print(dir_list)
    print("\n")

    return RAV_df
def ReadData_TESS():
    print("________________________________________ TESS read data ________________________________________")
    dir_list = os.listdir(TESS)
    dir_list.sort()
    print(dir_list)

    path = []
    emotion = []

    for i in dir_list:
        fname = os.listdir(TESS + i)
        for f in fname:
            if i == 'OAF_angry' or i == 'YAF_angry':
                emotion.append('angry')
            elif i == 'OAF_disgust' or i == 'YAF_disgust':
                emotion.append('disgust')
            elif i == 'OAF_Fear' or i == 'YAF_fear':
                emotion.append('fear')
            elif i == 'OAF_happy' or i == 'YAF_happy':
                emotion.append('happy')
            elif i == 'OAF_neutral' or i == 'YAF_neutral':
                emotion.append('neutral')
            elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
                emotion.append('surprise')
            elif i == 'OAF_Sad' or i == 'YAF_sad':
                emotion.append('sad')
            else:
                emotion.append('Unknown')
            path.append(TESS + i + "/" + f)

    TESS_df = pd.DataFrame(emotion, columns=['labels'])
    TESS_df['source'] = 'TESS'
    TESS_df = pd.concat(
        [TESS_df, pd.DataFrame(path, columns=['path'])], axis=1)
    print(TESS_df.labels.value_counts())
    print("\n")

    return TESS_df
def ReadData_CREMA():
    print("________________________________________ CREMA read data ________________________________________")

    dir_list = os.listdir(CREMA)
    dir_list.sort()
    print(dir_list[0:10])

    gender = []
    emotion = []
    path = []
    female = [1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018, 1020, 1021, 1024, 1025, 1028, 1029, 1030, 1037, 1043, 1046, 1047, 1049,
              1052, 1053, 1054, 1055, 1056, 1058, 1060, 1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079, 1082, 1084, 1089, 1091]

    for i in dir_list:
        part = i.split('_')
        if int(part[0]) in female:
            temp = 'female'
        else:
            temp = 'male'
        gender.append(temp)
        if part[2] == 'SAD' and temp == 'male':
            emotion.append('sad')
        elif part[2] == 'ANG' and temp == 'male':
            emotion.append('angry')
        elif part[2] == 'DIS' and temp == 'male':
            emotion.append('disgust')
        elif part[2] == 'FEA' and temp == 'male':
            emotion.append('fear')
        elif part[2] == 'HAP' and temp == 'male':
            emotion.append('happy')
        elif part[2] == 'NEU' and temp == 'male':
            emotion.append('neutral')
        elif part[2] == 'SAD' and temp == 'female':
            emotion.append('sad')
        elif part[2] == 'ANG' and temp == 'female':
            emotion.append('angry')
        elif part[2] == 'DIS' and temp == 'female':
            emotion.append('disgust')
        elif part[2] == 'FEA' and temp == 'female':
            emotion.append('fear')
        elif part[2] == 'HAP' and temp == 'female':
            emotion.append('happy')
        elif part[2] == 'NEU' and temp == 'female':
            emotion.append('neutral')
        else:
            emotion.append('Unknown')
        path.append(CREMA + i)

    CREMA_df = pd.DataFrame(emotion, columns=['labels'])
    CREMA_df['source'] = 'CREMA'
    CREMA_df = pd.concat(
        [CREMA_df, pd.DataFrame(path, columns=['path'])], axis=1)
    print(CREMA_df.labels.value_counts())
    print("\n")

    if GLB_DISPLAY_DATA:
        fname = CREMA + '1012_IEO_HAP_HI.wav'
        data, sampling_rate = librosa.load(fname)
        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(data, sr=sampling_rate)
        plt.show()

        # Lets play the audio
        playsound(fname)

    return CREMA_df
def DataframesToCsv(SAVEE_df, RAV_df, TESS_df, CREMA_df):
    df_all_DB = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis=0)
    print(df_all_DB.labels.value_counts())
    print(df_all_DB.head())
    df_all_DB.to_csv(ALL_PATH_CSV, index=False)
def MFCC_Example():
    # Source - RAVDESS; Gender - Female; Emotion - Angry
    path = "ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_08/03-01-05-02-01-01-08.wav"
    X, sample_rate = librosa.load(
        path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    # audio wave
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(X, sr=sample_rate)
    plt.title('Audio sampled at 44100 hrz')
    plt.show()
    # MFCC
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.ylabel('MFCC')
    # plt.colorbar()
    plt.show()
    # playsound(path) # PAth ToO lOnG

    # Source - RAVDESS; Gender - Male; Emotion - Angry
    path = "ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_09/03-01-05-01-01-01-09.wav"
    X, sample_rate = librosa.load(
        path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    # audio wave
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(X, sr=sample_rate)
    plt.title('Audio sampled at 44100 hrz')
    # MFCC
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.ylabel('MFCC')
    # playsound(path)
def ReadData_Path_CSV():
    # lets pick up the meta-data that we got from our first part of the Kernel
    ref = pd.read_csv("Data_path.csv")
    # print(ref.head())
    return ref
def DataframeAddMFCC(ref):
    # Note this takes a couple of minutes (~10 mins) as we're iterating over 4 datasets
    # loop feature extraction over the entire dataset
    df_all_DB = pd.DataFrame(columns=['feature'])

    counter = 0
    for index, path in enumerate(ref.path):
        X, sample_rate = librosa.load(
            path, res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)
        sample_rate = np.array(sample_rate)

        # mean as the feature. Could do min and max etc as well.
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=13), axis=0)
        df_all_DB.loc[counter] = [mfccs]
        counter = counter+1

        if counter % 100 == 0:
            print(f"Progress(MFCC): {counter}", end="\r")

    # Check a few records to make sure its processed successfully
    print(len(df_all_DB))
    df_all_DB.head()

    # Now extract the mean bands to its own feature columns
    df_all_DB = pd.concat(
        [ref, pd.DataFrame(df_all_DB['feature'].values.tolist())], axis=1)

    # replace NA with 0
    df_all_DB = df_all_DB.fillna(0)
    print(df_all_DB.shape)
    print(df_all_DB[:5])

    df_all_DB.to_csv(MFCC_PATH_CSV, index=False)

    return df_all_DB
def PrepareData(df_all_DB):
    # Split between train and test
    df_split = df_all_DB
    X_train, X_test, y_train, y_test = train_test_split(df_split.drop(
        ['path', 'labels', 'source'], axis=1), df_split.labels, test_size=0.25, shuffle=True, random_state=4)
    # Lets see how the data present itself before normalisation
    print(X_train[150:160])

    # Lts do data normalization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std

    # Check the dataset now
    X_train[150:160]

    # Lets few preparation steps to get it into the correct format for Keras
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # one shot encode the target
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    print(X_train.shape)
    print(lb.classes_)
    # print(y_train[0:10])
    # print(y_test[0:10])

    # Pickel the lb object for future use
    filename = 'labels'
    outfile = open(filename, 'wb')
    pickle.dump(lb, outfile)
    outfile.close()

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    X_train.shape

    return X_train, X_test, y_train, y_test, lb
def CNN_1D_Create(X_train, Y_train):
    # New model
    class_count = len(Y_train[0])
    
    model = Sequential()
    # X_train.shape[1] = No. of Columns
    model.add(Conv1D(256, 8, padding='same',
              input_shape=(X_train.shape[1], 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(class_count))  # Target class number
    model.add(Activation('softmax'))
    # opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    # opt = keras.optimizers.Adam(lr=0.0001)
    opt = keras.optimizers.legacy.RMSprop(lr=0.00001, decay=1e-6)
    model.summary()
    return model, opt
def CNN_1D_Train(model, opt, X_train, y_train, X_test, y_test):
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    model_history = model.fit(
        X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))
    # NOTE: I degressesed epeoch count for initial testing
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return model
def CNN_Save(model, model_name):
    # Save model and weights
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Save model and weights at %s ' % model_path)

    # Save the model to disk
    model_json = model.to_json()
    with open("model_json.json", "w") as json_file:
        json_file.write(model_json)
def CNN_1D_Load(model_name, X_test, y_test):
    # loading json and model architecture
    json_file = open('model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("saved_models/" + model_name)
    print("Loaded model from disk\n")

    # Keras optimiser
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer=opt, metrics=['accuracy'])
    score = loaded_model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    return loaded_model
def CNN_1D_Predictions(model, lb, X_test, y_test):
    preds = model.predict(X_test, batch_size=16, verbose=1)
    preds = preds.argmax(axis=1)
    preds
    # predictions
    preds = preds.astype(int).flatten()
    preds = (lb.inverse_transform((preds)))
    preds = pd.DataFrame({'predictedvalues': preds})

    # Actual labels
    actual = y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform((actual)))
    actual = pd.DataFrame({'actualvalues': actual})

    # Lets combined both of them into a single dataframe
    finaldf = actual.join(preds)
    print(finaldf[170:180])

    # Write out the predictions to disk
    finaldf.to_csv('Predictions.csv', index=False)
    finaldf.groupby('predictedvalues').count()

    # Get the predictions file
    finaldf = pd.read_csv("Predictions.csv")
    classes = finaldf.actualvalues.unique()
    classes.sort()

    # Confusion matrix
    c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)
    print(accuracy_score(finaldf.actualvalues, finaldf.predictedvalues))
    # print_confusion_matrix(c, class_names=classes)

    # Classification report
    classes = finaldf.actualvalues.unique()
    classes.sort()
    print(classification_report(finaldf.actualvalues,
          finaldf.predictedvalues, target_names=classes))

    modidf = finaldf
    # modidf['actualvalues'] = finaldf.actualvalues.replace({'female_angry': 'female', 'female_disgust': 'female', 'female_fear': 'female', 'female_happy': 'female', 'female_sad': 'female', 'female_surprise': 'female', 'female_neutral': 'female', 'male_angry': 'male', 'male_fear': 'male', 'male_happy': 'male', 'male_sad': 'male', 'male_surprise': 'male', 'male_neutral': 'male', 'male_disgust': 'male'
    #                                                        })

    # modidf['predictedvalues'] = finaldf.predictedvalues.replace({'female_angry': 'female', 'female_disgust': 'female', 'female_fear': 'female', 'female_happy': 'female', 'female_sad': 'female', 'female_surprise': 'female', 'female_neutral': 'female', 'male_angry': 'male', 'male_fear': 'male', 'male_happy': 'male', 'male_sad': 'male', 'male_surprise': 'male', 'male_neutral': 'male', 'male_disgust': 'male'
    #                                                              })

    # classes = modidf.actualvalues.unique()
    # classes.sort()

    # # Confusion matrix
    # c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)
    # print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))
    # print_confusion_matrix(c, class_names=classes)

    # Classification report
    classes = modidf.actualvalues.unique()
    classes.sort()
    print(classification_report(modidf.actualvalues,
          modidf.predictedvalues, target_names=classes))

    modidf = pd.read_csv("Predictions.csv")
    modidf['actualvalues'] = modidf.actualvalues.replace({'female_angry': 'angry', 'female_disgust': 'disgust', 'female_fear': 'fear', 'female_happy': 'happy', 'female_sad': 'sad', 'female_surprise': 'surprise', 'female_neutral': 'neutral', 'male_angry': 'angry', 'male_fear': 'fear', 'male_happy': 'happy', 'male_sad': 'sad', 'male_surprise': 'surprise', 'male_neutral': 'neutral', 'male_disgust': 'disgust'
                                                          })

    modidf['predictedvalues'] = modidf.predictedvalues.replace({'female_angry': 'angry', 'female_disgust': 'disgust', 'female_fear': 'fear', 'female_happy': 'happy', 'female_sad': 'sad', 'female_surprise': 'surprise', 'female_neutral': 'neutral', 'male_angry': 'angry', 'male_fear': 'fear', 'male_happy': 'happy', 'male_sad': 'sad', 'male_surprise': 'surprise', 'male_neutral': 'neutral', 'male_disgust': 'disgust'
                                                                })

    classes = modidf.actualvalues.unique()
    classes.sort()

    # Confusion matrix
    c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)
    print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))
    # print_confusion_matrix(c, class_names=classes)

    # Classification report
    classes = modidf.actualvalues.unique()
    classes.sort()
    print(classification_report(modidf.actualvalues,
          modidf.predictedvalues, target_names=classes))

def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    # the confusion matrix heat map plot
    # Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", ax=None) # Matplotlib not supported
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
def gender(row):
    # Gender recode funct
    if row == 'female_disgust' or 'female_fear' or 'female_happy' or 'female_sad' or 'female_surprise' or 'female_neutral':
        return 'female'
    elif row == 'male_angry' or 'male_fear' or 'male_happy' or 'male_sad' or 'male_surprise' or 'male_neutral' or 'male_disgust':
        return 'male'

# _________________________________________________________________________________________________
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled
def spectrogram_image(y, sr, out_dir, out_name, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    # mels = librosa.feature.melspectrogram(y=y, sr=sr)
    
    if 1:
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    else:  #testing !
        mels = np.mean(mels, axis=0)
        
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255 - img            # invert. make black==more energy
    
    # save as PNG
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cv2.imwrite((out_dir + "//" + out_name), img)
def save_wav_to_png(df, DATA_SAMPLES_CNT, BASE_PATH, IMG_HEIGHT, IMG_WIDTH, use_Kfold = False):
    """ 
    Saves spectograms data from sound files as png pictures
    """
    print("Saving pictures to drive")
    for i in range(DATA_SAMPLES_CNT):
        file_name = BASE_PATH  + "//audio//" + str(df["filename"][i])
        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        
        img_name = 'out' + str(i+1) + "_" + str(df["target"][i]) + '.png'
        hop_length = 512           # number of samples per time-step in spectrogram
        n_mels = IMG_HEIGHT        # number of bins in spectrogram. Height of image
        time_steps = IMG_WIDTH - 1 # number of time-steps. Width of image (TODO FIX it add 1 px to width!!)
        
        y = librosa.util.utils.fix_length(y, sr * 2.5)
        
        start_sample = 0 # starting at beginning
        length_samples = time_steps * hop_length
        window = y[start_sample:start_sample+length_samples]
        dir_name = "mel_img"
        
        spectrogram_image(y=window, sr=sr, out_dir=dir_name , out_name=img_name, hop_length=hop_length, n_mels=n_mels)
    print("Done saving pictures!")

def CNN_2D_ProcessData(ref):
    # Note this takes a couple of minutes (~10 mins) as we're iterating over 4 datasets 
    df = pd.DataFrame(columns=['feature'])
    # loop feature extraction over the entire dataset
    counter=0
    for index,path in enumerate(ref.path):
        X_mel, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=44100,offset=0.5)
        sample_rate = np.array(sample_rate)
        
        IMG_HEIGHT = 256    
        IMG_WIDTH = 256
        
        img_name = 'out_nr' + str(counter+1) + '.png'
        hop_length = 512           # number of samples per time-step in spectrogram
        n_mels = IMG_HEIGHT        # number of bins in spectrogram. Height of image
        time_steps = IMG_WIDTH - 1 # number of time-steps. Width of image (TODO FIX it add 1 px to width!!)
            
        X_mel = librosa.util.utils.fix_length(X_mel, size=110250) # 2.5 * sr
            
        start_sample = 0 # starting at beginning
        length_samples = time_steps * hop_length
        window = X_mel[start_sample:start_sample+length_samples]
        dir_name = "mel_img"
            
        spectrogram_image(y=window, sr=sample_rate, out_dir=dir_name , out_name=img_name, hop_length=hop_length, n_mels=n_mels)
        
        counter=counter+1
        if(counter % 100 == 0):
            print(f"Progress(Mel spectrogram): {counter}", end="\r")
    # Check a few records to make sure its processed successfully
    print(len(df))
    print(df.head(10))
    return df
def CNN_2D_DisplayData(y_train, y_test):
    # show some pics
    pic_cnt = 0
    path = "mel_img"

    for filename in os.listdir(path):
        if filename.endswith(".png"):
            image_path = os.path.join(path, filename)
            image = Image.open(image_path)
            image.show()
            print(image_path)
            
            pic_cnt += 1
            if pic_cnt > 5:
                break
    
    print(y_train.size)
    print(y_train.shape)
    print(y_train)
    print(y_test.size)
    print(y_test.shape)
    print(y_test)
    
    png_count = 0
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            png_count = png_count + 1
    print(png_count)
def CNN_2D_Label(df_all_DB):
    lb = LabelEncoder()
    Y_mell = np_utils.to_categorical(lb.fit_transform(df_all_DB.labels))
    # y_test_mell = np_utils.to_categorical(lb.fit_transform(y_test))
    print(Y_mell)
    print(Y_mell[0])
    print(Y_mell[1])
    
    # Invert the one-hot encoding
    Y_mell_labels = np.argmax(Y_mell, axis=1)

    # print(Y_mell_labels)
    print(min(Y_mell_labels))
    print(max(Y_mell_labels))
    return Y_mell_labels
   
def CNN_2D_LoadSpectograms(DATA_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH):
    print("Loading images from drive to RAM!")
    img_data_array = np.zeros((DATA_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH))
    
    for i in range(0, DATA_SAMPLES_CNT):
        image_path = "mel_img//out_nr" + str(i+1) + ".png"
        image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # TODO FIX: check color map
        # image= cv2.imread(image_path)
        if image is None:
            print("Error, image was not found from: " + image_path)
            quit()
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        img_data_array[i] = image
    print("Finish loading images from drive to RAM!")
    return img_data_array
def CNN_2D_LoadData(Y_mell_labels, IMG_HEIGHT, IMG_WIDTH):
    samples_cnt = 12162
    # samples_cnt = 8000
    Y_mell_labels = Y_mell_labels[0:samples_cnt]

    X_data_mell = CNN_2D_LoadSpectograms(samples_cnt, IMG_HEIGHT, IMG_WIDTH)
    
    # print(X_data_mell)
    # print(X_data_mell[0])
    
    x_train_mell, x_test_mell, y_train_mell, y_test_mell = train_test_split(X_data_mell, Y_mell_labels, test_size=0.25, random_state=7)
        
    x_train_mell = x_train_mell.reshape(x_train_mell.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    x_test_mell = x_test_mell.reshape(x_test_mell.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    
    return x_train_mell, x_test_mell, y_train_mell, y_test_mell
def CNN_2D_Create(img_h, img_w, class_cnt):
    # Initialize model
    model = Sequential()
    # Layer 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # Layer 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same' ))
    model.add(MaxPooling2D((2, 2)))
    # Layer 3
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.2))
    # Layer 4
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    # Layer 5
    model.add(Flatten())
    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()
    return model
def CNN_2D_FitModel(model, x_train_mell, x_test_mell, y_train_mell, y_test_mell):
    y_train_mell = to_categorical(y_train_mell)
    y_test_mell = to_categorical(y_test_mell)
    
    earlystopper = callbacks.EarlyStopping(patience=10, verbose=1, monitor='val_accuracy')
    checkpointer = callbacks.ModelCheckpoint('saved_models\\2D_CNN_checkpoint.h5', verbose=1, save_best_only=True)
        
    hist = model.fit(x_train_mell, y_train_mell, batch_size=32, epochs=20, verbose=1, validation_data=(x_test_mell, y_test_mell), callbacks = [earlystopper, checkpointer])
    #     draw_model_results(hist)
    return model

class DataAugmentation(Layer):
    def __init__(self, norm, SIZE):
        super(DataAugmentation, self).__init__()
        self.norm = norm
        self.SIZE = SIZE
        self.resize = Resizing(SIZE, SIZE)
        self.flip = RandomFlip('horizontal')
        self.rotation = RandomRotation(factor=0.02)
        self.zoom = RandomZoom(height_factor=0.2, width_factor=0.2)
    
    def call(self, X):
        x = self.norm(X)
        x = self.resize(x)
        x = self.flip(x)
        x = self.rotation(x)
        x = self.zoom(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "norm": self.norm,
            "SIZE": self.SIZE,
        })
        return config
class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0] # Get the Batch Size
        print(batch_size)
        patches = tfi.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1], # only along the Height and Width Dimension
            strides=[1, self.patch_size, self.patch_size, 1], # The next patch should not overlap the previus patch
            rates=[1,1,1,1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):
        config = super().get_config()
        config.update({
            "path-size": self.patch_size,
        })
        return config
class PatchEncoder(Layer):
    
    def __init__(self, num_patches, projection_dims): # Projection dims is  D
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.d = projection_dims

        self.dense = Dense(units=projection_dims)
        self.positional_embeddings = Embedding(input_dim=num_patches, output_dim=projection_dims)

    def call(self, X):
        positions = tf.range(0,limit=self.num_patches, delta=1)
        encoded = self.dense(X) + self.positional_embeddings(positions)
        return encoded
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_paches": self.num_patches,
            "d": self.d,
        })
        return config
class MLP(Layer):
    def __init__(self, units, rate):
        super(MLP, self).__init__()
        self.units = units
        self.rate = rate
        self.layers = [[Dense(unit, activation=gelu), Dropout(rate)] for unit in units]

    def call(self, x):
        for layers in self.layers:
          for layer in layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "rate": self.rate,
        })
        return config
class Transformer(Layer):
    
    def __init__(self, L, num_heads, key_dims, hidden_units):
        super(Transformer, self).__init__()
        self.L = L
        self.heads = num_heads
        self.key_dims = key_dims
        self.hidden_units = hidden_units

        self.norm = LayerNormalization(epsilon=1e-6) # Remember the Params
        self.MHA = MultiHeadAttention(num_heads=num_heads, key_dim=key_dims, dropout=0.1)
        self.net = MLP(units=hidden_units, rate=0.1)
        self.add= Add()

    def call(self, X):
        inputs = X
        x = X
        for _ in range(self.L):
          x = self.norm(x)
          x = self.MHA(x,x) # our Target and the Source element are the same
          y = self.add([x,inputs])
          x = self.norm(y)
          x = self.net(x)
          x = self.add([x,y])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "L": self.L,
            "heads": self.heads,
            "key_dims":self.key_dims,
            "hidden_units":self.hidden_units
        })
        return config



def ViT_Create(img_h, img_w, class_cnt, x_train):
    SIZE = img_h
    PATCH_SIZE = int(SIZE / 8)
    LR = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_PATCHES = (SIZE // PATCH_SIZE) ** 2
    PROJECTION_DIMS = 16
    NUM_HEADS = 4
    HIDDEN_UNITS = [PROJECTION_DIMS*2, PROJECTION_DIMS]
    OUTPUT_UNITS = [128,64]
    
    # Input Layer
    inputs = Input(shape= x_train.shape[1:])

    # Apply Data Aug
    norm = Normalization()
    norm.adapt(x_train)

    x = DataAugmentation(norm, SIZE)(inputs)

    # Get Patches
    x = Patches(PATCH_SIZE)(x)

    # PatchEncoding Network
    x = PatchEncoder(NUM_PATCHES, PROJECTION_DIMS)(x)

    # Transformer Network
    x = Transformer(8, NUM_HEADS, PROJECTION_DIMS, HIDDEN_UNITS)(x)

    # Output Network
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    x = MLP(OUTPUT_UNITS, rate=0.5)(x)

    # Ouput Layer
    outputs = Dense(class_cnt)(x)
    
    with tf.device('/GPU:0'):
        # Model
        model = Model(
            inputs=[inputs],
            outputs=[outputs],
        )

        # Compiling
        model.compile(
            loss=SCCe(from_logits=True),
            optimizer=AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
            metrics=[
                Acc(name="Accuracy"),
                KAcc(5, name="Top-5-Accuracy")
            ]
        )
        model.summary()
    return model
def ViT_FitModel(model, x_train_mell, x_test_mell, y_train_mell, y_test_mell):
    EPOCHS = 10
    # Callbacks
    cbs = [
        ModelCheckpoint("ViT-Model.h5", save_best_only=True),
        EarlyStopping(patience=5, monitor='val_Accuracy', mode='max' ,restore_best_weights=True)
    ]
    
    # Fit
    results = model.fit(
        x_train_mell, y_train_mell,
        epochs=EPOCHS,
        validation_data=(x_test_mell, y_test_mell),
        callbacks=cbs)
    return model


def main():
    # Read all data
    if (GLB_READ_DATA == True):
        SAVEE_df = ReadData_SAVEE()
        RAVDESS_df = ReadData_RAVDESS()
        TESS_df = ReadData_TESS()
        CREMA_df = ReadData_CREMA()
        DataframesToCsv(SAVEE_df, RAVDESS_df, TESS_df, CREMA_df)
        MFCC_Example()
        # Delte specific datasets dataframes to save data
        del SAVEE_df
        del RAVDESS_df
        del TESS_df
        del CREMA_df

    # Read processed data
    ref = ReadData_Path_CSV()
    if not os.path.isfile(MFCC_PATH_CSV):
        df_all_DB = DataframeAddMFCC(ref)
    else:
        df_all_DB = pd.read_csv(MFCC_PATH_CSV)
    X_train, X_test, y_train, y_test, lb = PrepareData(df_all_DB)

    # Create 1D CNN
    if GLB_MODE == 1:
        CNN_1D, CNN_1D_opt = CNN_1D_Create(X_train, y_train)
        CNN_1D = CNN_1D_Train(CNN_1D, CNN_1D_opt, X_train,
                              y_train, X_test, y_test)
        CNN_Save(CNN_1D, 'CNN1D_1.h5')

    # Load 1D CNN
    if GLB_MODE == 2:
        CNN_1D = CNN_1D_Load('CNN1D_1.h5', X_test, y_test)
        CNN_1D_Predictions(CNN_1D, lb, X_test, y_test)

    # Create 2D CNN
    if GLB_MODE == 3:
        IMG_HEIGHT = 256    
        IMG_WIDTH = 216
        if not os.path.isdir('mel_img'):
            df2d = CNN_2D_ProcessData(ref)
        if GLB_DISPLAY_DATA:
            CNN_2D_DisplayData(y_train, y_test)
        labels = CNN_2D_Label(df_all_DB)
        class_count = len(np.unique(labels))
        x_train_mell, x_test_mell, y_train_mell, y_test_mell = CNN_2D_LoadData(labels, IMG_HEIGHT, IMG_WIDTH)
        CNN_2D = CNN_2D_Create(IMG_HEIGHT, IMG_WIDTH, class_count)
        CNN_2D_FitModel(CNN_2D, x_train_mell, x_test_mell, y_train_mell, y_test_mell)
        CNN_Save(CNN_2D, 'CNN2D_1.h5')
        
    # Load 2D CNN
    if GLB_MODE == 4:
        IMG_HEIGHT = 256    
        IMG_WIDTH = 216
        labels = CNN_2D_Label(df_all_DB)
        class_count = len(np.unique(labels))
        x_train_mell, x_test_mell, y_train_mell, y_test_mell = CNN_2D_LoadData(labels, IMG_HEIGHT, IMG_WIDTH)
        y_train_mell = to_categorical(y_train_mell)
        y_test_mell = to_categorical(y_test_mell)
        CNN_2D = CNN_1D_Load('CNN2D_1.h5', x_test_mell, y_test_mell)
        
    # Create ViT
    if GLB_MODE == 5:
        IMG_HEIGHT = 256    
        IMG_WIDTH = 216
        if not os.path.isdir('mel_img'):
            df2d = CNN_2D_ProcessData(ref)
        if GLB_DISPLAY_DATA:
            CNN_2D_DisplayData(y_train, y_test)
        labels = CNN_2D_Label(df_all_DB)
        class_count = len(np.unique(labels))
        x_train_mell, x_test_mell, y_train_mell, y_test_mell = CNN_2D_LoadData(labels, IMG_HEIGHT, IMG_WIDTH)
        VIT = ViT_Create(IMG_HEIGHT, IMG_WIDTH, class_count, x_train_mell)
        VIT = ViT_FitModel(VIT, x_train_mell, x_test_mell, y_train_mell, y_test_mell)
        CNN_Save(VIT, 'VIT_1.h5')
    
    print("FINISHED!\n")


if __name__ == "__main__":
    main()
