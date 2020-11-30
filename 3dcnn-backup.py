import numpy as np
import cv2 as cv
import os
from numpy import newaxis

import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras import layers
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
tf.config.experimental.set_memory_growth(gpus[0], True)

lr = 0.001
batch_size = 8                ## refers to mini-batch Gradient Descent
epoch = 20                       ## 80
image_width = image_height = 64
image_channels = 1
kernel_size = 3
loss_function = sparse_categorical_crossentropy
optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
#optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

verbosity = 2
padding='SAME'
#padding='valid'
strides_cv = (1,1,1)
strides_mx = (2,2,2)
filter_num = 128
pool_size = 2
num_folds = 4
test_size = 0.2
input_shape = (None, 90, image_width, image_height, 1)

data_region_path = './lefteye'

classes = ["1", "2", "3"]

window_size = 90
X = []
Y = []

time_steps = 0
frames = []

print('Data and label extraction...')

data_region = sorted(os.listdir(data_region_path))

for level in data_region:
    level_path = os.path.join(data_region_path, level)
    participant_list = sorted(os.listdir(level_path))

    for participants in participant_list:
        participant_image_list = os.path.join(level_path, participants)
        individual_list = sorted(os.listdir(participant_image_list))

        for image in individual_list:
            # 3D-CNN purpose
            '''
            temp_path = os.path.join(participant_image_list, image)
            cur_image = cv.imread(temp_path)
            cur_image = cv.cvtColor(cur_image, cv.COLOR_BGR2GRAY)
            cur_image = cur_image[:,:, newaxis]
            X.append(cur_image)
            Y.append(float(level))

            #y = [0] * len(classes)
            #Y.append(y)
            '''
            
            #ConvLSTM purpose
            if(time_steps == 90):

                X.append(frames)
                y = [0] * len(classes)
                y[classes.index(level)] = 1
                Y.append(y)
                time_steps = 0
                frames = []
            else:
                time_steps = time_steps + 1
                temp_path = os.path.join(participant_image_list, image)
                cur_image = cv.imread(temp_path)
                cur_image = cv.cvtColor(cur_image, cv.COLOR_BGR2GRAY)
                cur_image = cur_image[:, :, newaxis]
                frames.append(cur_image)
            

X = np.asarray(X)
#X = X[np.newaxis, ::, ::, ::, ::]
print(X.shape)
Y = np.asarray(Y)
print(Y.shape)

print('Data and labels are ready!')

print('train and test data are splitting...')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=True, random_state=0)

X_train = X_train.reshape(X_train.shape[0], 90, 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 90, 64, 64, 1)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

acc_per_fold = []
loss_per_fold = []

#kfold = KFold(n_splits=num_folds, shuffle=True)

#fold_no = 1

def ConvLSTMModel_Test():
    model = tf.keras.models.Sequential()

    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3,3), return_sequences=False, data_format="channels_last", input_shape=(90, 64, 64, 1)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(3, activation="softmax"))

    model.summary()
 
    return model

def ConvLSTMModel():

    model = tf.keras.models.Sequential()

    keras.Input(shape=(None, 90, 64, 64, 1))
    model.add(
            layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", batch_input_shape = (1, None, 64, 64, 1), return_sequences=True))
            #layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", batch_input_shape = (None, 64, 64, 1), return_sequences=True))
    
    model.add(layers.BatchNormalization())
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3,3), padding="same", return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3,3), padding="same", return_sequences=True))
    model.add(layers.BatchNormalization())

    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3,3), padding="same", return_sequences=True))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(filters=1, kernel_size=(3,3,3), activation="sigmoid", padding="same"))
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    return model

def D3Net():

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv3D(128, (3,3,3), activation='relu', padding="same", input_shape=(90,64,64,1)))
    model.add(tf.keras.layers.Conv3D(128, (3,3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2)))

    model.add(tf.keras.layers.Conv3D(128, (3,3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv3D(128, (3,3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2,2,2)))

    model.add(tf.keras.layers.Conv3D(128, (3,3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv3D(128, (3,3,3), activation='relu', padding="same"))
    model.add(tf.keras.layers.Conv3D(128, (3,3,3), activation='relu', padding="same"))

    model.add(layers.GlobalMaxPool3D())

    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.build(input_shape)
    model.summary()

    return model

print('Cross validation and fit...')

acc_per_fold = []
loss_per_fold = []

inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

kfold = KFold(n_splits=10, shuffle=True)

fold_no = 1

#model = Net()
#model = ConvLSTMModel_Test()

model = D3Net()
opt = keras.optimizers.SGD(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

earlystop = EarlyStopping(patience=20)

filepath = "save-model-D3Net-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True)

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, verbose=verbosity, validation_split=0.2, callbacks=[earlystop, checkpoint])
#history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, verbose=verbosity, validation_split=0.2)

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_pred))

score = model.evaluate(y_test, target_test, verbose=1)

plt.plot(history.history['val_loss'])
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('Number of Epoch')

plt.show()
plt.savefig('loss_convlstm.eps', dpi=600, format='eps')

plt.plot(history.history(['val_accuracy']))
plt.title('Validation accuracy history')
plt.ylabel('Accuracy value (percentage)')
plt.xlabel('Number of Epoch')
plt.show()
plt.savefig('accuracy_convlstm.eps', dpi=600, format='eps')
plt.close()


for train, test in kfold.split(inputs, targets):
    print('Training for fold {}'.format(fold_no))
    print(inputs[train].shape)
    print(targets[train].shape)

    history = model.fit(inputs[train], targets[train], batch_size = batch_size, epochs = epoch, verbose=verbosity)
    model.summary()

    scores = model.evaluate(inputs[test], targets[test], verbose=1)
    print(f'Scores for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1]*100)
    loss_per_fold.append(scores[0])

    fold_no = fold_no + 1

print('Score per fold')

for i in range(0, len(acc_per_fold)):
    print(f'Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')

print('Average scores for all folds')
print(f'Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold})')
print(f'Loss: {np.means(loss_per_fold)}'








