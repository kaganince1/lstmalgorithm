from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
import keras
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint  

num_classes = 3
img_width, img_height, channels = 64, 64, 3

def load_dataset(path, shuffle):
    data = load_files(path,shuffle=shuffle)
    condition_files = np.array(data['filenames'])
    condition_targets = np_utils.to_categorical(np.array(data['target']), num_classes)
    return condition_files, condition_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('CaltechTiny/train', True)
valid_files, valid_targets = load_dataset('CaltechTiny/val', False)
test_files, test_targets = load_dataset('CaltechTiny/test', False)

print('\nThere are %s total images.' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d validation images.' % len(valid_files))
print('There are %d test images.'% len(test_files))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(img_width, img_height))
    # convert PIL.Image.Image type to 3D tensor with shape (48, 48, 3)
    img = np.float32(img)
    img = img/255
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)
    
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/1
valid_tensors = paths_to_tensor(valid_files).astype('float32')/1
test_tensors = paths_to_tensor(test_files).astype('float32')/1
    
x_test = np.array(test_tensors)
y_train = np.array(train_tensors)
val_train = np.array(valid_tensors)    

#Reshape the training and test set
y_train = y_train.reshape(y_train.shape[0], 64, 64, 3)
x_test = x_test.reshape(x_test.shape[0], 64, 64, 3)
val_train = x_test.reshape(val_train.shape[0], 64, 64, 3)

# #Standardization
mean_px = y_train.mean().astype(np.float32)
std_px = y_train.std().astype(np.float32)
y_train = (y_train - mean_px)/(std_px)

y_train = y_train.reshape(y_train.shape[0], 64*64, 3)
x_test = x_test.reshape(x_test.shape[0], 64*64, 3)
val_train = x_test.reshape(val_train.shape[0], 64*64, 3)

seg_length = 64*64
num_class  = 3
    
model = Sequential()
model.add(LSTM(64, return_sequences=True,input_shape=(seg_length, 3),activation='relu', kernel_initializer='random_uniform'))
model.add(LSTM(64, return_sequences=True, activation='relu', kernel_initializer='random_uniform'))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64, activation='relu', kernel_initializer='random_uniform'))  # return a single vector of dimension 32
model.add(Dense(64, activation='linear'))
model.add(Dense(units = num_class, activation = 'softmax'))

opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
filename = 'lstm_model.h5'
checkpoint = ModelCheckpoint(filepath= filename, period = 1, verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')

callbacks_list = [checkpoint]
model.fit(y_train , train_targets, validation_data=(val_train, valid_targets), batch_size = 8, epochs = 3, shuffle=True, callbacks=[checkpoint], verbose=1 )

model = keras.models.load_model(filename)

# get index of predicted label for each image in test set
condition_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in x_test]

# report test accuracy
test_accuracy = 100*np.sum(np.array(condition_predictions)==np.argmax(test_targets, axis=1))/len(condition_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

y_true = np.argmax(test_targets, axis=1)
y_pred = np.array(condition_predictions)

from sklearn.metrics import roc_auc_score
deneme = np_utils.to_categorical(condition_predictions)
auc_score = roc_auc_score(test_targets, deneme)
print('ROC AUC=%.3f' % (auc_score))

conf = confusion_matrix(y_true, y_pred)
print('confussion matrix')
print(conf)