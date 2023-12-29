#loading the required libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,Flatten,BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.losses import SparseCategoricalCrossentropy
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

#configuring the GPU 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num of GPUs available:", len(physical_devices))

#-----------------------------Generating the embeddings------------------------------------------------------
#loading dataset from tensorflow datasets
(ds_train, ds_validation,ds_test),ds_info = tfds.load(
    'oxford_iiit_pet',
    split=["train[:90%]", "train[90%:]", "test"], #train set is split into validation set and train dataset
    shuffle_files=True,
    as_supervised=True, #[image,label]
    with_info=True)

print("Num training samples-", tf.data.experimental.cardinality(ds_train) )
print("Num val samples-", tf.data.experimental.cardinality(ds_validation) )
print("Num test samples-", tf.data.experimental.cardinality(ds_test) )


#resizing the images of train, validation and test datasets to (224,224)
size = (224, 224)
ds_train = ds_train.map(lambda x,y:(tf.image.resize(x,size), y))
ds_validation = ds_validation.map(lambda x,y:(tf.image.resize(x,size), y))
ds_test = ds_test.map(lambda x,y:(tf.image.resize(x,size), y))

#Dividing the train, validation and test datasets to batch size of 32 
batch_size = 32
ds_train = ds_train.cache().batch(batch_size, drop_remainder=True).prefetch(buffer_size = 10)
ds_validation = ds_validation.cache().batch(batch_size, drop_remainder=True).prefetch(buffer_size = 10)
ds_test = ds_test.cache().batch(batch_size, drop_remainder=True).prefetch(buffer_size = 10)

#Import the ResNet50 base model
base_model = keras.applications.ResNet50(
    weights = "imagenet",
    input_shape = (224,224,3),
    pooling = 'avg',
    include_top =False,  #Remove the ImageNet classifier at the top
)

#Freeze the base model
base_model.trainable = False

base_model.summary()

#Defining the model for generating the embeddings
base_inputs = base_model.layers[0].input
base_outputs = base_model.layers[-1].output

model = keras.Model(inputs=base_inputs, outputs=base_outputs )

def preprocess(images, labels):
  return tf.keras.applications.resnet50.preprocess_input(images), labels

#train embeddings
ds_train = ds_train.map(preprocess)
train_embeddings = ds_train.map(model)

#validation embeddings
ds_validation = ds_validation.map(preprocess)
val_embeddings = ds_validation.map(model)

#test embeddings
ds_test = ds_test.map(preprocess)
test_embeddings = ds_test.map(model)

#---------------------------------------------------K-NN classification------------------------------------------------
#extract labels and images
train_x = ds_train.map(lambda x, y: x)
train_y = ds_train.map(lambda x, y: y)

val_x = ds_validation.map(lambda x, y: x)
val_y = ds_validation.map(lambda x, y: y)

test_x = ds_test.map(lambda x, y: x)
test_y = ds_test.map(lambda x, y: y)

#train dataset
train_x_np = np.stack(list(train_embeddings))
train_y_np = np.stack(list(train_y))

train_x = train_x_np.reshape((-1, train_x_np.shape[-1]))
train_y = train_y_np.flatten()

#test dataset
test_x_np = np.stack(list(test_embeddings))
test_y_np = np.stack(list(test_y))

test_x = test_x_np.reshape((-1, test_x_np.shape[-1]))
test_y = test_y_np.flatten()

# Standardize the features using StandardScaler 
scaler = StandardScaler()
X_train = scaler.fit_transform(train_x)
X_test = scaler.transform(test_x)

# try K=1 through K=30 and record testing accuracy
k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,  train_y)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(test_y, y_pred))

# plot the relationship between K and testing accuracy
sns.lineplot(x = k_range, y = scores, marker = 'o')
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

best_k = k_range[np.argmax(scores)] #Pick the k value that gives the highest accuracy value
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train,  train_y)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(test_y, y_pred)

#generating the confusion matrix
fig,ax = plt.subplots(figsize=(15, 15))
cm = confusion_matrix(test_y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
plt.show()

# Evaluate the performance
report = classification_report(test_y, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

#-------------------------------------------------Multiclass logistic regression-----------------------------------
model.summary()

#classification head
prediction_layer = tf.keras.layers.Dense(37, activation="softmax")
preprocess_input = tf.keras.applications.resnet50.preprocess_input

#build the model end to end
inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = model(x, training=False) #make the feature extraction part of the model non trainable
x = tf.keras.layers.Dropout(0.3)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

#model with classification head
model.summary()

base_learning_rate = 0.005
model.compile(optimizer=SGD(learning_rate= base_learning_rate), 
              loss=SparseCategoricalCrossentropy(), 
              metrics=["accuracy"])

#Training the model
history = model.fit(ds_train,
                    epochs= 40,
                    validation_data=ds_validation)

#Accuracy of trained model using test dataset
loss, accuracy = model.evaluate(ds_test)
print('Test accuracy of regression model:', accuracy)

#plotting the training and validation accuracy/loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


#----------------------------------------------Fine tuning-----------------------------------------
#using data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

#model with data augmentation and dropout
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False) #Letting the batch normalization layers run in inference mode
x = tf.keras.layers.Dropout(0.3)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

#for fine tuning
model.trainable = True

#Training only the layers that extracts the features more specific to the dataset on which the model was trained
tune_start_at = 140
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:tune_start_at]:
  layer.trainable = False

model.compile(optimizer=SGD(learning_rate= base_learning_rate/10), 
              loss=SparseCategoricalCrossentropy(), 
              metrics=["accuracy"])

model.summary()

#Training the model end to end
history_fine = model.fit(ds_train,
                         epochs=40,
                         validation_data=ds_validation)

acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']

#Accuracy of trained model using test dataset
loss, accuracy = model.evaluate(ds_test)
print('Test accuracy :', accuracy)

#plotting the training and validation accuracy/loss
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

