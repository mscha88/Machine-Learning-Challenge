
# coding: utf-8

# In[1]:


## Load all required packages
get_ipython().run_line_magic('pylab', 'inline --no-import-all')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from keras.models import Sequential
from keras import layers
from keras import models
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.metrics import CategoricalAccuracy, categorical_accuracy
import seaborn as sns
plt.style.use('ggplot')


# In[2]:


## EDA and preprocessing


# In[2]:


## Load data
with np.load("training-dataset.npz") as data:
    img = data["x"] ## test dataset
    lbl = data["y"]


# In[3]:


## Check shapes and types of the dataset
print(img.shape)
print(lbl.shape)
print(type(img))
print(type(lbl))


# In[4]:


## Display example digits
plt.figure(figsize=(15,8))
for index, (image, label) in enumerate(zip(img[0:10], lbl[0:10])):
    plt.subplot(2, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
    plt.savefig('sample.pdf')


# In[5]:


## Visualize one instance showing all features (28 x 28 pixels = 784)
# zero (black) indicated that the feature is zero. The darker the line, the higher the number e.g. 254.
# compare this with the picture above
def visualize_input(img, ax):
    ax.imshow(img, cmap='hot')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)

visualize_input(img[1,:].reshape(28,28), ax)

#plt.title(img[3,0])
plt.axis("off")
#plt.show()
plt.savefig('sample1.pdf')

## source: https://www.kaggle.com/darkside92/simple-best-digit-recognizer-with-cnn-top-5/comments#497794


# In[6]:


## Check if the labels are even distributed - this is the case
g = sns.countplot(lbl)


# In[ ]:


# Linear Models


# In[5]:


## Load data again - Note: Data does not have to be binarized 
with np.load("training-dataset.npz") as data:
    img = data["x"] ## test dataset
    lbl = data["y"]


# In[6]:


## Check shapes and types of the dataset
print(img.shape)
print(lbl.shape)
print(type(img))
print(type(lbl))


# In[7]:


## Split into train and test sets (80 vs 20%)
x_train, x_test, y_train, y_test = train_test_split(img, lbl, test_size=0.2, random_state=0)


# In[8]:


# Reshape the images data (each pixel can have a value between 0.0 to 1.0)
norm = MinMaxScaler()
x_train = norm.fit_transform(x_train)
x_test = norm.fit_transform(x_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# In[9]:


## Logistic Regression
# In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option 
# is set to ‘ovr’, and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’. 
logisticRegr = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, verbose = 1, random_state=0)


# In[10]:


# Fit the model
logisticRegr.fit(x_train, y_train)


# In[11]:


# Make predictions with the test set
predictions = logisticRegr.predict(x_test)
print(predictions[0:10,])


# In[12]:


acc = accuracy_score(logisticRegr.predict(x_train), y_train)
print("Train Accuracy: {:.3}".format(acc))


# In[13]:


acc = accuracy_score(predictions, y_test)
print("Test Accuracy: {:.3}".format(acc))


# In[19]:


## SGD Classifier from sklearn
sgd_clf = SGDClassifier(loss='log', random_state=0, verbose=0)
result = sgd_clf.fit(x_train, y_train)


# In[20]:


acc_sgd = accuracy_score(sgd_clf.predict(x_train), y_train)
print("Train Accuracy: {:.3}".format(acc_sgd))


# In[21]:


acc_sgd = accuracy_score(sgd_clf.predict(x_test), y_test)
print("Test Accuracy: {:.3}".format(acc_sgd))


# In[ ]:


## Non-linear models


# In[22]:


## Random Forest Classifier from sklearn
rdf_clf = RandomForestClassifier(random_state=0)
rdf_clf.fit(x_train, y_train)


# In[23]:


acc_rdf = accuracy_score(rdf_clf.predict(x_train), y_train)
print("Train Accuracy: {:.3}".format(acc_rdf))


# In[24]:


acc_rdf = accuracy_score(rdf_clf.predict(x_test), y_test)
print("Test Accuracy: {:.3}".format(acc_rdf))


# In[25]:


## Multi-layer perceptron classifier from sklearn ## TBC: Shall we use different parameters, how can we
# get the validation score
mlp = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', alpha=0.0001,
                    solver='adam', tol=1e-4, random_state=0, validation_fraction=0.1, early_stopping=True,
                    learning_rate_init=.001, verbose=True)


# In[26]:


mlp.fit(x_train, y_train)


# In[27]:


acc_mlp = accuracy_score(mlp.predict(x_train), y_train)
print("Train Accuracy: {:.3}".format(acc_mlp))


# In[28]:


acc_mlp = accuracy_score(mlp.predict(x_test), y_test)
print("Accuracy: {:.3}".format(acc_mlp))


# In[ ]:


## MLP Base - Keras


# In[29]:


## Load data again
with np.load("training-dataset.npz") as data:
    img = data["x"] ## test dataset
    lbl = data["y"]


# In[30]:


## Split into train and test sets (80 vs 20%)
x_train, x_test, y_train, y_test = train_test_split(img, lbl, test_size=0.2, random_state=0)


# In[31]:


## Split into train and val sets (80 vs 20%)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)


# In[32]:


# Change data tyoe to float and normalize pixel data
norm = MinMaxScaler()
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
X = norm.fit_transform(x_train)
X_val = norm.fit_transform(x_val)
X_test = norm.fit_transform(x_test)


# In[35]:


## Encode categorical label data & adjust naming
lb = LabelBinarizer()
Y = lb.fit_transform(y_train)
Y_val = lb.fit_transform(y_val)
#Y_test = lb.fit_transform(y_test)


# In[36]:


## Check the shape
print(X.shape)
print(Y.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
print(Y_test.shape)


# In[37]:


## Define a multi perceptron model
model = Sequential()
# Add two hidden layers, use the relu activation
# Input dimenstion has to be equal to the number of features in our case pixels
model.add(Dense(256, input_dim=784, activation='relu'))
model.add(Dense(128, activation='relu'))
# The final layer uses softmax activation function since we are looking into a multiclass classification problem
model.add(Dense(26, activation='softmax'))
# Use the Adam optimizer. Adam works similar to regular SGD 
lr = 0.001
opt = Adam(lr)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ["accuracy"])


# In[38]:


# training the model and saving metrics in history
history = model.fit(X, Y,
          batch_size=128, epochs=5,
          verbose=1,
          validation_data=(X_val, Y_val))


# In[39]:


# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

plt.tight_layout()


# In[40]:


loss_and_metrics_train = model.evaluate(X, Y, verbose=2)
loss_and_metrics_val = model.evaluate(X_val, Y_val, verbose=2)

print("Train Loss", loss_and_metrics_train[0])
print("Train Accuracy", loss_and_metrics_train[1])
print()
print("Val Loss", loss_and_metrics_val[0])
print("Val Accuracy", loss_and_metrics_val[1])


# In[ ]:


## CNN Model


# In[41]:


## Load data
with np.load("training-dataset.npz") as data:
    img = data["x"] ## test dataset
    lbl = data["y"]


# In[42]:


## Split into train and test sets (80 vs 20%)
x_train, x_test, y_train, y_test = train_test_split(img, lbl, test_size=0.2, random_state=0)


# In[43]:


## Split into train and val sets (80 vs 20%)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)


# In[44]:


# Reshape to 28 x 28 pixels
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255
print(x_train.shape)


# In[45]:


# Reshape to 28 x 28 pixels
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_val = x_val.astype('float32') / 255
print(x_val.shape)


# In[46]:


# Reshape to 28 x 28 pixels
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255
print(x_test.shape)


# In[47]:


## Encode categorical label data
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_val = lb.fit_transform(y_val)
#y_test = lb.fit_transform(y_test)


# In[48]:


# Check the shape after the split
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)


# In[49]:


# Model 1

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Add a Flatten layer to the model
model.add(layers.Flatten())
# Add a Dense layer with 64 units and relu activation
model.add(layers.Dense(64, activation='relu'))
# Add the last Dense layer.
model.add(layers.Dense(26, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ["accuracy"])


# In[50]:


history = model.fit(x_train, y_train,
          batch_size=32, epochs=5,
          verbose=1,
          validation_data=(x_val, y_val))


# In[52]:


# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

plt.tight_layout()


# In[53]:


loss_and_metrics_train = model.evaluate(x_train, y_train, verbose=2)
loss_and_metrics_val = model.evaluate(x_val, y_val, verbose=2)

print("Train Loss", loss_and_metrics_train[0])
print("Train Accuracy", loss_and_metrics_train[1])
print()
print("Train Loss", loss_and_metrics_val[0])
print("Train Accuracy", loss_and_metrics_val[1])


# In[54]:


# Model 2

model2 = models.Sequential()
model2.add(layers.Conv2D(26, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(52, (3, 3), activation='relu'))
# Add a Flatten layer to the model
model2.add(layers.Flatten())
model2.add(layers.Dropout(0.4))
# Add a Dense layer with 64 units and relu activation
model2.add(layers.Dense(64, activation='relu'))
# Add the last Dense layer.
model2.add(layers.Dense(26, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ["accuracy"])


# In[55]:


history = model2.fit(x_train, y_train,
          batch_size=32, epochs=5,
          verbose=1,
          validation_data=(x_val, y_val))


# In[56]:


# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

plt.tight_layout()


# In[57]:


loss_and_metrics_train = model2.evaluate(x_train, y_train, verbose=2)
loss_and_metrics_val = model2.evaluate(x_val, y_val, verbose=2)

print("Train Loss", loss_and_metrics_train[0])
print("Train Accuracy", loss_and_metrics_train[1])
print()
print("Train Loss", loss_and_metrics_val[0])
print("Train Accuracy", loss_and_metrics_val[1])


# In[56]:


## Compare model with different parameters

settings_train = []
settings_val = []

for a in ['tanh', 'relu']:
    for e in [1, 5, 10]:
        model = Sequential()
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation=a, input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation=a))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation=a))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation=a))
        model.add(layers.Dense(26, activation='softmax'))
                
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ["accuracy"])
        model.fit(x_train, y_train, batch_size=32, epochs=e, verbose=2, validation_data=(x_val, y_val))
        score_train = model.evaluate(x_train, y_train, verbose=2)
        score_val = model.evaluate(x_val, y_val, verbose=2)
        settings_train.append((a, e, score_train[0], score_train[1]))
        settings_val.append((a, e, score_val[0], score_val[1]))
        print(settings_train[-1])
        print(settings_val[-1])

best_loss_train = min(settings_train, key=lambda x: x[-2])
best_accuracy_train =  max(settings_train, key=lambda x: x[-1])
best_loss_val = min(settings_val, key=lambda x: x[-2])
best_accuracy_val =  max(settings_val, key=lambda x: x[-1])

print("Best settings according to train loss {}".format(best_loss_train))
print("Best settings according to train accuracy {}".format(best_accuracy_train))
print("---")
print("Best settings according to val loss {}".format(best_loss_val))
print("Best settings according to val accuracy {}".format(best_accuracy_val))


# In[57]:


loss_and_metrics_train = model.evaluate(x_train, y_train, verbose=2)
loss_and_metrics_val = model.evaluate(x_val, y_val, verbose=2)

print("Train Loss", loss_and_metrics_train[0])
print("Train Accuracy", loss_and_metrics_train[1])
print()
print("Train Loss", loss_and_metrics_val[0])
print("Train Accuracy", loss_and_metrics_val[1])


# In[58]:


## Make class predictions
pred_class = model.predict_classes(x_test)


# In[61]:


print(pred_class.shape)


# In[66]:


print(pred_class[0:5])


# In[73]:


y_test_adj = y_test-1


# In[76]:


print(y_test_adj.shape)


# In[75]:


print(y_test_adj[0:5])


# In[77]:


## accuracy per class
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test_adj, pred_class)
acc_per_class = matrix.diagonal()/matrix.sum(axis=1)
print(acc_per_class[0])
d = {}
for i in range(0,26):
    d[i] = round(acc_per_class[i],4)
print(d)


# In[78]:


# Classification report
print(classification_report(y_test_adj, pred_class))


# In[79]:


## Confusion matrix
cm = metrics.confusion_matrix(y_test_adj, pred_class)


# In[83]:


plt.figure(figsize=(20,20))
sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix'
plt.title(all_sample_title, size = 15);
plt.savefig('Letters_NN.png')
plt.show()


# In[ ]:


## Image can be changed


# In[209]:


## Task 2
## Load data
X_test = np.load('test-dataset.npy')


# In[210]:


print(X_test.shape)
print(type(X_test))


# In[211]:


# Reshape and normalize
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_test = X_test.astype('float32') / 255
print(X_test.shape)


# In[173]:


## Load data
with np.load("training-dataset.npz") as data:
    img = data["x"] ## test dataset
    lbl = data["y"]


# In[174]:


## Split into train and test sets (80 vs 20%)
x_train, x_test, y_train, y_test = train_test_split(img, lbl, test_size=0.2, random_state=0)


# In[175]:


## Split into train and val sets (80 vs 20%)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)


# In[180]:


## Shuffle the data - random splits
n = 10000
x_train, y_train = x_train[0:n], y_train[0:n]
x_test, y_test = x_test[0:n], y_test[0:n]
x_val, y_val = x_val[0:n], y_val[0:n]


# In[181]:


# Check the shape after the split
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)


# In[185]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
some_digit = x_train[10]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
plt.show()


# In[186]:


print(y_train[10])


# In[187]:


# Reshape to 28 x 28 pixels
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255
print(x_train.shape)


# In[194]:


a = np.zeros(56).reshape(2,28)
b = np.zeros(4200).reshape(30,140)

x_train_adj = []

for i in range(x_train.shape[0]):
   # print(x_train[i,:,:,0].shape)
    z = x_train[i,:,:,0]
    z = np.concatenate((z,a))
    z = np.concatenate((z,b), axis= 1)
  #  print(z.shape)
  #  print()
    x_train_adj.append(z)
x_train_adj = np.array(x_train_adj)
print(x_train_adj.shape)


# In[195]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
some_digit = x_train_adj[0]
some_digit_image = some_digit.reshape(30, 168)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
plt.show()


# In[196]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
some_digit = x_train[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
plt.show()


# In[197]:


# Reshape to 28 x 28 pixels
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_val = x_val.astype('float32') / 255
print(x_val.shape)


# In[198]:


a = np.zeros(56).reshape(2,28)
b = np.zeros(4200).reshape(30,140)

x_val_adj = []

for i in range(x_val.shape[0]):
   # print(x_val[i,:,:,0].shape)
    z = x_val[i,:,:,0]
    z = np.concatenate((z,a))
    z = np.concatenate((z,b), axis= 1)
  #  print(z.shape)
  #  print()
    x_val_adj.append(z)
x_val_adj = np.array(x_val_adj)
print(x_val_adj.shape)


# In[199]:


# Reshape to 28 x 28 pixels
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255
print(x_test.shape)


# In[200]:


a = np.zeros(56).reshape(2,28)
b = np.zeros(4200).reshape(30,140)

x_test_adj = []

for i in range(x_test.shape[0]):
   # print(x_test[i,:,:,0].shape)
    z = x_test[i,:,:,0]
    z = np.concatenate((z,a))
    z = np.concatenate((z,b), axis= 1)
  #  print(z.shape)
  #  print()
    x_test_adj.append(z)
x_test_adj = np.array(x_test_adj)
print(x_test_adj.shape)


# In[201]:


## Encode categorical label data
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_val = lb.fit_transform(y_val)
y_test = lb.fit_transform(y_test)


# In[202]:


# Check the shape after the split
print(x_train_adj.shape)
print(y_train.shape)
print(x_val_adj.shape)
print(y_val.shape)
print(x_test_adj.shape)
print(y_test.shape)


# In[203]:


x_train_adj = x_train_adj.reshape(x_train_adj.shape[0], 30, 168, 1)
x_val_adj = x_val_adj.reshape(x_val_adj.shape[0], 30, 168, 1)
x_test_adj = x_test_adj.reshape(x_test_adj.shape[0], 30, 168, 1)


# In[204]:


# Check the shape after the split
print(x_train_adj.shape)
print(y_train.shape)
print(x_val_adj.shape)
print(y_val.shape)
print(x_test_adj.shape)
print(y_test.shape)


# In[205]:


# Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 168, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Add a Flatten layer to the model
model.add(layers.Flatten())
# Add a Dense layer with 64 units and relu activation
model.add(layers.Dense(64, activation='relu'))
# Add the last Dense layer.
model.add(layers.Dense(26, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ["accuracy"])


# In[206]:


history = model.fit(x_train_adj, y_train,
          batch_size=32, epochs=5,
          verbose=1,
          validation_data=(x_val_adj, y_val))


# In[207]:


# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')

plt.tight_layout()


# In[208]:


loss_and_metrics_train = model.evaluate(x_train_adj, y_train, verbose=2)
loss_and_metrics_test = model.evaluate(x_test_adj, y_test, verbose=2)
loss_and_metrics_val = model.evaluate(x_val_adj, y_val, verbose=2)

print("Train Loss", loss_and_metrics_train[0])
print("Train Accuracy", loss_and_metrics_train[1])
print()
print("Test Loss", loss_and_metrics_test[0])
print("Test Accuracy", loss_and_metrics_test[1])
print()
print("Train Loss", loss_and_metrics_val[0])
print("Train Accuracy", loss_and_metrics_val[1])


# In[213]:


pred_class = model.predict_classes(X_test)
print(pred_class[0:10,])


# In[223]:


pred = model.predict(X_test)
print(pred[0,:])


# In[212]:


# Check the shape after the split
print(x_train_adj.shape)
print(y_train.shape)
print(x_val_adj.shape)
print(y_val.shape)
print(x_test_adj.shape)
print(y_test.shape)
print(X_test.shape)


# In[218]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
some_digit = X_test[1]
some_digit_image = some_digit.reshape(30, 168)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
plt.show()


# In[230]:


## Display example digits
plt.figure(figsize=(40,18))
for index, (image) in enumerate(X_test[0:10]):
    plt.subplot(2, 5, index + 1)
    plt.imshow(np.reshape(image, (30,168)), cmap=plt.cm.gray)


# In[ ]:


## idea: combinge 5 pictures and train the algo again


# In[235]:


## Load data
with np.load("training-dataset.npz") as data:
    img = data["x"] ## test dataset
    lbl = data["y"]


# In[236]:


## Split into train and test sets (80 vs 20%)
x_train, x_test, y_train, y_test = train_test_split(img, lbl, test_size=0.2, random_state=0)


# In[237]:


## Split into train and val sets (80 vs 20%)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)


# In[238]:


## Shuffle the data - random splits
n = 50000
x_train, y_train = x_train[0:n], y_train[0:n]
x_test, y_test = x_test[0:n], y_test[0:n]
x_val, y_val = x_val[0:n], y_val[0:n]


# In[239]:


# Check the shape after the split
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)


# In[279]:


# Reshape to 28 x 28 pixels
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255
print(x_train.shape)


# In[282]:


## Shuffle the data - random splits
n = 10000
x_train1 = x_train[0:10000]
x_train2 = x_train[10000:20000]
x_train3 = x_train[20000:30000]
x_train4 = x_train[30000:40000]
x_train5 = x_train[40000:50000]


# In[303]:


x_train_new = []
for i in range(10000):
    w = np.hstack((x_train1[i], x_train2[i]))
    w = np.hstack((w, x_train3[i]))
    w = np.hstack((w, x_train4[i]))
    w = np.hstack((w, x_train5[i]))
    x_train_new.append(w)
x_train_new = numpy.array(x_train_new)
print(x_train_new.shape)


# In[306]:


print(x_train_new[i,:,:,0].shape)


# In[287]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
some_digit = x_train_new[0]
some_digit_image = some_digit.reshape(28, 140)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
plt.show()


# In[307]:


a = np.zeros(280).reshape(2,140)
b = np.zeros(840).reshape(30,28)

x_train_adj = []

for i in range(x_train_new.shape[0]):
   # print(x_train[i,:,:,0].shape)
    z = x_train_new[i,:,:,0]
    z = np.concatenate((z,a))
    z = np.concatenate((z,b), axis= 1)
  #  print(z.shape)
  #  print()
    x_train_adj.append(z)
x_train_adj = np.array(x_train_adj)
print(x_train_adj.shape)


# In[309]:


x_train_adj = x_train_adj.reshape(x_train_adj.shape[0], 30, 168, 1)
print(x_train_adj.shape)


# In[298]:


## Encode categorical label data
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
print(y_train.shape)


# In[299]:


## Shuffle the data - random splits
n = 10000
y_train1 = y_train[0:10000]
y_train2 = y_train[10000:20000]
y_train3 = y_train[20000:30000]
y_train4 = y_train[30000:40000]
y_train5 = y_train[40000:50000]


# In[300]:


y_train_new = []
for i in range(10000):
    w = np.hstack((y_train1[i], y_train2[i]))
    w = np.hstack((w, y_train3[i]))
    w = np.hstack((w, y_train4[i]))
    w = np.hstack((w, y_train5[i]))
    y_train_new.append(w)
y_train_new = numpy.array(y_train_new)
print(y_train_new.shape)


# In[301]:


print(y_train_new[0])
print(y_train_new.shape)


# In[312]:


# Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 168, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Add a Flatten layer to the model
model.add(layers.Flatten())
# Add a Dense layer with 64 units and relu activation
model.add(layers.Dense(64, activation='relu'))
# Add the last Dense layer.
model.add(layers.Dense(130, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ["accuracy"])


# In[313]:


history = model.fit(x_train_adj, y_train_new,
          batch_size=32, epochs=5,
          verbose=1)


# In[330]:


loss_and_metrics_train = model.evaluate(x_train_adj, y_train_new, verbose=2)
loss_and_metrics_test = model.evaluate(X_test, pred, verbose=2)

print("Train Loss", loss_and_metrics_train[0])
print("Train Accuracy", loss_and_metrics_train[1])
print()
print("Test Loss", loss_and_metrics_test[0])
print("Test Accuracy", loss_and_metrics_test[1])


# In[329]:


print(x_train_adj.shape)
print(y_train_new.shape)
print(X_test.shape)


# In[320]:


pred_class = model.predict_classes(X_test)
print(pred_class[0])


# In[318]:


pred = model.predict(X_test)
print(pred.shape)


# In[321]:


print(pred[0])


# In[325]:


### %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
some_digit = pred[3].reshape(5,26)
#some_digit_image = some_digit.reshape(28, 140)
plt.imshow(some_digit, cmap = matplotlib.cm.binary,
interpolation="nearest")
plt.axis("off")
plt.show()

