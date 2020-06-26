
# coding: utf-8

## Load all required packages
# get_ipython().run_line_magic('pylab', 'inline --no-import-all')
import numpy as np
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.metrics import CategoricalAccuracy, categorical_accuracy
# import seaborn as sns
plt.style.use('ggplot')

# In[2]:


from keras import Sequential
from keras import layers
from keras import models

## Task 1

# Create a sequential model
model = Sequential()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[3]:


model.summary()


# In[4]:


# Add a Flatten layer to the model
model.add(layers.Flatten())
# Add a Dense layer with 64 units and relu activation
model.add(layers.Dense(64, activation='relu'))
# Add the last Dense layer.
model.add(layers.Dense(26, activation='softmax'))


# In[5]:


lr = 0.001
opt = Adam(lr)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ["accuracy", "categorical_accuracy"])


# In[7]:


## Load data
with np.load("training-dataset.npz") as data:
    img = data["x"]
    lbl = data["y"]


# In[8]:


## Check shapes and types of the dataset
print(img.shape)
print(lbl.shape)
print(type(img))
print(type(lbl))


# In[9]:


## Split into train and val sets (80 vs 20%)
x_train, x_test, y_train, y_test = train_test_split(img, lbl, test_size=0.2, random_state=0)


# In[10]:


# Check the shape after the split
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[11]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
train_images = x_train.astype('float32') / 255
print(x_train.shape)


# In[12]:


x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
test_images = x_test.astype('float32') / 255
print(x_test.shape)


# In[13]:


## Encode categorical label data

lb = LabelBinarizer()
train_labels = lb.fit_transform(y_train)
test_labels = lb.fit_transform(y_test)


# In[14]:


# Check the shape after the split
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


# In[15]:


print(train_labels[0])


# In[16]:


history = model.fit(train_images, train_labels,
          batch_size=128, epochs=5,
          verbose=1,
          validation_data=(test_images, test_labels))


# In[80]:


loss_and_metrics_train = model.evaluate(train_images, train_labels, verbose=2)
loss_and_metrics = model.evaluate(test_images, test_labels, verbose=2)

print("Train Loss", loss_and_metrics_train[0])
print("Train Accuracy", loss_and_metrics_train[1])
print()
print("Validation Loss", loss_and_metrics[0])
print("Validation Accuracy", loss_and_metrics[1])


# In[82]:


loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


# In[83]:


epochs = range(1, 6)

plt.plot(epochs, loss, 'ko', label = 'Training Loss')
plt.plot(epochs, val_loss, 'k', label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Training and Validation Loss.png')


# In[84]:


plt.plot(epochs, acc, 'yo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'y', label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('Training and Validation Accuracy.png')


# In[ ]:


## Task 2

## Load data
testdata = np.load('test-dataset.npy')

print(testdata.shape)
print(type(testdata))

print(testdata[0,0])

## Source: Hands-On Machine Learning Book
some_pic = testdata[1]
#print(one of the images)
some_pic_image = some_pic.reshape(30, 168)
plt.imshow(some_pic_image, cmap = plt.cm.binary,
interpolation="nearest")
plt.axis("off")
plt.show()

import matplotlib.pyplot as plt
import cv2
import pickle
import os.path
import numpy as np
import pandas as pd

# source https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/
img=cv2.imread('/content/download.png')
# source https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#thresholding to remove background
thr = cv2.threshold(img, 0, 255,  cv2.THRESH_OTSU)[1]

#control kernel
kernel = np.ones((3,3), np.uint8)
img=cv2.dilate(thr,kernel)
kernel = np.ones((5, 5), np.uint8)
img = cv2.erode(img, kernel)

#invert the image if you not invert them will detect outer contour
img = cv2.bitwise_not(img)

contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

imgplot = plt.imshow(img)        
plt.show()

hierarchy

newContours=[]
#loop over zip of contours and hierarchy so to get matched cuples of contours and hierarchs
for contour,hierarch in zip(contours,hierarchy[0]):#it is dimensional
    if hierarch[3] == -1:#get the parent only which is the last of the last of each list
        newContours.append(contour)
    else:
        continue

lenghtContours=len(newContours)

newContours=[]#for the specified shapes only , no outer image nor inner of items
xs=[]
ys=[]
ws=[]
hs=[]
WdevH=[]
a=[]
letter_image_regions=[]
for contour in newContours:
    area = cv2.contourArea(contour)
    a.append(area)
    if area > 50:
        a.append(area)
        # Get the rectangle that contains the contour
        x, y, w, h = cv2.boundingRect(contour) 
    #     xs.append(x) 
    #     ys.append(y)
    #     ws.append(w)
    #     hs.append(h)
    #     WdevH.append(wdevh)
        if  lenghtContours==5:
            roi = img[y:y+h, x:x+w]
#                 cv2.imwrite("{}.png".format(str(w/h)), roi)
            letter_image_regions.append((x,y,w,h))
        elif  lenghtContours==4:
            if w/h >=1.2:
                    half_width = int(w / 2)
                    letter_image_regions.append((x, y, half_width, h))
                    letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                    # This is a normal letter by itself
                    letter_image_regions.append((x, y, w, h))

        elif lenghtContours==1:
            if w/h >=2 and w/h <=4 :
                    one_item=int(w/4)
                    letter_image_regions.append((x,y,one_item,h))
                    letter_image_regions.append((x+one_item,y,one_item,h))
                    letter_image_regions.append((x+(2*one_item),y,one_item,h))
                    letter_image_regions.append((x+(3*one_item),y,one_item,h))
            elif w/h > 4 :
                    one_item_from5=int(w/5)
                    letter_image_regions.append((x,y,one_item_from5,h))
                    letter_image_regions.append((x+one_item_from5,y,one_item_from5,h))
                    letter_image_regions.append((x+(2*one_item_from5),y,one_item_from5,h))
                    letter_image_regions.append((x+(3*one_item_from5),y,one_item_from5,h))
                    letter_image_regions.append((x+(4*one_item_from5),y,one_item_from5,h))
            else:
                    letter_image_regions.append((x,y,w,h))
        elif lenghtContours==3:
            if w/h >= 1.2 and w/h <= 2.1  :
                    half_width = int(w / 2)
                    print(half_width)
                    letter_image_regions.append((x, y, half_width, h))
                    letter_image_regions.append((x + half_width, y, half_width, h))
            elif w/h > 2.1:
                    one_item_from3 = int(w/3)
                    letter_image_regions.append((x,y,one_item_from3,h))
                    letter_image_regions.append((x+one_item_from3,y,one_item_from3,h))
                    letter_image_regions.append((x+(2*one_item_from3),y,one_item_from3,h))

            else:
                    letter_image_regions.append((x,y,w,h))
        elif lenghtContours==2:

            if w/h > 1.2 and w/h <= 2:
                    one_item_from3=int(w/3)
                    letter_image_regions.append((x,y,one_item_from3,h))
                    letter_image_regions.append((x+one_item_from3,y,one_item_from3,h))
                    letter_image_regions.append((x+(2*one_item_from3),y,one_item_from3,h))   
            elif w/h >2:
                    one_item=int(w/4)
                    letter_image_regions.append((x,y,one_item,h))
                    letter_image_regions.append((x+one_item,y,one_item,h))
                    letter_image_regions.append((x+(2*one_item),y,one_item,h))
                    letter_image_regions.append((x+(3*one_item),y,one_item,h))
            else:
                    letter_image_regions.append((x,y,w,h))   
    else:
        continue

letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

img= cv2.subtract(255, img)#reverse color again 
# Create an output image and a list to hold our predicted letters
output = cv2.merge([img] * 3)
predictions = []    
# cv2.imwrite('x.png', output)
# loop over the letters
for letter_bounding_box in letter_image_regions:
    # Grab the coordinates of the letter in the image
    x, y, w, h = letter_bounding_box
#     print(x,y,w,h)
      # Extract the letter from the original image with a 2-pixel margin around the edge
    letter_image = img[y - 2:y + h + 2, x - 2:x + w + 2]
    letter_image = cv2.copyMakeBorder(letter_image, 5, 5,5 , 5, cv2.BORDER_CONSTANT, value=255)

    # Re-size the letter image to 28x28 pixels to match training data

    letter_image = cv2.resize(letter_image,(28,28))  

    letter_image= cv2.subtract(255, letter_image)#reverse color again 
    letter_image = letter_image /255.0

    plt.imshow(letter_image)

    # Turn the single image into a 4d list of images to make Keras happy
    letter_image = np.expand_dims(letter_image, axis=2)
    letter_image = np.expand_dims(letter_image, axis=0)

    # Ask the cnn to make a prediction

# %%
