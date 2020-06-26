# Load all required packages
# get_ipython().run_line_magic('pylab', 'inline --no-import-all')
import matplotlib
import pandas as pd
import os.path
import pickle
import cv2
from keras import models
from keras import layers
from keras import Sequential
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

# Task 1

# Create a sequential model
model = Sequential()
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

# Add a Flatten layer to the model
model.add(layers.Flatten())
# Add a Dense layer with 64 units and relu activation
model.add(layers.Dense(64, activation='relu'))
# Add the last Dense layer.
model.add(layers.Dense(26, activation='softmax'))

lr = 0.001
opt = Adam(lr)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=["accuracy", "categorical_accuracy"])

# Load data
with np.load("training-dataset.npz") as data:
    img = data["x"]
    lbl = data["y"]

# Check shapes and types of the dataset
print(img.shape)
print(lbl.shape)
print(type(img))
print(type(lbl))

# Split into train and val sets (80 vs 20%)
x_train, x_test, y_train, y_test = train_test_split(
    img, lbl, test_size=0.2, random_state=0)

# Check the shape after the split
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
train_images = x_train.astype('float32') / 255
print(x_train.shape)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
test_images = x_test.astype('float32') / 255
print(x_test.shape)

# Encode categorical label data
lb = LabelBinarizer()
train_labels = lb.fit_transform(y_train)
test_labels = lb.fit_transform(y_test)

# Check the shape after the split
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

print(train_labels[0])

history = model.fit(train_images, train_labels,
                    batch_size=128, epochs=5,
                    verbose=1,
                    validation_data=(test_images, test_labels))

loss_and_metrics_train = model.evaluate(train_images, train_labels, verbose=2)
loss_and_metrics = model.evaluate(test_images, test_labels, verbose=2)

print("Train Loss", loss_and_metrics_train[0])
print("Train Accuracy", loss_and_metrics_train[1])
print()
print("Validation Loss", loss_and_metrics[0])
print("Validation Accuracy", loss_and_metrics[1])

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, 6)

plt.plot(epochs, loss, 'ko', label='Training Loss')
plt.plot(epochs, val_loss, 'k', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('Training and Validation Loss.png')

plt.plot(epochs, acc, 'yo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'y', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('Training and Validation Accuracy.png')

# Task 2

# Load required packages

# Load data
testdata = np.load('test-dataset.npy')

print(testdata.shape)
print(type(testdata))

print(testdata[0, 0])

for index, image in zip(range(1,len(testdata)+1),testdata):#it is dimensional
        print(index,image)
        some_image = image.reshape(30, 168)
        img=cv2.imwrite('images{}.png'.format(index),some_image)

import os
arr = os.listdir("images")
image_paths = [ os.path.join("images",x) for x in arr]
# print(image_paths)

# # Source: Hands-On Machine Learning Book
# some_pic = testdata[1]
# # print(one of the images)
# print(some_pic.shape)
# some_image = some_pic.reshape(30, 168)
# plt.imshow(some_image, cmap=matplotlib.cm.binary,
# interpolation="nearest")
# plt.axis("off")
# plt.show()

# Letter labels
letters = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"

index_to_leter = dict(zip(list(range(0, len(letters)+2)), letters.split(" ")))
print(len(index_to_leter))

# Pre-process images to remove noise

images = []
predic = []
for i in image_paths:
#     some_image = i.reshape(30, 168)
#     img = cv2.imwrite('0705071916.png', some_image)
#     plt.imshow(some_image, cmap=plt.cm.binary,
#                interpolation="nearest")
#     plt.axis("off")
#     plt.show()

#     img = cv2.imread('0705071916.png')
    # source https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresholding to remove background
    thr = cv2.threshold(img, 0, 255,  cv2.THRESH_OTSU)[1]

# TBD
# source https://www.meccanismocomplesso.org/en/opencv-python-the-otsus-binarization-for-thresholding/
# img = cv2.imread('noisy_leaf.jpg',0)
# ret,
# imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plt.subplot(3,1,1), plt.imshow(img,cmap = 'gray')
# plt.title('Original Noisy Image'),
# plt.xticks([]),
# plt.yticks([])

# plt.subplot(3,1,2),
# plt.hist(img.ravel(), 256)
# plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
# plt.title('Histogram'),
# plt.xticks([]),
# plt.yticks([])

# plt.subplot(3,1,3),
# plt.imshow(thr,cmap = 'gray')
# plt.title('Otsu thresholding'),
# plt.xticks([]),
# plt.yticks([])
# plt.show()
# ###########

# Morphological Operations
# structuring element
# Note that kernal sizes must be positive and odd and the kernel must be square.
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel
    img = cv2.dilate(thr, kernel)
    kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel
    img = cv2.erode(img, kernel)

# Invert colours, so gridlines have non-zero pixel values, help to detect outer contour
    img = cv2.bitwise_not(img)

# Boundary estimation

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contour is a Numpy array of (x,y) coordinates of boundary points of the object.
# CHAIN_APPROX_SIMPLE removes redundant points and compresses the contour, thereby saving memory.

# Each array is in format (Next, Prev, First child, Parent)
    newContours = []  # for the specified shapes only , no outer image nor inner of items
    xs = []
    ys = []
    ws = []
    hs = []
    WdevH = []
    a = []
    letter_image_regions = []

    for contour, hierarch in zip(contours, hierarchy[0]):  # it is dimensional
            # Filter the ones without parent
        if hierarch[3] == -1:  # get the parent only which is the last of the last of each list
                newContours.append(contour)
        else:
                continue

    lenghtContours = len(newContours)
    for contour in newContours:
        area = cv2.contourArea(contour)
        # find the contours with large surface area
        if area > 50:
                a.append(area)
                # Get the rectangle that contains the contour
                x, y, w, h = cv2.boundingRect(contour)

                xs.append(x)
                ys.append(y)
                ws.append(w)
                hs.append(h)

                if lenghtContours == 5:
                        roi = img[y:y+h, x:x+w]
                        #cv2.imwrite("{}.png".format(str(w/h)), roi)
                        letter_image_regions.append((x, y, w, h))

                elif lenghtContours == 4:
                        if w/h >= 1.2:
                                half_width = int(w / 2)
                                letter_image_regions.append((x, y, half_width, h))
                                letter_image_regions.append((x + half_width, y, half_width, h))
                        else:
                        # This is a normal letter by itself
                                letter_image_regions.append((x, y, w, h))

                elif lenghtContours == 1:
                        if w/h >= 2 and w/h <= 4:
                                one_item = int(w/4)
                                letter_image_regions.append((x, y, one_item, h))
                                letter_image_regions.append((x+one_item, y, one_item, h))
                                letter_image_regions.append((x+(2*one_item), y, one_item, h))
                                letter_image_regions.append((x+(3*one_item), y, one_item, h))
                        elif w/h > 4:
                                one_item_from5 = int(w/5)
                                letter_image_regions.append((x, y, one_item_from5, h))
                                letter_image_regions.append(
                                        (x+one_item_from5, y, one_item_from5, h))
                                letter_image_regions.append(
                                        (x+(2*one_item_from5), y, one_item_from5, h))
                                letter_image_regions.append(
                                        (x+(3*one_item_from5), y, one_item_from5, h))
                                letter_image_regions.append(
                                        (x+(4*one_item_from5), y, one_item_from5, h))
                        else:
                                letter_image_regions.append((x, y, w, h))

                elif lenghtContours == 3:
                        if w/h >= 1.2 and w/h <= 2.1:
                                half_width = int(w / 2)
                                print(half_width)
                                letter_image_regions.append((x, y, half_width, h))
                                letter_image_regions.append((x + half_width, y, half_width, h))
                        elif w/h > 2.1:
                                one_item_from3 = int(w/3)
                                letter_image_regions.append((x, y, one_item_from3, h))
                                letter_image_regions.append(
                                        (x+one_item_from3, y, one_item_from3, h))
                                letter_image_regions.append(
                                        (x+(2*one_item_from3), y, one_item_from3, h))
                        else:
                                letter_image_regions.append((x, y, w, h))

                elif lenghtContours == 2:
                        if w/h > 1.2 and w/h <= 2:
                                one_item_from3 = int(w/3)
                                letter_image_regions.append((x, y, one_item_from3, h))
                                letter_image_regions.append(
                                        (x+one_item_from3, y, one_item_from3, h))
                                letter_image_regions.append(
                                        (x+(2*one_item_from3), y, one_item_from3, h))
                        elif w/h > 2:
                                one_item = int(w/4)
                                letter_image_regions.append((x, y, one_item, h))
                                letter_image_regions.append((x+one_item, y, one_item, h))
                                letter_image_regions.append((x+(2*one_item), y, one_item, h))
                                letter_image_regions.append((x+(3*one_item), y, one_item, h))
                        else:
                                letter_image_regions.append((x, y, w, h))
                else:
                        letter_image_regions.append((x, y, w, h))
        else:
                continue

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    # source https://www.programcreek.com/python/example/86843/cv2.contourArea

    img = cv2.subtract(255, img)  # reverse color again

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([img] * 3)
    predictions = []

    # loop over the letters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = img[y - 2:y + h + 2, x - 2:x + w + 2]
        letter_image = cv2.copyMakeBorder(
            letter_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
        # source https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=copymakeborder

        # Re-size the letter image to 28x28 pixels to match training data
        letter_image = cv2.resize(letter_image, (28, 28))

        # reverse color again
        letter_image = cv2.subtract(255, letter_image)
        letter_image = letter_image / 255.0

        plt.imshow(letter_image)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the cnn to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = index_to_leter[prediction.argmax()]
        if len(letter) == 2:
            letter = letter[0]
        predictions.append(letter)
    predic.append(predictions)  # append prediction in one array
    images.append(some_image)

    # Print the captcha's text
    print(predictions)

#Build a dataframe
df = pd.DataFrame(data={'images':images,'predictions': predic})
#convertint list of prediction classes into columns
df[['class1','class2','class3','class4','class5']] = pd.DataFrame(df.predictions.tolist(), index= df.index)

letter=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
encodeNum=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26']

df['predictions']=df['predictions'].astype(str).str.replace('A','01')
df['predictions']=df['predictions'].astype(str).str.replace('B','02')
df['predictions']=df['predictions'].astype(str).str.replace('C','03')
df['predictions']=df['predictions'].astype(str).str.replace('D','04')
df['predictions']=df['predictions'].astype(str).str.replace('E','05')
df['predictions']=df['predictions'].astype(str).str.replace('F','06')
df['predictions']=df['predictions'].astype(str).str.replace('G','07')
df['predictions']=df['predictions'].astype(str).str.replace('H','08')
df['predictions']=df['predictions'].astype(str).str.replace('I','09')
df['predictions']=df['predictions'].astype(str).str.replace('J','10')
df['predictions']=df['predictions'].astype(str).str.replace('K','11')
df['predictions']=df['predictions'].astype(str).str.replace('L','12')
df['predictions']=df['predictions'].astype(str).str.replace('M','13')
df['predictions']=df['predictions'].astype(str).str.replace('N','14')
df['predictions']=df['predictions'].astype(str).str.replace('O','15')
df['predictions']=df['predictions'].astype(str).str.replace('P','16')
df['predictions']=df['predictions'].astype(str).str.replace('Q','17')
df['predictions']=df['predictions'].astype(str).str.replace('R','18')
df['predictions']=df['predictions'].astype(str).str.replace('S','19')
df['predictions']=df['predictions'].astype(str).str.replace('T','20')
df['predictions']=df['predictions'].astype(str).str.replace('U','21')
df['predictions']=df['predictions'].astype(str).str.replace('V','22')
df['predictions']=df['predictions'].astype(str).str.replace('W','23')
df['predictions']=df['predictions'].astype(str).str.replace('X','24')
df['predictions']=df['predictions'].astype(str).str.replace('Y','25')
df['predictions']=df['predictions'].astype(str).str.replace('Z','26')

df.to_csv('Predictions.csv')