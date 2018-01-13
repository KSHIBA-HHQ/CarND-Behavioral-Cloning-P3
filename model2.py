# Neural Networks in Keras (Lesson5)
# dowloaded    small_test_traffic.p, small_train_traffic.

#!python --version
import keras; print('Keras ' + keras.__version__)
import tensorflow as tf; print('TensorFlow ' + tf.__version__)

#################################################################
#################################################################
#################################################################
def figerimage(img):
    import cv2
    import matplotlib.pyplot as plt

    plt.figure(figsize = ( 5 , 5 ))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.title(['raw',img.dtype])

    plt.subplot(2,1,2)
    ximg=img
    h=ximg.shape[0]
    w=ximg.shape[1]
    ximg=cv2.rectangle(ximg, (0, 70), (w, h-25), (255, 0, 0), 5)
    plt.imshow(ximg)
    plt.title(['cropping',ximg.dtype])
    plt.show()

def comphistgram(dat,title):
    import numpy as np
    import matplotlib.pyplot as plt

# ヒストグラムを出力
    plt.figure(figsize = ( 5 , 5 ))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.subplot(2,1,1)
    plt.hist(dat[0])
    plt.title([title[0],len(dat[0])])

    plt.subplot(2,1,2)
    plt.hist(dat[1])
    plt.title([title[1],len(dat[1])])
    plt.show()


import csv
import cv2
from skimage import io


#default
ubu_file00='./data/data/'
ubu_file01='./bag/add00/'
win_file02='./bag/bag00/'
win_file03='./bag/bag01/'
win_file04='./bag/rev00/'
win_file05='./bag/rev01/'
win_file06='./bag/xdd00/'

xfile=[ubu_file00,ubu_file01,win_file02,win_file03,win_file04,win_file05,win_file06]

import numpy as np

xskip=np.ones(len(xfile))*40
xskip[0]=int(xskip[0]*2)
xskip[4]=int(xskip[0]*0.5)
xskip[5]=int(xskip[0]*0.5)
xskip[6]=int(xskip[0]*0.5)

cImages = np.array( [] )
lImages = np.array( [] )
rImages = np.array( [] )

cSteering=np.array( [] )
lSteering=np.array( [] )
rSteering=np.array( [] )
speed=np.array( [] )

step=0
element=0

all_steer=[]
for j in range(len(xfile)):

        lines=[]
        print(xfile[j])
        with open(xfile[j]+'driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    lines.append(line)


        for line in lines:
            if(line[3]=='steering'):
                 continue;
            if(float(line[6])<0.1):
                 continue;

            #################
            #for debug
            step=step+1;
            if(step%xskip[j]):
                continue
            check=float(line[3])
            all_steer.append(check)

            if(np.absolute(check)<0.001):
                   continue;
            #################
            element=element+1

            if(speed.shape[0]==0):
                speed=np.array([float(line[6])])
            else:
                speed=np.append(speed,float(line[6]))

            for i in range (3):
                source_path=line[i]
                #  csvから画像ファイル名を取得
                filename=source_path.split('/')[-1]
                filename=filename.rsplit('\\')[-1]
                #print(filename)

                #　パス名を変更
                #  print(filename)
                current_path=xfile[j]+'IMG/'+filename
                #  print(current_path)

                #　OpenCVのimreadを使うとBGRで読み込み
                #  images.append(cv2.imread(current_path))
                #　skimageのimreadを使うとRGBで読み込み

                if(i==0):
                    if(cImages.shape[0]==0):
                        cImages=io.imread(current_path)
                        cSteering=np.array(float(line[3]))
                    else:
                        cImages=np.append(cImages,io.imread(current_path))
                        cSteering=np.append(cSteering,float(line[3]))
                if(i==1):
                    S=float(line[3])
                    if(S<0):
                        S=S*0.8
                    else:
                        S=S+0.4
                    if(lImages.shape[0]==0):
                        lImages=io.imread(current_path)
                        lSteering=np.array(S)
                    else:
                        lImages=np.append(lImages,io.imread(current_path))
                        lSteering=np.append(lSteering,S)
                if(i==2):
                    S=float(line[3])
                    if(S>0):
                        S=S*0.8
                    else:
                        S=S-0.4
                    if(rImages.shape[0]==0):
                        rImages=io.imread(current_path)
                        rSteering=np.array(S)
                    else:
                        rImages=np.append(rImages,io.imread(current_path))
                        rSteering=np.append(rSteering,S)


cImages=np.reshape(cImages,[element,160,320,3])
rImages=np.reshape(rImages,[element,160,320,3])
lImages=np.reshape(lImages,[element,160,320,3])

xImages=cImages
xMesurements=cSteering

#################################################################
#################################################################
#################################################################

lesson=14
#lesson 11 (training add mirror image )
#lesson 12 (using multi camera-view and tuneup steering-info )
#lesson 13 Kerasで画像を切り取る

if(lesson>=12):
    #左カメラ画像に対してはステアをプラス増分、右カメラについてはステア角度をマイナス増分
    xImages=np.append(xImages,lImages)
    xMesurements=np.append(xMesurements,lSteering)
    xImages=np.append(xImages,rImages)
    xMesurements=np.append(xMesurements,rSteering)

    xImages=np.reshape(xImages,[3*element,160,320,3])

if(lesson>=11):
    augument_images, augument_mesurements=[],[]
    for image,mesurements in zip(xImages,xMesurements):
        augument_images.append(image)
        augument_mesurements.append(mesurements)
        augument_images.append(cv2.flip(image,1))
        augument_mesurements.append(mesurements*-1.0)
    X_train=np.array(augument_images)
    y_train=np.array(augument_mesurements)
else:
    X_train=np.array(xImages)
    y_train=np.array(xMesurements)


print(cImages.shape,X_train.shape,y_train.shape)
figerimage(X_train[0])
comphistgram([all_steer,y_train.tolist()],['steer raw','steer custom'])
#################################################################
#################################################################
###########################################################
########

#################
from keras.models import Sequential
#lesson=7
#from keras.layers import Flatten, Dense
#lesson>=9 (with lambda)
from keras.layers import Flatten, Dense,Lambda,Dropout,Cropping2D
#lesson>=10 (with convolution)
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


DROPOUTS=[0.5,0.7,0.9,1]


for DOUT in DROPOUTS:
    model = Sequential()
    #################

    if(lesson<14):
            if(lesson==9):
                model.add(Lambda(lambda x:x /255.0 -0.5 ,input_shape=(160,320,3)))
                model.add(Flatten())
            if(lesson==7):
                model.add(Flatten(input_shape=(160,320,3)))

            if(lesson>=10):
                #Lambda正規化を行わないの場合、第1階層Convolution2Dの引数に対してinput_shapeの指定が必要
                #model.add(Convolution2D(6,5,5 ,input_shape=(160,320,3),activation="relu"))
                if(lesson>=13):
                    model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160,320,3)))
                    model.add(Lambda(lambda x:x /255.0 -0.5 ))
                else:
                    model.add(Lambda(lambda x:x /255.0 -0.5 ,input_shape=(160,320,3)))

                model.add(MaxPooling2D())
                model.add(Dropout(0.9))
                model.add(Convolution2D(6,5,5,activation="rlesson=14elu"))
                model.add(MaxPooling2D())
                model.add(Dropout(DOUT))
                model.add(Flatten())
                model.add(Dense(120))
                model.add(Dense(84))
    if(lesson==14):
                model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160,320,3)))
                model.add(Lambda(lambda x:x /255.0 -0.5))
                model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
                model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
                model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
                model.add(Convolution2D(64,3,3,activation="relu"))
                model.add(Convolution2D(64,3,3,activation="relu"))
                model.add(Dropout(DOUT))
                model.add(Flatten())
                model.add(Dense(100))
                model.add(Dense(50))
                model.add(Dense(10))

    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam')


    ########
    from keras.models import Model
    import matplotlib.pyplot as plt

    """
    if(lesson>=14):
        history_object=model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)
    if(lesson==9):
        history_object=model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=2)
    if(lesson==7):
        history_object=model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=7)
    """
    history_object=model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=20)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch

    plt.ylim([0,0.5])
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title(['model mean squared error loss','Dropout',DOUT])
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    h5name = '_model%s.h5' % int(DOUT*100)
    model.save(h5name)
    print('complete training')
