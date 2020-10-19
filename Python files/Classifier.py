# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 02:48:12 2019

@author: user
"""

import pyaudio # Soundcard audio I/O access library
import wave # Python 3 module for reading / writing simple .wav files


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from glob import glob
import cv2
import numpy as np

import matplotlib as mpl
from sklearn.metrics import confusion_matrix

import librosa
#from  librosa import display

from scipy.io import wavfile

from sklearn.metrics import auc,roc_curve,classification_report

from sklearn.svm import SVC



# Setup channel info
FORMAT = pyaudio.paInt16 # data type formate
CHANNELS = 2 # Adjust to your number of channels
RATE = 22050 # Sample Rate
CHUNK = 1024 # Block Size
RECORD_SECONDS = 15 # Record time
WAVE_OUTPUT_FILENAME = r"C:\Users\user\Desktop\loudy audio\test_folder\file.wav"

# Startup pyaudio instance
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print ("recording...")
frames = []
c=1

# Record for RECORD_SECONDS

for c in range(1,17):
    print(c)
    for i in range(0, int(RATE / CHUNK * 1)):
            
        data = stream.read(CHUNK)
        
        frames.append(data)

print ("finished recording")


# Stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

# Write your new .wav file with built in Python 3 Wave module
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()



fs, data = wavfile.read(WAVE_OUTPUT_FILENAME)


data=np.array(data)
data_fl=data.flatten('F')



index1=len(data_fl)//3

index2=len(data_fl)//2
    
array1=data_fl[0:index1]

array2=data_fl[index1:index2]

array3=data_fl[index2:968704]

plt.plot(array1)
plt.show()

plt.plot(array2)
plt.show()

plt.plot(array3)
plt.show()


path_loudy=r'C:\Users\user\Desktop\loudy audio\loudy\spectrogram\*.png'

path_nonloudy=r'C:\Users\user\Desktop\loudy audio\nonloudy\spectrogram\*.png'

loudy_pngs=glob(path_loudy)

nonloudy_pngs=glob(path_nonloudy)

len(loudy_pngs),len(nonloudy_pngs)



dataset=[]

y_true=[]


for i in range(len(loudy_pngs)):
    img=cv2.imread(loudy_pngs[i])
    
    dataset.append(img)
    
    y_true.append(1)
    
for i in range(len(nonloudy_pngs)):
    
    img=cv2.imread(nonloudy_pngs[i])
    
    dataset.append(img)
    
    y_true.append(0)

len(loudy_pngs),len(nonloudy_pngs)

y_true = np.array(y_true)

dataset = np.array(dataset)


X_train, X_test,y_train, y_test = train_test_split(dataset, y_true, train_size=0.8, random_state=0)


X_train.shape

X_train=X_train.reshape(57,200*200*3)

X_test=X_test.reshape(15,200*200*3)


X_test.shape


classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


cm = confusion_matrix(y_test, y_pred)




classi_report=classification_report(y_test, y_pred)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)

auc_curve=auc(fpr_keras, tpr_keras)


print('auc score is: '+str(auc_curve))
print('\n')


print(cm)
print('\n')

print(classi_report)
print('\n')


sr=22050

test_data=[array1,array2,array3]

#fs, data = wavfile.read(r"C:\Users\user\Desktop\loudy audio\test_folder\file.wav")

for i in range(3):
    
    data=np.array(test_data[i])

    data=data.ravel()

    y=np.array(data).astype(float)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
    
    fig = plt.figure(frameon=False)
    
    fig.set_size_inches((2.78,2.78)) 
    
    extent = mpl.transforms.Bbox(((0, 0), (2.78,2.78))) 
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    
    ax.set_axis_off()
    
    fig.add_axes(ax)

    name=r'C:\Users\user\Desktop\loudy audio\test_folder\out'+str(i+1)+'.png' 

    librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel',fmax=8000,x_axis='time')

    fig.savefig(name, bbox_inches=extent)

    plt.close()
    
test_data=glob(r"C:\Users\user\Desktop\loudy audio\test_folder\*.png")

acutal_test=[]

for i in range(len(test_data)):
    
    img=cv2.imread(test_data[i])
    
    acutal_test.append(img)

acutal_test=np.array(acutal_test)

acutal_test=acutal_test.reshape(3,200*200*3)

acutal_test.shape

actual_test_pred = classifier.predict(acutal_test)


if(actual_test_pred[0]==1):
    print('From 0 to 5 sec the audio is loudy')

else:
    print("From 0 to 5 sec the audio is nonloudy")
    
    
if(actual_test_pred[1]==1):
    print('From 5 to 10 sec the audio is loudy')

else:
    print("From 5 to 10 sec the audio is nonloudy")
    
if(actual_test_pred[2]==1):
    print('From 10 to 15 sec the audio is loudy')

else:
    print("From 10 to 15 sec the audio is nonloudy")
