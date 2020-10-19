# Audio Based Mood Detection Classier using Python

This is my work in SkyBits Technologies Pvt Ltd (http://sky-bits.com/) from 1st August, 2019 to 30th September, 2019. During this time, I was to design a novel machine learning based model where using my Laptop’s microphone it recorded 15 sec audio and divided 15sec audio into three parts, took decision for each part whether that part was loudly or non-loudly.

# Dataset description

-- The given problem dataset is not present in opensource, so I have to create this dataset. At first,  downloded some movie clips,then extract the audios from those clips.

-- Then, make short audio clips (15 secs) length, compute the MFCCs and plot the spectrogram in dB scale.

# Libraries:

1. Pandas -- for tabular dataset handeling

2. Numpy -- for array related works

3. Scipy -- for peak detection and gaussian smoothing

4. Scikit learn -- for building SVM based classification model

5. librosa -- for audio signal processing anf MFCC spectrogram plotting

6. Pyaudio -- for record audio from laptop's microphone


# My work

Using Librosa library 1st extract the MFCC features from .wav files and plot it with matplotlib library then save the pictures with .png extension.
The wav files are collected from Youtube video file. Divided the YouTube file into 5-7sec files then extract the MFCC features using Librosa.
For testing used Laptop’s Microphone to record live audio and then plot it using Matplotlib then using Librosa library extract MFCC features and test it.

For train the accuracy almost 95%.

## Time domain visualization of loudy and non loudy audio signals

![image](https://user-images.githubusercontent.com/33135767/96457317-3c5d8900-123d-11eb-932b-9bc1689679aa.png) ![image](https://user-images.githubusercontent.com/33135767/92584576-71c39e00-f2b1-11ea-980a-5490b8adf52c.png)

## Spectrograms of both these signals
![spectrogram_loudy_8](https://user-images.githubusercontent.com/33135767/92584105-daf6e180-f2b0-11ea-9e25-e21dd1e7d5a9.png) ![spectrogram_nonloudy0](https://user-images.githubusercontent.com/33135767/92584168-ee09b180-f2b0-11ea-9bf9-355e1c2e8036.png)



