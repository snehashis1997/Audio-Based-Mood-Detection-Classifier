# Audio Based Mood Detection Classier using Python

This is my work in SkyBits Technologies Pvt Ltd (http://sky-bits.com/) from 1st August, 2019 to 30th September, 2019. During this time, I was to design a novel machine learning based model where using my Laptop’s microphone it recorded 15 sec audio and divided 15sec audio into three parts, took decision for each part whether that part was loudly or non-loudly.

https://www.google.com/url?sa=i&url=https%3A%2F%2Fstackoverflow.com%2Fquestions%2F51125356%2Fproper-way-to-build-menus-with-python-telegram-bot&psig=AOvVaw3H7NAn5TJT9iGvOKCdmAV6&ust=1599392747505000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCIDfsrf40esCFQAAAAAdAAAAABAD


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

Here I worked with some Bio medical signals.  I used some preprocessing stpes like abrupt Peak finding, removing noise by Gausiinan filtering (see notebook: name of the file)
Then using histrogram plots try to create some class.

At first I solved the given problem as a regression problem, later I solved the problem as a classification problem.

So, in regression problem, I tried to use diffrent regression algorithims like SVR,Random forest(Bagging), XGBOOST(Boosting).

Also the given dataset was very much imbalanced, so while solving classification problems I used Imblearn library's SMOTE algorithim to reduce imblance.
