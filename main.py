import numpy as np # linear algebra
from scipy.io import wavfile #audioprocessing
from scipy.signal import welch
import matplotlib.pyplot as plt # plotting
from sklearn.decomposition import FastICA
import os


def getListOfWavFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfWavFiles(fullPath)
        else:
            if (".wav" in fullPath):
                allFiles.append(fullPath)

    return allFiles


audio_data =[]
path = '/Users/jwalinjoshi/Downloads/LibriSpeech/dev-clean/'
for file in getListOfWavFiles(path):
    audio_data.append(wavfile.read(file)[1])
    print("loading " + file)
print("done loading")
print(audio_data)
clip_length  = 128 #8ms * 16KHz
audio_data = [x[:clip_length] for x in audio_data]
for x in audio_data:
    if len(x) <clip_length:
        audio_data.remove(x)

ica = FastICA(n_components=128, max_iter=500)
ica.fit(audio_data)
learning_dict = ica.mixing_


for i in range(len(learning_dict)):
    freqs, psd = welch(learning_dict[:,i],160000,nperseg=128)
    plt.figure(figsize=(5, 4))
    plt.plot(freqs, 10*np.log(psd))
    plt.title('PSD: power spectral density {}'.format(i))
    plt.xlabel('Frequency')
    plt.ylabel('Power (dB)')
    plt.show()
    plt.plot(np.arange(0, len(learning_dict[i]))/16,learning_dict[i])
    plt.title("Time {}".format(i))
    plt.xlabel('Time (ms)')
    plt.ylabel('PCM')
    plt.show()




