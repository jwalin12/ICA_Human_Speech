import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.io import wavfile #audioprocessing
from scipy.stats import gennorm
from scipy.signal import welch
import matplotlib.pyplot as plt # plotting
from torchaudio.transforms import MFCC, Resample
from torchaudio import load
from sklearn.decomposition import FastICA
from scipy.optimize import minimize, basinhopping
import os
import glob
import wave


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
#audio_data = np.array(audio_data)
flat_audio_data = [item for sublist in audio_data for item in sublist]
print(np.shape(np.array(audio_data)))
ica = FastICA(n_components=128, max_iter=500)
ica.fit(audio_data)
learning_dict = ica.mixing_

#(flat_audio_data)
# prior = gennorm.fit(flat_audio_data)
#
#
#
# def cost_func_derivative(x,prior):
#     beta = prior[0]
#     loc = prior[1]
#     scale = prior[2]
#
#     return beta*(np.abs(x-loc)**beta)/(scale**beta*(x-loc))
#
# def cost_func(x, prior):
#     beta = prior[0]
#     loc = prior[1]
#     scale = prior[2]
#     return -np.sum(np.abs(x-loc)/scale)**beta
#
#
# def neg_log_likelihood(A):
#     A = A.reshape((300, 300))
#     s = np.matmul(np.linalg.inv(A), audio_data)
#     print(A)
#     return -(cost_func(s, prior) - np.log(np.linalg.det(A)))
# A0 =  np.random.random_sample((300, 300))
# print("minimizing")
# res = basinhopping(neg_log_likelihood, A0,niter=10)
# A = res.A



# tol = 0.0001
# for i in range(1000):
#     s = np.matmul(np.linalg.inv(A), audio_data)
#
#     z = cost_func_derivative(s, prior)
#     del_A = np.matmul(A, np.dot(z, np.transpose(s))) - A
#     A += del_A
#
#     if(np.mean(np.linalg.norm(del_A))) < tol:
#         break
#


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



# for i in range(len(learning_dict)):
#     freqs, psd = welch(learning_dict[i])
#     plt.figure(figsize=(5, 4))
#     plt.plot(freqs, psd)
#     plt.title('PSD: power spectral density {}'.format(i))
#     plt.xlabel('Frequency')
#     plt.ylabel('Power')
#     plt.tight_layout()
#     plt.show()
#     plt.plot(np.arange(0, len(learning_dict[i]))/48,np.array(learning_dict[i]))
#     plt.title('Time {}'.format(i))
#     plt.tight_layout()
#     plt.show()





