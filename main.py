import numpy as np # linear algebra
from scipy.io import wavfile #audioprocessing
from scipy.signal import welch
from scipy.stats import gennorm
import matplotlib.pyplot as plt # plotting
from sklearn.decomposition import FastICA
from scipy.misc import derivative
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


def whiten(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten

def cost_func(x):
    return np.log(gennorm.pdf(x, prior[0], prior[1], prior[2]))

def cost_function_derivative(x):
    z = []
    for i in np.transpose(x):
        z.append(np.sum(derivative(cost_func, i, dx =1e-6)))
    return np.transpose(z)

# def iterate_A(A, x):
#     s = np.matmul(np.linalg.inv(A),x)
#     z = cost_function_derivative(s)
#     del_A = np.matmul(A, np.dot(z, np.transpose(s))) - A
#     A_new = A+del_A
#     A_new /= np.sqrt((A_new ** 2).sum())
#
#     return A_new

def g(x):
    return np.tanh(x)
def g_prime(X):
    return 1 - np.tanh(x)**2

def calculate_new_ai(ai, x):
    ai_new = (x * g(np.dot(ai.T, x))).mean(axis=1) - g_prime(np.dot(ai.T, x)).mean() * ai
    ai_new /= np.sqrt((ai_new ** 2).sum())
    return ai_new


def calculate_A(x, max_iter = 128,tol = 0.0001):
    n_components= len(x)
    A = np.zeros((n_components, n_components))
    for i in range(n_components):

        ai = np.random.rand(n_components)

        for j in range(max_iter):

            ai_new = calculate_new_ai(ai, x)

            if i >= 1:
                ai_new -= np.dot(np.dot(ai_new, A[:i].T), A[:i])

            distance = np.abs(np.abs((ai * ai_new).sum()) - 1)

            ai = ai_new

            if distance < tol:
                break

        A[i, :] = ai
        print("done with row {}".format(i))

    return A

def get_S(A, X):
    return np.matmul(A, X)

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

audio_data = audio_data[:500] #computational constraints

A = calculate_A(np.array(audio_data))




for i in range(len(A)):
    freqs, psd = welch(A[:,i],160000,nperseg=128)
    plt.figure(figsize=(5, 4))
    plt.plot(freqs, 10*np.log(psd))
    plt.title('PSD: power spectral density {}'.format(i))
    plt.xlabel('Frequency')
    plt.ylabel('Power (dB)')
    plt.show()
    plt.plot(np.arange(0, len(A[i]))/16,A[i])
    plt.title("Time {}".format(i))
    plt.xlabel('Time (ms)')
    plt.ylabel('PCM')
    plt.show()




