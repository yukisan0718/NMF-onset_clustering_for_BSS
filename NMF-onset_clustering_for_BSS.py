#!/usr/bin/env python
# coding: utf-8

import sys
import time
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
from scipy import fftpack as fp
from scipy import linalg
from museval.metrics import bss_eval_images, bss_eval_sources

### Function for audio pre-processing ###
def pre_processing(data, Fs, down_sam):
    
    #Transform stereo into monoral
    if data.ndim == 2:
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    
    #Down sampling and normalization of the wave
    if down_sam is not None:
        wavdata = sg.resample_poly(wavdata, down_sam, Fs)
        Fs = down_sam
    
    return wavdata, Fs

### Function for getting STFT ###
def get_STFT(wav, Fs, frame_length, frame_shift):
    
    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Execute STFT
    freqs, times, dft = sg.stft(wav, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    arg = np.angle(dft) #Preserve the phase
    Adft = np.abs(dft) #Preserve the absolute amplitude
    Y = Adft
    
    #Display the size of input
    print("Spectrogram size (freq, time) = " + str(Y.shape))
    
    return Y, arg, Fs, freqs, times

### Function for getting inverse STFT ###
def get_invSTFT(Y, arg, Fs, frame_length, frame_shift):
    
    #Restrive the phase from original wave
    Y = Y * np.exp(1j*arg)
    
    #Get the inverse STFT
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    _, rec_wav = sg.istft(Y, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    
    return rec_wav, Fs

### Function for removing components closing to zero ###
def get_nonzero(tensor):
    
    tensor = np.where(np.abs(tensor) < 1e-10, 1e-10+tensor, tensor)
    return tensor

### Function for computing numerator of temporal continuity term ###
def continuity_numer(U):
    
    #Get the value at the start and end point in U
    start = U[:, 0][:, np.newaxis]
    end = U[:, -1][:, np.newaxis]
    
    #Get summation of squared U
    U2 = np.sum(U**2, axis=1, keepdims=True)
    
    #Compute the first term
    term1 = (np.append(U, end, axis=1) - np.append(start, U, axis=1))**2
    term1 = U * np.sum(term1, axis=1, keepdims=True) / get_nonzero(U2**2)
    
    #Compute the second term
    term2 = np.append(np.append(U, end, axis=1), end, axis=1)
    term2 = term2 + np.append(start, np.append(start, U, axis=1), axis=1)
    term2 = term2[:, 1:-1] / get_nonzero(U2)
    
    output = term1 + term2
    
    #Return numerator of temporal continuity term
    return output

### Function for computing denominator of temporal continuity term ###
def continuity_denom(U):
    
    output = U / get_nonzero(np.sum(U**2, axis=1, keepdims=True))
    return output

### Function for computing temporal continuity cost ###
def continuity_cost(U):
    
    #Get the value at the start and end point in U
    start = U[:, 0][:, np.newaxis]
    end = U[:, -1][:, np.newaxis]
    
    #Subtract adjacent values in U
    output = np.append(U, end, axis=1) - np.append(start, U, axis=1)
    
    #Get the sum of squares
    output = np.sum((output[:, 1:])**2, axis=1) / get_nonzero(np.sum(U**2, axis=1))
    output = np.sum(output)
    
    #Retern temporal continuity cost
    return output

### Function for getting basements and weights matrix by NMF ###
def get_NMF(Y, num_iter, num_base, loss_func, alpha, norm_H):
    
    #Initialize basements and weights based on the Y size(k, n)
    K, N = Y.shape[0], Y.shape[1]
    if num_base >= K or num_base >= N:
        print("The number of basements should be lower than input size.")
        sys.exit()
    
    #Remove Y entries closing to zero
    Y = get_nonzero(Y)
    
    #Initialize as random number
    H = np.random.rand(K, num_base) #basements (distionaries)
    U = np.random.rand(num_base, N) #weights (coupling coefficients)
    
    #Initialize loss
    loss = np.zeros(num_iter)
    
    #For a progress bar
    unit = int(np.floor(num_iter/10))
    bar = "#" + " " * int(np.floor(num_iter/unit))
    start = time.time()
    
    #In the case of squared Euclidean distance
    if loss_func == "EU":
        
        #Repeat num_iter times
        for i in range(num_iter):
            
            #Display a progress bar
            print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            if i % unit == 0:
                bar = "#" * int(np.ceil(i/unit)) + " " * int(np.floor((num_iter-i)/unit))
                print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            
            #Update the basements
            X = H @ U
            H = H * (Y @ U.T) / get_nonzero(X @ U.T)
            #Normalize the basements
            if norm_H == True:
                H = H / H.sum(axis=0, keepdims=True)
            
            #Update the weights
            X = H @ U
            denom_U = H.T @ X + 4*alpha*N*continuity_denom(U)
            numer_U = H.T @ Y + 2*alpha*N*continuity_numer(U)
            U = U * numer_U / get_nonzero(denom_U)
            
            #Normalize to ensure equal energy
            if norm_H == False:
                A = np.sqrt(np.sum(U**2, axis=1)/np.sum(H**2, axis=0))
                H = H * A[np.newaxis, :]
                U = U / A[:, np.newaxis]
            
            #Compute the loss function
            X = H @ U
            loss[i] = np.sum((Y - X)**2)
            loss[i] = loss[i] + alpha*continuity_cost(U)
    
    #In the case of Kullback–Leibler divergence
    elif loss_func == "KL":
        
        #Repeat num_iter times
        for i in range(num_iter):
            
            #Display a progress bar
            print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            if i % unit == 0:
                bar = "#" * int(np.ceil(i/unit)) + " " * int(np.floor((num_iter-i)/unit))
                print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            
            #Update the basements
            X = get_nonzero(H @ U)
            denom_H = U.T.sum(axis=0, keepdims=True)
            H = H * ((Y / X) @ U.T) / get_nonzero(denom_H)
            #Normalize the basements
            if norm_H == True:
                H = H / H.sum(axis=0, keepdims=True)
            
            #Update the weights
            X = get_nonzero(H @ U)
            denom_U = H.T.sum(axis=1, keepdims=True) + 4*alpha*N*continuity_denom(U)
            numer_U = H.T @ (Y / X) + 2*alpha*N*continuity_numer(U)
            U = U * numer_U / get_nonzero(denom_U)
            
            #Normalize to ensure equal energy
            if norm_H == False:
                A = np.sqrt(np.sum(U**2, axis=1)/np.sum(H**2, axis=0))
                H = H * A[np.newaxis, :]
                U = U / A[:, np.newaxis]
            
            #Compute the loss function
            X = get_nonzero(H @ U)
            loss[i] = np.sum(Y*np.log(Y) - Y*np.log(X) - Y + X)
            loss[i] = loss[i] + alpha*continuity_cost(U)
    
    #In the case of Itakura–Saito divergence
    elif loss_func == "IS":
            
        #Repeat num_iter times
        for i in range(num_iter):
            
            #Display a progress bar
            print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            if i % unit == 0:
                bar = "#" * int(np.ceil(i/unit)) + " " * int(np.floor((num_iter-i)/unit))
                print("\rNMF:[{0}] {1}/{2} Processing..".format(bar, i, num_iter), end="")
            
            #Update the basements
            X = get_nonzero(H @ U)
            denom_H = np.sqrt(X**-1 @ U.T)
            H = H * np.sqrt((Y / X**2) @ U.T) / get_nonzero(denom_H)
            #Normalize the basements (it is recommended when IS divergence)
            H = H / H.sum(axis=0, keepdims=True)
            
            #Update the weights
            X = get_nonzero(H @ U)
            denom_U = np.sqrt(H.T @ X**-1) + 4*alpha*N*continuity_denom(U)
            numer_U = np.sqrt(H.T @ (Y / X**2)) + 2*alpha*N*continuity_numer(U)
            U = U * numer_U / get_nonzero(denom_U)
            
            #Compute the loss function
            X = get_nonzero(X)
            loss[i] = np.sum(Y / X - np.log(Y) + np.log(X) - 1)
            loss[i] = loss[i] + alpha*continuity_cost(U)
    
    else:
        print("The deviation shold be either 'EU', 'KL', or 'IS'.")
        sys.exit()
    
    #Finish the progress bar
    bar = "#" * int(np.ceil(num_iter/unit))
    print("\rNMF:[{0}] {1}/{2} {3:.2f}sec Completed!".format(bar, i+1, num_iter, time.time()-start), end="")
    print()
    
    return H, U, loss

### Function for plotting Spectrogram and loss curve ###
def display_graph(Y, X, times, freqs, loss_func, num_iter):
    
    #Plot the original spectrogram
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.title('An original spectrogram')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    Y = 10*np.log10(np.abs(Y))
    plt.pcolormesh(times, freqs, Y, cmap='jet')
    plt.colorbar(orientation='horizontal').set_label('Power')
    plt.savefig("./log/original_spec.png", dpi=200)
    
    #Plot the approximated spectrogram
    plt.subplot(1, 2, 2)
    plt.title('The spectrogram approximated by NMF')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    X = 10*np.log10(np.abs(X))
    cm = plt.pcolormesh(times, freqs, X, cmap='jet', vmin=np.min(Y), vmax=np.max(Y))
    plt.colorbar(cm, orientation='horizontal').set_label('Power')
    plt.savefig("./log/reconstructed_spec.png", dpi=200)
    
    #Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, num_iter+1), loss[:], marker='.')
    plt.title(loss_func + '_loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.savefig("./log/loss_curve.png", dpi=200)
    
    return

### Function for applying low-pass filter ###
def lowpass(signal, cutoff):
    
    #Normalize passband and stopband edge frequency
    #Nyq = sr / 2
    #wp = fp / Nyq
    #ws = fs / Nyq
    #order, cutoff = sg.buttord(wp, ws, 3, 16) #if you wanna set details of filter
    
    #Compute Butterworth filter
    order = 4 
    b, a = sg.butter(order, cutoff, btype="lowpass", analog=False)
    
    #Apply the low-pass filter
    #output = sg.lfilter(b, a, signal) #Normal digital filter
    output = sg.filtfilt(b, a, signal) #Zero phase filtering
    
    return output

### Function for getting a shape-based distance (SBD) ###
def get_SBD(x, y):
    
    #Define FFT-size based on the length of input
    p = int(x.shape[0])
    FFTlen = int(2**np.ceil(np.log2(2*p-1)))
    
    #Compute the normalized cross-correlation function (NCC)
    CC = fp.ifft(fp.fft(x, FFTlen)*fp.fft(y, FFTlen).conjugate()).real
    
    #Reorder the IFFT result
    CC = np.concatenate((CC[-(p-1):], CC[:p]), axis=0)
    
    #To avoid zero division
    denom = linalg.norm(x) * linalg.norm(y)
    if denom < 1e-10:
        denom = numpy.inf
    NCC = CC / denom
    
    #Search for the argument to maximize NCC
    ndx = np.argmax(NCC, axis=0)
    dist = 1 - NCC[ndx]
    #Get the shift parameter (s=0 if no shift)
    s = ndx - p + 1
    
    #Insert zero padding based on the shift parameter s
    if s > 0:
        y_shift = np.concatenate((np.zeros(s), y[0:-s]), axis=0)
    elif s == 0:
        y_shift = np.copy(y)
    else:
        y_shift = np.concatenate((y[-s:], np.zeros(-s)), axis=0)
    
    return dist, y_shift

### Function for assigning fuzzy-label matrix ###
def assign_fuzzylabel(X, center, num_clu, m):
    
    #Define the length of input
    N = int(X.shape[0])
    
    #Initialize valuable for fuzzy-label
    label = np.zeros((num_clu, N))
    
    #Construct fuzzy label matrix for each row
    for i in range(N):
        for clu in range(num_clu):
            #Compute SBD for numerator
            numer_dist, _ = get_SBD(X[i, :], center[clu, :])
            
            #Get summation of the ratio between SBD
            for j in range(num_clu):
                #Compute SBD for denominator
                denom_dist, _ = get_SBD(X[i, :], center[j, :])
                if denom_dist < 1e-10:
                    denom_dist = 1e-10
                label[clu, i] = label[clu, i] + (numer_dist / denom_dist)**(1/(m-1))
    
    #Avoid zero division
    label = np.where(label < 1e-10, 1e-10, label)
    label = label**(-1)
    
    #Normalization (it is needed due to error handling)
    label = label / np.sum(label, axis=0, keepdims=True)
    
    return label

### Function for updating k-shape centroid ###
def shape_extraction(X, v):
    
    #Define the length of input
    N = int(X.shape[0])
    p = int(X.shape[1])
    
    #Construct the phase shifted signal
    Y = np.zeros((N, p))
    for i in range(N):
        #Call my function for getting the SBD between centeroid and data
        _, Y[i, :] = get_SBD(v, X[i, :])
    
    #Construct the matrix M for Rayleigh quotient
    S = Y.T @ Y
    Q = np.eye(p) - np.ones((p, p)) / p
    M = Q.T @ (S @ Q)
    
    #Get the eigenvector corresponding to the maximum eigenvalue
    eigen_val, eigen_vec = linalg.eig(M)
    ndx = np.argmax(eigen_val, axis=0)
    new_v = eigen_vec[:, ndx].real
    
    #The ill-posed problem has both +v and -v as solution
    MSE_plus = np.sum((Y - new_v)**2)
    MSE_minus = np.sum((Y + new_v)**2)
    if MSE_minus < MSE_plus:
        new_v = -1*new_v
    
    return new_v

### Function for checking empty clusters ###
def check_empty(label, num_clu):
    
    #Get unique label (which must include all number 0~num_clu-1)
    label = np.unique(label)
    
    #Search empty clusters
    emp_ind = []
    for i in range(num_clu):
        if i not in label:
            emp_ind.append(i)
    
    #Output the indices corresponding to the empty clusters
    return emp_ind

### Function for getting KShape clustering ###
def get_KShape(X, num_clu, max_iter, num_init):
    
    #Define the length of input
    N = int(X.shape[0])  #The number of data
    p = int(X.shape[1])  #The length of temporal axis
    
    #For a progress bar
    unit = int(np.floor(num_init/10))
    bar = "#" + " " * int(np.floor(num_init/unit))
    start = time.time()
    
    #Repeat for each trial (initialization)
    minloss = np.inf
    for init in range(num_init):
        
        #Display a progress bar
        print("\rk-shape:[{0}] {1}/{2} Processing..".format(bar, init, num_init), end="")
        if init % unit == 0:
            bar = "#" * int(np.ceil(init/unit)) + " " * int(np.floor((num_init-init)/unit))
            print("\rk-shape:[{0}] {1}/{2} Processing..".format(bar, init, num_init), end="")
        
        #Initialize label, centroid, loss as raondom numbers
        label = np.round((num_clu-1) * np.random.rand(N))
        center = np.random.rand(num_clu, p)
        loss = np.inf
        
        #Normalize the centroid
        center = center - np.average(center, axis=1)[:, np.newaxis]
        center = center / np.std(center, axis=1)[:, np.newaxis]
        
        #Copy the label temporarily
        new_label = np.copy(label)
        new_center = np.copy(center)
        
        #Repeat for each iteration
        for rep in range(max_iter):
            
            #Reset loss value
            new_loss = 0
            
            ### Refinement step (update center) ###
            #Repeat for each cluster
            for j in range(num_clu):
                
                #Construct data matrix for the j-th cluster
                clu_X = []
                for i in range(N):
                    #If the i-th data belongs to the j-th cluster
                    if label[i] == j:
                        clu_X.append(X[i, :])
                clu_X = np.array(clu_X)
                
                #Call my function for updating centroid
                new_center[j,:] = shape_extraction(clu_X, center[j,:])
                
                #Normalize the centroid
                new_center = new_center - np.average(new_center, axis=1)[:, np.newaxis]
                new_center = new_center / np.std(new_center, axis=1)[:, np.newaxis]
            
            ### Assignment step (update label) ###
            #Repeat for each data
            for i in range(N):
                
                #Define the minimum distance
                mindist = np.inf
                
                #Repeat for each cluster
                for j in range(num_clu):
                    
                    #Call my function for getting the shape based distance
                    dist, _ = get_SBD(new_center[j,:], X[i, :])
                    
                    #Assign the label corresponding to the minimum distance
                    if dist < mindist:
                        #Update minimum distance
                        mindist = dist
                        new_label[i] = j
                
                #Get summation of the SBD
                new_loss = new_loss + mindist
            
            ### Error handling (avoid empty clusters) ###
            #Call my function for checking empty clusters
            emp_ind = check_empty(new_label, num_clu)
            if len(emp_ind) > 0:
                for ind in emp_ind:
                    #Assign the same index of data as cluster
                    new_label[ind] = ind
            
            #Get out of the loop if loss and label unchange
            if loss - new_loss < 1e-6 and (new_label == label).all():
                #print("The iteration stopped at {}".format(rep+1))
                break
            
            #Update parameters
            label = np.copy(new_label)
            center = np.copy(new_center)
            loss = np.copy(new_loss)
            #print("Loss value: {:.3f}".format(new_loss))
        
        #Output the result corresponding to minimum loss
        if loss < minloss:
            out_label = np.copy(label).astype(np.int16)
            out_center = np.copy(center)
            minloss = loss
    
    #Finish the progress bar
    bar = "#" * int(np.ceil(num_init/unit))
    print("\rk-shape:[{0}] {1}/{2} {3:.2f}sec Completed!".format(bar, init+1, num_init, time.time()-start), end="")
    print()
    
    #Output the label vector and centroid matrix
    return out_label, out_center, minloss

### Function for getting KShape clustering ###
def get_fuzzyCShape(X, num_clu, max_iter, num_init, m):
    
    #Fuzzy coefficient m must be more than 1
    if m <= 1:
        m = 1 + 1e-5
    
    #Define the length of input
    N = int(X.shape[0])  #The number of data
    p = int(X.shape[1])  #The length of temporal axis
    
    #For a progress bar
    unit = int(np.floor(num_init/10))
    bar = "#" + " " * int(np.floor(num_init/unit))
    start = time.time()
    
    #Repeat for each trial (initialization)
    minloss = np.inf
    for init in range(num_init):
        
        #Display a progress bar
        print("\rc-shape:[{0}] {1}/{2} Processing..".format(bar, init, num_init), end="")
        if init % unit == 0:
            bar = "#" * int(np.ceil(init/unit)) + " " * int(np.floor((num_init-init)/unit))
            print("\rc-shape:[{0}] {1}/{2} Processing..".format(bar, init, num_init), end="")
        
        #Initialize label, centroid, loss as raondom numbers
        label = np.round((num_clu-1) * np.random.rand(N))
        center = np.random.rand(num_clu, p)
        loss = np.inf
        
        #Normalize the centroid
        center = center - np.average(center, axis=1)[:, np.newaxis]
        center = center / np.std(center, axis=1)[:, np.newaxis]
        
        #Copy the label temporarily
        new_label = np.copy(label)
        new_center = np.copy(center)
        
        #Repeat for each iteration
        for rep in range(max_iter):
            
            ### Assignment step (update label) ###
            #Call my function for getting fuzzy-label matrix
            fuzzy_label = assign_fuzzylabel(X, center, num_clu, m)
            
            #Harden the fuzzy-label matrix
            new_label = np.argmax(fuzzy_label, axis=0)
            
            #Call my function for checking empty clusters
            emp_ind = check_empty(new_label, num_clu)
            if len(emp_ind) > 0:
                for ind in emp_ind:
                    #Assign the same index of data as the one of cluster
                    new_label[ind] = ind
            
            ### Refinement step (update center) ###
            #Repeat for each cluster
            for j in range(num_clu):
                
                #Construct data matrix for the j-th cluster
                clu_X = []
                for i in range(N):
                    #If the i-th data belongs to the j-th cluster
                    if new_label[i] == j:
                        clu_X.append(X[i, :])
                clu_X = np.array(clu_X)
                
                #Call my function for updating centroid
                new_center[j,:] = shape_extraction(clu_X, center[j,:])
                
                #Normalize the centroid
                new_center = new_center - np.average(new_center, axis=1)[:, np.newaxis]
                new_center = new_center / np.std(new_center, axis=1)[:, np.newaxis]
            
            #Get the difference between the old and new center
            delta = linalg.norm(new_center - center)
            
            #Compute the loss function (generalized mean squares error)
            new_loss = 0
            for i in range(N):
                for j in range(num_clu):
                    dist, _ = get_SBD(X[i, :], new_center[j, :])
                    new_loss = new_loss + (fuzzy_label[j, i]**m) * dist
            
            #Get out of the loop if loss and label unchange
            if np.abs(loss-new_loss) < 1e-6 and (new_label == label).all():
                #print("The iteration stopped at {}".format(rep+1))
                break
            
            #Update parameters
            label = np.copy(new_label)
            center = np.copy(new_center)
            loss = np.copy(new_loss)
            #print("Loss value: {:.3f}".format(loss))
        
        #Output the result corresponding to minimum loss
        if loss < minloss:
            out_label = np.copy(fuzzy_label)
            out_center = np.copy(center)
            minloss = loss
    
    #Finish the progress bar
    bar = "#" * int(np.ceil(num_init/unit))
    print("\rc-shape:[{0}] {1}/{2} {3:.2f}sec Completed!".format(bar, init+1, num_init, time.time()-start), end="")
    print()
    
    #Output the label vector and centroid matrix
    return out_label, out_center, minloss

### Function for getting metrics such as SDR ###
def get_metrics(truth, estimates):
    
    #Compute the SDR by bss_eval from museval library ver.4
    truth = truth[np.newaxis, :, np.newaxis]
    estimates = estimates[np.newaxis, :, np.newaxis]
    sdr, isr, sir, sar, perm = bss_eval_images(truth, estimates)
    #The function 'bss_eval_sources' is NOT recommended by documentation
    #[Ref] J. Le Roux et.al., "SDR-half-baked or well done?" (2018)
    #[URL] https://arxiv.org/pdf/1811.02508.pdf
    #sdr, sir, sar, perm = bss_eval_sources(truth, estimates)
    
    return sdr[0,0], isr[0,0], sir[0,0], sar[0,0], perm[0,0]

### Main ###
if __name__ == "__main__":
    
    #Setup
    down_sam = None        #Downsampling rate (Hz) [Default]None
    frame_length = 0.064   #STFT window width (second) [Default]0.064
    frame_shift = 0.032    #STFT window shift (second) [Default]0.032
    num_iter = 200         #The number of iteration in NMF [Default]200
    num_base = 25          #The number of basements in NMF [Default]20~30
    alpha = 1e-4           #Weight of temporal continuity [Default]0 or 1e-4
    loss_func = "KL"       #Select either EU, KL, or IS divergence [Default]KL
    clu_mode = "kshape"    #Select either kshape or cshape [Default]kshape
    m = 1.3                #Using cshape, specify the fuzzy coefficient [Default]1.3 (>1.0)
    cutoff = 1.0           #Cutoff frequency for low-pass filter [Default]0.5 (<1.0=Nyquist)
    num_rep = 5            #The number of repetitions [Default]5
    
    #Define random seed
    np.random.seed(seed=32)
    
    #File path
    source1 = "./music/mixed.wav" #decompose it without training
    source2 = "./music/instrument1.wav" #for evaluation only
    source3 = "./music/instrument2.wav" #for evaluation only
    
    #Initialize variable for each metric
    SDR = np.zeros(num_rep)
    ISR = np.zeros(num_rep)
    SAR = np.zeros(num_rep)
    
    #Repeat for each iteration
    for rep in range(num_rep):
        
        #Prepare for process-log
        if clu_mode == "kshape":
            log_path = "./log/{},normalNMF_{},onset_{}".format(music, loss_func, clu_mode) + ".txt"
        elif clu_mode == "cshape":
            log_path = "./log/{},normalNMF_{},onset_{}{:.1f}".format(music, loss_func, clu_mode, m) + ".txt"
        else:
            print("The 'clu_mode' should be either 'kshape' or 'cshape'.")
            sys.exit()
        with open(log_path, "w") as f:
            f.write("")
        
        ### NMF step (to get basements matrix H) ###
        #Read mixed audio and true sources
        data, Fs = sf.read(source1)
        truth1, Fs = sf.read(source2)
        truth2, Fs = sf.read(source3)
        
        #Call my function for audio pre-processing
        data, Fs = pre_processing(data, Fs, down_sam)
        truth1, Fs = pre_processing(truth1, Fs, down_sam)
        truth2, Fs = pre_processing(truth2, Fs, down_sam)
        
        #Call my function for getting STFT (amplitude or power)
        Y, arg, Fs, freqs, times = get_STFT(data, Fs, frame_length, frame_shift)
        
        #Call my function for updating NMF basements and weights
        H, U, loss = get_NMF(Y, num_iter, num_base, loss_func, alpha, True)
        
        #Call my function for getting inverse STFT
        X = H @ U
        rec_wav, Fs = get_invSTFT(X, arg, Fs, frame_length, frame_shift)
        rec_wav = rec_wav[: int(data.shape[0])] #inverse stft includes residual part due to zero padding
        
        #Call my function for displaying graph
        #display_graph(Y, X, times, freqs, loss_func, num_iter)
        
        ### Clustering step (to get label for each sound source) ###
        #Define feature vectors (activation matrix)
        onset = np.copy(U)
        
        #Normalization
        onset = onset - np.mean(onset, axis=1, keepdims=True)
        onset = onset / np.std(onset, axis=1, keepdims=True)
        
        #Apply a low pass filter along with temporal axis
        for i in range(num_base):
            if cutoff < 1.0:
                #Call my function for applying low-pass filter
                onset[i, :] = lowpass(onset[i, :], cutoff)
                #plt.figure(figsize=(10, 5))
                #plt.plot(feat[i, :], marker='.')
                #plt.show()
        
        #Get clustering by either DTW_kmeans, or KShape
        if clu_mode == "kshape":
            max_iter, num_init = 100, 10
            label1, _, loss = get_KShape(onset, 2, max_iter, num_init)
        
        elif clu_mode == "cshape":
            max_iter, num_init = 100, 10
            fuzzy_label, _, loss = get_fuzzyCShape(onset, 2, max_iter, num_init, m)
            label1 = np.argmax(fuzzy_label, axis=0)
        
        #print("Clustering vector a(i):{}".format(label1))
        label2 = np.ones(num_base) - label1
        label1 = label1[np.newaxis, :]
        label2 = label2[np.newaxis, :]
        
        #Decide which label corresponds to source1
        X = (H * label1) @ U
        rec_wav, Fs = get_invSTFT(X, arg, Fs, frame_length, frame_shift)
        rec_wav = rec_wav[: int(truth1.shape[0])] #inverse stft includes residual part due to zero padding
        sdr1,_,_,_,_ = get_metrics(truth1, rec_wav)
        sdr2,_,_,_,_ = get_metrics(truth2, rec_wav)
        if sdr1 > sdr2:
            H1 = H * label1
            H2 = H * label2
        else:
            H1 = H * label2
            H2 = H * label1
        
        #Get separation by using Wiener filter
        X1 = Y * (H1 @ U) / (H @ U)
        X2 = Y * (H2 @ U) / (H @ U)
            
        #Call my function for getting inverse STFT
        sep_wav1, Fs = get_invSTFT(X1, arg, Fs, frame_length, frame_shift)
        sep_wav1 = sep_wav1[: int(truth1.shape[0])] #inverse stft includes residual part due to zero padding
        sep_wav2, Fs = get_invSTFT(X2, arg, Fs, frame_length, frame_shift)
        sep_wav2 = sep_wav2[: int(truth2.shape[0])] #inverse stft includes residual part due to zero padding
        
        ### Evaluation step (to get SDR (signal-to-distortion ratio) of estimates) ###
        #Save the estimated sources
        sf.write("./log/" + str(rep) + "_Truth1.wav", truth1, Fs)
        sf.write("./log/" + str(rep) + "_Estimate1.wav", sep_wav1, Fs)
        sf.write("./log/" + str(rep) + "_Truth2.wav", truth2, Fs)
        sf.write("./log/" + str(rep) + "_Estimate2.wav", sep_wav2, Fs)
        
        #Call my function for getting metrics such as SDR
        sdr, isr, sir, sar, perm = get_metrics(truth1, sep_wav1)
        print("SDR: {:.3f} [dB]".format(sdr))
        
        #Save metric for each iteration
        SDR[rep], ISR[rep], SAR[rep] = sdr, isr, sar
    
    #Calculate average and confidence interval for each metric
    aveSDR, seSDR = np.average(SDR), 1.96*np.std(SDR) / np.sqrt(num_rep-1)
    aveISR, seISR = np.average(ISR), 1.96*np.std(ISR) / np.sqrt(num_rep-1)
    aveSAR, seSAR = np.average(SAR), 1.96*np.std(SAR) / np.sqrt(num_rep-1)
    with open(log_path, "a") as f:
        f.write(u"SDR={:.4f}\u00b1{:.4f}[dB]\nISR={:.4f}\u00b1{:.4f}[dB]\nSAR={:.4f}\u00b1{:.4f}[dB]".format(
            aveSDR, seSDR, aveISR, seISR, aveSAR, seSAR))