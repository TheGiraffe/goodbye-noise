import scipy.io.wavfile as wavfile
from scipy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

fs, noiseData = wavfile.read('talking_clip.wav')
# fs, noiseData = wavfile.read('bg_noise.wav')
nSamplesNoise = noiseData.shape[0]
print(noiseData)
print(nSamplesNoise) #35200
print(fs) #48000

def rebuildStereo(ChL, ChR, nSamples):
    rows = nSamples
    cols = 2
    dataStereo = np.zeros((rows, cols)).astype(complex)
    for i in range(nSamples):
        dataStereo[i][0] = ChL[i]
        dataStereo[i][1] = ChR[i]
    return dataStereo

noiseDataL = noiseData[:, 0]
noiseDataR = noiseData[:, 1]

noiseFFTL = fft(noiseDataL)
noiseFFTR = fft(noiseDataR)

noiseFFT = rebuildStereo(noiseFFTL, noiseFFTR, nSamplesNoise)

# Plot Style
frequencies = (fs/nSamplesNoise)*np.linspace(-nSamplesNoise, nSamplesNoise, nSamplesNoise)
axBgCol = (62/255, 66/255, 77/255)
bgCol = (33/255, 35/255, 41/255)
textCol = "mintcream"
ax1Col = "darkturquoise"
ax2Col = "deeppink"
ax1GridCol = "powderblue"
ax2GridCol = "thistle"
plt.rcParams['text.color'] = textCol
plt.rcParams['axes.labelcolor'] = textCol
plt.rcParams['xtick.color'] = textCol
plt.rcParams['ytick.color'] = textCol

midpoint_n = int(nSamplesNoise/2) # this occurs at sample 1223240 in orig.wav.

# noiseFFTdB = np.log10(noiseFFT)*20

noiseFFTgraph1 = abs(noiseFFT[midpoint_n:int(nSamplesNoise)])
noiseFFTgraph2 = abs(noiseFFT[0:midpoint_n])

plt.rcParams["font.family"] = ["American Typewriter", "serif"]

fig, ax = plt.subplots()
ax.plot(frequencies[0:midpoint_n], noiseFFTgraph1, color=ax2Col, label="orig.wav")
ax.plot(frequencies[midpoint_n:nSamplesNoise], noiseFFTgraph2, label='_nolegend_', color=ax2Col)
ax.set_facecolor(axBgCol)
ax.grid(True, color=ax1GridCol)
fig.set_facecolor(bgCol)

ylab = "Amplitude (dB)"
xlab="Frequency (Hz)"
fig.supxlabel(xlab)
fig.supylabel(ylab)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
frame = leg.get_frame()
frame.set_facecolor(bgCol)
frame.set_linewidth(0)
fig.suptitle("Frequency Domain of Noise", weight="bold", fontsize="x-large")
plt.show()

def goodbyeNoise(freqCutoff1, freqCutoff2, dataFFT, fs, nSamples):
    # midpoint_freq = int(fs/2) 
    # midpoint_sample = int(nSamples/2)
    # sampleCutoff1  = int((freqCutoff1*midpoint_sample)/midpoint_freq)
    # sampleCutoff2 = int((freqCutoff2*midpoint_sample)/midpoint_freq)
    sampleCutoff1 = int((freqCutoff1 / fs) * nSamples/2)
    sampleCutoff2 = int((freqCutoff2 / fs) * nSamples/2)
    mirror_sampleCutoff1 = nSamples - sampleCutoff1
    mirror_sampleCutoff2 = nSamples - sampleCutoff2
    dataFFT[sampleCutoff1:sampleCutoff2] = np.zeros_like(dataFFT[sampleCutoff1:sampleCutoff2])
    dataFFT[mirror_sampleCutoff2:mirror_sampleCutoff1] = np.zeros_like(dataFFT[mirror_sampleCutoff2:mirror_sampleCutoff1])
    return dataFFT

def amplifyVoice(gain, freqCutoff1, freqCutoff2, dataFFT, fs, nSamples):
    sampleCutoff1 = int((freqCutoff1 / fs) * nSamples/2)
    sampleCutoff2 = int((freqCutoff2 / fs) * nSamples/2)
    mirror_sampleCutoff1 = nSamples - sampleCutoff1
    mirror_sampleCutoff2 = nSamples - sampleCutoff2
    dataFFT[sampleCutoff1:sampleCutoff2] = dataFFT[sampleCutoff1:sampleCutoff2]*gain
    dataFFT[mirror_sampleCutoff2:mirror_sampleCutoff1] = dataFFT[mirror_sampleCutoff2:mirror_sampleCutoff1]*gain
    return dataFFT

denoisedFFT = noiseFFT.copy()
denoisedFFT = goodbyeNoise(200, 500, denoisedFFT, fs, nSamplesNoise)
# denoisedFFT = goodbyeNoise(430, 500, denoisedFFT, fs, nSamplesNoise)
# denoisedFFT = goodbyeNoise(1080, 1200, denoisedFFT, fs, nSamplesNoise)
denoisedFFT = amplifyVoice(2, 1600, 1800, denoisedFFT, fs, nSamplesNoise)
# denoisedFFT = goodbyeNoise(1750, 1800, denoisedFFT, fs, nSamplesNoise)
denoisedFFT = goodbyeNoise(17000, 18000, denoisedFFT, fs, nSamplesNoise)

denoisedFFTgraph1 = abs(denoisedFFT[midpoint_n:int(nSamplesNoise)])
denoisedFFTgraph2 = abs(denoisedFFT[0:midpoint_n])

fig, ax = plt.subplots()
ax.plot(frequencies[0:midpoint_n], noiseFFTgraph1, color=ax2Col, label="orig.wav")
ax.plot(frequencies[midpoint_n:nSamplesNoise], noiseFFTgraph2, label='_nolegend_', color=ax2Col)
ax.plot(frequencies[0:midpoint_n], denoisedFFTgraph1, color=ax1Col, alpha=0.25, label="denoised.wav")
ax.plot(frequencies[midpoint_n:nSamplesNoise], denoisedFFTgraph2, color=ax1Col, alpha=0.25)
ax.set_facecolor(axBgCol)
ax.grid(True, color=ax1GridCol)
fig.set_facecolor(bgCol)

ylab = "Amplitude (dB)"
xlab="Frequency (Hz)"
fig.supxlabel(xlab)
fig.supylabel(ylab)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
frame = leg.get_frame()
frame.set_facecolor(bgCol)
frame.set_linewidth(0)
fig.suptitle("Frequency Domain of Noise vs Denoised", weight="bold", fontsize="x-large")
plt.show()

def buildNewData(newDataFFT, nSamples):
    # Separate out into L and R channels again:
    newDataFFTL = newDataFFT[:, 0]
    newDataFFTR = newDataFFT[:, 1]
    # IFFT 'em:
    newDataChL = ifft(newDataFFTL)
    newDataChR = ifft(newDataFFTR)
    # Rebuild 'em into Stereo data again:
    newData = rebuildStereo(newDataChL, newDataChR, nSamples)
    newData = np.int32(newData)
    return newData

denoisedData = buildNewData(denoisedFFT, nSamplesNoise)
regularData = buildNewData(noiseFFT, nSamplesNoise)
wavfile.write("denoised_bg_noise.wav", int(fs), denoisedData)

# print(regularData)
# print(denoisedData)

fig, ax = plt.subplots()
ax.plot(noiseData[:, 0])
ax.plot(regularData[:, 0])
ax.plot(denoisedData[:, 0])
plt.show()