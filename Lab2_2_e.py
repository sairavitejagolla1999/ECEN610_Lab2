import numpy as np
import matplotlib.pyplot as plt


freq = 200e6
#for Part A
fs= 400e6

res_bits = 12
quan_levels = 2**12

no_period = 100
# no_period = 100

t = np.arange(0,no_period/freq,1/fs)
x = np.cos(2*np.pi*freq*t)



noise = np.random.normal(0,np.sqrt(7.9244e-5),len(x))
x = x + noise

window = np.hanning(len(x))
x = x*window

quan_x= np.round(x*(quan_levels/2))/(quan_levels/2)


psd,frequency_fft = plt.psd(quan_x, len(quan_x),fs, pad_to=2*len(quan_x))

psd_max = np.sum(psd[len(psd)-6:len(psd)])

noise_power = np.sum(psd) - psd_max

snr = 10*np.log10(psd_max/noise_power)
print("snr calculated is =",snr)



plt.figure(figsize=(8, 8))


plt.subplot(3, 1, 1)
plt.stem(t, x)
plt.title('Sampled Tone')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# plt.subplot(4, 1, 2)
# plt.stem(t, window)
# plt.title('Window')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(t, quan_x)
plt.title('Quantized x')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1,3)
plt.plot(frequency_fft, psd)
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Freq')
plt.ylabel('PSD')
plt.grid(True)

plt.tight_layout()
plt.show()