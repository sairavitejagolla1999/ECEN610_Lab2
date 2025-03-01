import numpy as np
import matplotlib.pyplot as plt

freq = 2e6
fs= 5e6
ampl = 1
t = np.arange(0 , 5*50e-6,1/fs)
x = ampl*np.sin(2*np.pi*freq*t)


plt.figure(figsize=(8, 8))
plt.plot(t,x)
plt.title('Sampled Tone')
plt.xlabel('Time')
plt.ylabel('Ampl')
plt.grid()
plt.show()

# /*Part1*/

x_noise = x + np.random.normal(0,np.sqrt(5e-6),len(t))

plt.figure(figsize=(8, 8))
plt.plot(x_noise)
plt.title('X signal with Noise')
plt.xlabel('Time')
plt.ylabel('Ampl')
plt.grid()
plt.show()


plt.figure(figsize=(8, 8))
plt.plot(x_noise-x)
plt.title('Noise')
plt.xlabel('Time')
plt.ylabel('Ampl')
plt.grid()
plt.show()


fft_result = np.fft.fft(x_noise)
fft_sqr = np.abs(np.fft.fft(x_noise))**2/(len(x_noise)*fs)
fft_freq = np.fft.fftfreq(len(x_noise),1/fs)

frequncy_fft = fft_freq[:len(fft_freq)//2]
psd = fft_sqr[:len(fft_freq)//2]

psd_max = np.max(psd)
noise_power = np.sum(psd) - psd_max
snr = 10*np.log10(psd_max/noise_power)

print("PSD Max is", psd_max)
print("snr calculated is =",snr)
print("snr expected is 50dB")


plt.figure(figsize=(8, 8))
plt.plot(frequncy_fft,psd)
plt.title('PSD of x_noise')
plt.xlabel('Freq')
plt.ylabel('PSD')
plt.grid()
plt.show()


#Part B - Hanning Window
window = np.hanning(len(x_noise))
x_noise = x_noise * window
fft_sqr = np.abs(np.fft.fft(x_noise))**2/(len(x_noise)*fs)
fft_freq = np.fft.fftfreq(len(x_noise),1/fs)

frequncy_fft = fft_freq[:len(fft_freq)//2]
psd = fft_sqr[:len(fft_freq)//2]

bin_range = 0
index_peak = np.argmax(psd)
start_idx = max(0,index_peak - bin_range)
end_idx = min(len(psd),index_peak + bin_range + 1)
sig_power = np.sum(psd[start_idx:end_idx])

noise_power = np.sum(psd) - sig_power
snr = 10*np.log10(sig_power/noise_power)

print("Bin Range is ",bin_range)
print("snr calculated Hanning is =",snr)
print("snr expected is 50dB")


plt.figure(figsize=(8, 8))
plt.plot(frequncy_fft,psd)
plt.title('PSD of x_noise Hanning Window')
plt.xlabel('Freq')
plt.ylabel('PSD')
plt.grid()
plt.show()

#Part B - Hamming Window
window = np.hamming(len(x_noise))
x_noise = x_noise * window
fft_sqr = np.abs(np.fft.fft(x_noise))**2/(len(x_noise)*fs)
fft_freq = np.fft.fftfreq(len(x_noise),1/fs)

frequncy_fft = fft_freq[:len(fft_freq)//2]
psd = fft_sqr[:len(fft_freq)//2]

bin_range = 2
index_peak = np.argmax(psd)
start_idx = max(0,index_peak - bin_range)
end_idx = min(len(psd),index_peak + bin_range + 1)
sig_power = np.sum(psd[start_idx:end_idx])

noise_power = np.sum(psd) - sig_power
snr = 10*np.log10(sig_power/noise_power)

print("Bin Range is ",bin_range)
print("snr calculated Hamming is =",snr)
print("snr expected is 50dB")


plt.figure(figsize=(8, 8))
plt.plot(frequncy_fft,psd)
plt.title('PSD of x_noise Hamming Window')
plt.xlabel('Freq')
plt.ylabel('PSD')
plt.grid()
plt.show()

#Part B - Blackman Window
window = np.blackman(len(x_noise))
x_noise = x_noise * window
fft_sqr = np.abs(np.fft.fft(x_noise))**2/(len(x_noise)*fs)
fft_freq = np.fft.fftfreq(len(x_noise),1/fs)

frequncy_fft = fft_freq[:len(fft_freq)//2]
psd = fft_sqr[:len(fft_freq)//2]

bin_range = 2
index_peak = np.argmax(psd)
start_idx = max(0,index_peak - bin_range)
end_idx = min(len(psd),index_peak + bin_range + 1)
sig_power = np.sum(psd[start_idx:end_idx])

noise_power = np.sum(psd) - sig_power
snr = 10*np.log10(sig_power/noise_power)

print("Bin Range is ",bin_range)
print("snr calculated Blackman is =",snr)
print("snr expected is 50dB")


plt.figure(figsize=(8, 8))
plt.plot(frequncy_fft,psd)
plt.title('PSD of x_noise Blackman Window')
plt.xlabel('Freq')
plt.ylabel('PSD')
plt.grid()
plt.show()