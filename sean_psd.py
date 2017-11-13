import numpy as nm
import scipy
import scipy.fftpack
import matplotlib.pyplot as plt

def amplitude_and_power_spectrum( timestream, tau, return_amplitudes=False ):
    # in the scanned pdf notes
    # timestream is v
    # yfft is v-tilde

    # delete the last data point if the timestream has an odd number of data points
    if nm.mod(len(timestream),2):
        timestream = timestream[0:-1]
        # gently warn that this has happened
        print 'Note: final data point deleted for FFT'

    # number of data points in the voltage timestream
    N = len(timestream)

    # take fft, shift it, and normalize with the factor of N
    yfft = scipy.fftpack.fft(timestream)
    yfft = scipy.fftpack.fftshift(yfft)
    yfft = (1.0/N)*yfft

    # make frequency array for fft
    delta = 1.0/(N*tau)
    nyquist = 1.0/(2.0*tau)
    freq = nm.arange(-nyquist,nyquist,delta) # note that the last element will be (nyquist-delta) because of python's conventions

    # make positive frequency array
    pos_freq = freq[(N/2):]
    # because of roundoff error, the zeroth frequency bin appears to be 1e-12 or something
    # fix it to zero by definition
    pos_freq[0] = 0.0

    # as an intermediate step, normalize the FFT such that sum(psd) = rms(timestream)
    temp = yfft[(N/2):] # positive freq half of fft
    psd = temp*nm.conj(temp)
    psd = 2.0*psd
    psd[0] = psd[0]/2.0 # special case for DC

    if return_amplitudes:
        # Amplitude and Phase spectrum
        # calculate amplitudes from the psd variable
        amplitudes = nm.sqrt(2.0*psd) # root-two comes from the conversion from rms to peak-to-peak amplitude
        amplitudes[0] = nm.sqrt(psd[0]) # special case for DC
        # calculate phase angles from yfft
        phases = nm.arctan2(nm.imag(yfft[(N/2):]),nm.real(yfft[(N/2):]))
        return pos_freq,amplitudes,phases   
    else:
        v2_per_hz = psd / delta
        # since it's zero, throw away imaginary part
        v2_per_hz = nm.real(v2_per_hz)
        return pos_freq,v2_per_hz

