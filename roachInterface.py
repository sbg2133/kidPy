# This software is a work in progress. It is a console interface designed 
# to operate the BLAST-TNG ROACH2 firmware. 
#
# Copyright (C) November, 2017  Gordon, Sam <sbgordo1@asu.edu>
# Author: Gordon, Sam <sbgordo1@asu.edu>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import matplotlib, time, struct
import numpy as np
import shutil
np.set_printoptions(threshold=np.nan)
matplotlib.use("TkAgg")
import matplotlib.lines
import matplotlib.pyplot as plt
import casperfpga 
from myQdr import Qdr as myQdr
import types
import logging
import glob  
import os
import sys
import valon_synth9
from scipy import signal
import scipy.fftpack
import pygetdata as gd
from sean_psd import amplitude_and_power_spectrum as sean_psd

class roachInterface(object):
    
    def __init__(self, fpga, gc, regs, valon):
	self.gc = gc
	self.fpga = fpga
	self.regs = regs
	self.synth = valon
	self.firmware = self.gc[np.where(self.gc == 'FIRMWARE_FILE')[0][0]][1]
	self.Nfreq = int(self.gc[np.where(self.gc == 'Nfreq')[0][0]][1])
	self.max_pos_freq = np.float(self.gc[np.where(self.gc == 'max_pos_freq')[0][0]][1])
	self.min_pos_freq = np.float(self.gc[np.where(self.gc == 'min_pos_freq')[0][0]][1])
	self.max_neg_freq = np.float(self.gc[np.where(self.gc == 'max_neg_freq')[0][0]][1])
	self.min_neg_freq = np.float(self.gc[np.where(self.gc == 'min_neg_freq')[0][0]][1])
	self.symm_offset = np.float(self.gc[np.where(self.gc == 'symm_offset')[0][0]][1])
	pos_freqs, self.pos_delta = np.linspace(self.min_pos_freq, self.max_pos_freq, self.Nfreq, retstep = True)
	neg_freqs, self.neg_delta = np.linspace(self.min_neg_freq + self.symm_offset, self.max_neg_freq + self.symm_offset, self.Nfreq, retstep = True)
	self.center_freq = np.float(self.gc[np.where(self.gc == 'center_freq')[0][0]][1]) 
	self.dac_samp_freq = 512.0e6
        self.fpga_samp_freq = 256.0e6
	self.bin_fs = 500.0e3 # FFT bin sampling freq
	self.hanning = signal.firwin(23, 10.0e3, window='hanning',nyq = 0.5*self.bin_fs)
	self.boxcar = (1./23.)*np.ones(23)
	self.LUTbuffer_len = 2**21
        self.dac_freq_res = 2*self.dac_samp_freq/self.LUTbuffer_len
        self.fft_len = 1024
        self.accum_len = 2**19 
        self.accum_freq = self.fpga_samp_freq / self.accum_len
	self.I_dds = np.zeros(self.LUTbuffer_len)
	self.freq_comb = []
    	
    def uploadfpg(self):
        print 'Connecting...'
        t1 = time.time()
        timeout = 10
        while not self.fpga.is_connected():
            if (time.time() - t1) > timeout:
        	raise Exception("Connection timeout to roach")
        time.sleep(0.1)
        if self.fpga.is_connected():
            print 'Connected to', self.fpga.host
            self.fpga.upload_to_ram_and_program(self.firmware)
        else:
            print 'No connection to the ROACH'
	    return -1
        time.sleep(2)
        print 'Uploaded', self.firmware
	return 0

    def makeFreqComb(self):
	neg_freqs, neg_delta = np.linspace(self.min_neg_freq + self.symm_offset, self.max_neg_freq + self.symm_offset, self.Nfreq/2, retstep = True)
	pos_freqs, pos_delta = np.linspace(self.min_pos_freq, self.max_pos_freq, self.Nfreq/2, retstep = True)
        freq_comb = np.concatenate((neg_freqs, pos_freqs))
	freq_comb = freq_comb[freq_comb != 0]
	freq_comb = np.roll(freq_comb, - np.argmin(np.abs(freq_comb)) - 1)
        if len(freq_comb) > 400:
            self.fpga.write_int(self.regs[np.where(self.regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)    
            time.sleep(0.1)
        else:
            self.fpga.write_int(self.regs[np.where(self.regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)    
            time.sleep(0.1)
	self.freq_comb = freq_comb
	return

    def lpf(self, window):
	window *=(2**31 - 1)
	for i in range(len(window)/2 + 1):
		coeff = np.binary_repr(int(window[i]), 32)
		coeff = int(coeff, 2)
		#print 'h' + str(i), coeff	
		self.fpga.write_int(self.regs[np.where(self.regs == 'fir_prefix_reg')[0][0]][1] + str(i),coeff)
	return 

    def qdrCal(self):    
    # Calibrates the QDRs. Run after writing to QDR.      
        self.fpga.write_int(self.regs[np.where(self.regs == 'dac_reset_reg')[0][0]][1],1)
        print 'DAC on'
        bFailHard = False
        calVerbosity = 1
        qdrMemName = self.regs[np.where(self.regs == 'qdr0_reg')[0][0]][1]
        qdrNames = [self.regs[np.where(self.regs == 'qdr0_reg')[0][0]][1],self.regs[np.where(self.regs == 'qdr1_reg')[0][0]][1]]
        print 'Fpga Clock Rate =', self.fpga.estimate_fpga_clock()
        self.fpga.get_system_information()
        results = {}
        for qdr in self.fpga.qdrs:
            print qdr
            mqdr = myQdr.from_qdr(qdr)
            results[qdr.name] = mqdr.qdr_cal2(fail_hard=bFailHard)
        print 'qdr cal results:',results
        for qdrName in ['qdr0','qdr1']:
            if not results[qdr.name]:
                print 'Calibration Failed'
                return -1
        print '\n************ QDR Calibrated ************'
        return 0

    def rudinshapiro(self, N):
    	"""
    	Return first N terms of Rudin-Shapiro sequence
   	https://en.wikipedia.org/wiki/Rudin-Shapiro_sequence
        Confirmed correct output to N = 10000:
        https://oeis.org/A020985/b020985.txt
        """
    	def hamming(x):
        	"""
        	Hamming weight of a binary sequence
        	http://stackoverflow.com/a/407758/125507
        	"""
        	return bin(x).count('1')
    
   	out = np.empty(N, dtype=int)
    	for n in xrange(N):
        	b = hamming(n << 1 & n)
        	a = (-1)**b
        	out[n] = a
	return out	

    def read_mixer_snaps(self, chan, mixer_out = True):
     	if (chan % 2) > 0: # if chan is odd
            self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_chan_sel_reg')[0][0]][1], (chan - 1) / 2)
        else:
            self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_chan_sel_reg')[0][0]][1], chan/2)
        self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_fftbin_ctrl_reg')[0][0]][1], 0)
        self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_mixerout_ctrl_reg')[0][0]][1], 0)
	self.fpga.write_int(self.regs[np.where(self.regs == 'fir_snap_ctrl_reg')[0][0]][1], 0)
        self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_fftbin_ctrl_reg')[0][0]][1], 1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_mixerout_ctrl_reg')[0][0]][1], 1)
	self.fpga.write_int(self.regs[np.where(self.regs == 'fir_snap_ctrl_reg')[0][0]][1], 1)
        mixer_in = np.fromstring(self.fpga.read(self.regs[np.where(self.regs == 'DDC_fftbin_bram_reg')[0][0]][1], 16*2**14),dtype='>i2').astype('float')
        if mixer_out:
            mixer_out = np.fromstring(self.fpga.read(self.regs[np.where(self.regs == 'DDC_mixerout_bram_reg')[0][0]][1], 8*2**14),dtype='>i2').astype('float')
	    lpf_out = np.fromstring(self.fpga.read(self.regs[np.where(self.regs == 'DDC_fftbin_bram_reg')[0][0]][1], 8*2**14),dtype='>i2').astype('float')
            return mixer_in, mixer_out, lpf_out
        else:
            return mixer_in

    def read_mixer_shift(self, shift, chan, mixer_out = True):
    # returns snap data for the dds mixer inputs and outputs
        self.fpga.write_int(self.regs[np.where(self.regs == 'dds_shift')[0][0]][1], shift)
        if (chan % 2) > 0: # if chan is odd
            self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_chan_sel_reg')[0][0]][1], (chan - 1) / 2)
        else:
            self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_chan_sel_reg')[0][0]][1], chan/2)
        self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_fftbin_ctrl_reg')[0][0]][1], 0)
        self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_mixerout_ctrl_reg')[0][0]][1], 0)
        self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_fftbin_ctrl_reg')[0][0]][1], 1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'DDC_mixerout_ctrl_reg')[0][0]][1], 1)
        mixer_in = np.fromstring(self.fpga.read(self.regs[np.where(self.regs == 'DDC_fftbin_bram_reg')[0][0]][1], 16*2**10),dtype='>i2').astype('float')
        if mixer_out:
            mixer_out = np.fromstring(self.fpga.read(self.regs[np.where(self.regs == 'DDC_mixerout_bram_reg')[0][0]][1], 8*2**10),dtype='>i2').astype('float')
            return mixer_in, mixer_out
        else:
            return mixer_in
    
    def mixer_comp(self, chan, I0 = True):
	mixer_in, mixer_out, lpf = self.read_mixer_snaps(chan)
	if I0:
		I_in = mixer_in[0::8]
		Q_in = mixer_in[1::8]
		I_dds_in = mixer_in[2::8]
		Q_dds_in = mixer_in[3::8]
		I_out = mixer_out[0::4]
		Q_out = mixer_out[1::4]
		I_lpf = lpf[0::4]
		Q_lpf = lpf[1::4]
	else:
		I_in = mixer_in[4::8]
		Q_in = mixer_in[5::8]
		I_dds_in = mixer_in[6::8]
		Q_dds_in = mixer_in[7::8]
		I_out = mixer_out[2::4]
		Q_out = mixer_out[3::4]
		I_lpf = lpf[2::4]
		Q_lpf = lpf[3::4]
	return I_in, Q_in, I_dds_in, Q_dds_in, I_out, Q_out, I_lpf, Q_lpf 

    def plotBin(self, chan):
	fs = 500.0
	#w, h = signal.freqz(zeros)
	#freq_resp = np.concatenate((h[::-1], h))
	#f = w*fs/(2*np.pi)
	#freqs = np.concatenate((-f[::-1], f))
	fig = plt.figure(figsize=(20,12))
	N = 16384
	T = 1./fs
	nyquist = 0.5*fs
	pos_freqs = np.linspace(0, nyquist, 8192)
	neg_freqs = np.linspace(-nyquist, 0, 8192)
	xf = np.concatenate((pos_freqs, neg_freqs))
	#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
	plot1 = fig.add_subplot(411)
	#plt.xticks(np.arange(-250., 250., 10))
	#plt.title("chan " + str(chan) + ", freq = " + str(self.freq_comb[chan]/1.0e6) + " MHz", size = 12)
	line1, = plot1.plot(xf, np.zeros(N), label = 'Bin in', color = 'green', linewidth = 1)
	plt.grid()
	plot2 = fig.add_subplot(412)
	#plt.xticks(np.arange(-250., 250., 10))
	plt.title("DDC", size = 12)
	line2, = plot2.plot(xf, np.zeros(N), label = 'DDC', color = 'red', linewidth = 2)
	plt.grid()
	plot3 = fig.add_subplot(413)
	#plt.xticks(np.arange(-250., 250., 10))
	plt.title('chan out = F_in - F_ddc', size = 12)
	line3, = plot3.plot(xf, np.zeros(N), label = 'Bin out', color = 'blue', linewidth = 2)
	plt.grid()
	plot4 = fig.add_subplot(414)
	line4, = plot4.plot(xf, np.zeros(N), label = 'FIR out', color = 'k', linewidth = 2)
	#plot4.plot(freqs, 20*np.log10(np.abs(freq_resp)/np.max(np.abs(freq_resp))), color = 'r', linewidth = 2)
	plt.grid()
	plt.tight_layout()
	plt.show(block = False)
	count = 0
	stop = 10000
	while (count < stop):
		if (chan % 2) > 0:
			I_in, Q_in, I_dds_in, Q_dds_in, I_out, Q_out, I_lpf, Q_lpf = self.mixer_comp(chan, I0 = False)
		else:
			I_in, Q_in, I_dds_in, Q_dds_in, I_out, Q_out, I_lpf, Q_lpf = self.mixer_comp(chan)
		
		sig_in = I_in + 1j*Q_in
		ddc = I_dds_in + 1j*Q_dds_in
		sig_out = I_out + 1j*Q_out
		lpf = I_lpf + 1j*Q_lpf
		fft_in = np.abs(scipy.fftpack.fft(sig_in))
		fft_in /= np.max(fft_in)
		fft_in = 20*np.log10(fft_in)
		ddc_fft = np.abs(scipy.fftpack.fft(ddc))
		ddc_fft /= np.max(ddc_fft)
		fft_out = np.abs(scipy.fftpack.fft(sig_out))
		fft_out /= np.max(fft_out)
		fft_out = 20*np.log10(fft_out)
		lpf_fft = np.abs(scipy.fftpack.fft(lpf))
		lpf_fft /= np.max(lpf_fft)
		lpf_fft = 20*np.log10(lpf_fft)
		line1.set_ydata(fft_in) 
		plot1.set_ylim(np.min(fft_in), 0)
		line2.set_ydata(ddc_fft)
		plot2.set_ylim(-0.5, 1.5)
		line3.set_ydata(fft_out)
		plot3.set_ylim(np.min(fft_out), 0)
		line4.set_ydata(lpf_fft)
		plot4.set_ylim(np.min(lpf_fft), 0)
		fig.canvas.draw()
		count += 1
	return
    
    def return_shift(self, chan):
    # Returns the dds shift
        dds_spec = np.abs(np.fft.rfft(self.I_dds[chan::self.fft_len],self.fft_len))
        dds_index = np.where(np.abs(dds_spec) == np.max(np.abs(dds_spec)))[0][0]
        print 'Finding LUT shift...' 
        for i in range(self.fft_len/2):
            #print i
            mixer_in = self.read_mixer_shift(i, chan, mixer_out = False)
            I0_dds_in = mixer_in[2::8]    
       	    #print I0_dds_in[:100]
            snap_spec = np.abs(np.fft.rfft(I0_dds_in,self.fft_len))
            snap_index = np.where(np.abs(snap_spec) == np.max(np.abs(snap_spec)))[0][0]
            if dds_index == snap_index:
                #print 'LUT shift =', i
		shift = i
                break
            else:
	       shift = -1
        return shift

    def get_transfunc(self):
        print "Calculating transfer function...",
    	mag_array = np.zeros((100, len(self.freq_comb)))
	for i in range(100):
		I, Q = self.read_accum_snap()
		mags = np.sqrt(I**2 + Q**2)
		mag_array[i] = mags[2:len(self.freq_comb)+2]
	mean = np.mean(mag_array, axis = 0)
	transfunc = 1./ (mean / np.max(mean))
	np.save('./last_transfunc.npy', transfunc)
	print "Done"
	return transfunc

    def freqComb(self, freqs, samp_freq, resolution, random_phase = True, DAC_LUT = True, apply_transfunc = False):
    # Generates a frequency comb for the DAC or DDS look-up-tables. DAC_LUT = True for the DAC LUT. Returns I and Q 
        freqs = np.round(freqs/self.dac_freq_res)*self.dac_freq_res
	amp_full_scale = (2**15 - 1)
        if DAC_LUT:
	    fft_len = self.LUTbuffer_len
            k = self.fft_bin_index(freqs, fft_len, samp_freq)
	    if random_phase == True:
	    	np.random.seed()
            	phase = np.random.uniform(0., 2.*np.pi, len(k))
	    else: 
   		phase = -np.pi*self.rudinshapiro(len(self.freq_comb))
		phase[phase == -np.pi] = 0	 
            if apply_transfunc:
	    	print "Applying transfer function to DAC LUTS"
		self.amps = self.get_transfunc()
	    else:
	        self.amps = np.array([1.]*len(k))
                #wn = np.load('noise_tf.npy')
                #self.amps = np.mean(wn)/wn
	    spec = np.zeros(fft_len,dtype='complex')
	    spec[k] = self.amps*np.exp(1j*(phase))
	    wave = np.fft.ifft(spec)
	    waveMax = np.max(np.abs(wave))
	    I = (wave.real/waveMax)*(amp_full_scale)
            Q = (wave.imag/waveMax)*(amp_full_scale)
        else:
            fft_len = (self.LUTbuffer_len/self.fft_len)
            k = self.fft_bin_index(freqs, fft_len, samp_freq)
	    spec = np.zeros(fft_len,dtype='complex')
            amps = np.array([1.]*len(k))
            phase = 0.
	    spec[k] = amps*np.exp(1j*(phase))
            wave = np.fft.ifft(spec)
	    waveMax = np.max(np.abs(wave))
	    I = (wave.real/waveMax)*(amp_full_scale)
            Q = (wave.imag/waveMax)*(amp_full_scale)
	return I, Q    
    
    def fft_bin_index(self, freqs, fft_len, samp_freq):
    # returns the fft bin index for a given frequency, fft length, and sample frequency
        k = np.round((freqs/samp_freq)*fft_len).astype('int')
        return k
    
    def select_bins(self, freqs):
    # Calculates the offset from each bin center, to be used as the DDS LUT frequencies, and writes bin numbers to RAM
        nyquist = 250.0e3
	k = self.fft_bin_index(freqs, self.fft_len, 2*self.fpga_samp_freq)
        f_bin = k*self.dac_samp_freq/self.fft_len
	k[ k < 0 ] += self.fft_len
	freq_residuals = freqs - f_bin
        bin_freqs = np.unique(f_bin)
	#print k
	#for i in range(len(bin_freqs)):
            # print "f_bin =", bin_freqs[i]/1.0e6
            #for j in range(len(freqs)):
            #    if (freqs[j] < bin_freqs[i] + nyquist) and (freqs[j] > bin_freqs[i] - nyquist):
            #        print "\tfreq, \tf_ddc, \tch:", np.round(freqs[j]/1.0e6, 3), np.round(freq_residuals[j],3), j
	ch = 0
        for idx in k:
	    self.fpga.write_int(self.regs[np.where(self.regs == 'bins_reg')[0][0]][1], idx)
            self.fpga.write_int(self.regs[np.where(self.regs == 'load_bins_reg')[0][0]][1], 2*ch + 1)
	    self.fpga.write_int(self.regs[np.where(self.regs == 'load_bins_reg')[0][0]][1], 0)
            ch += 1
        return freq_residuals
    
    def define_DDS_LUT(self, freqs):
    # Builds the DDS look-up-table from I and Q given by freq_comb. freq_comb is called with the sample rate equal to the sample rate for a single FFT bin. There are two bins returned for every fpga clock, so the bin sample rate is 256 MHz / half the fft length  
        freq_residuals = self.select_bins(freqs)
        I_dds, Q_dds = np.array([0.]*(self.LUTbuffer_len)), np.array([0.]*(self.LUTbuffer_len))
        for m in range(len(freq_residuals)):
            I, Q = self.freqComb(np.array([freq_residuals[m]]), self.fpga_samp_freq/(self.fft_len/2.), self.dac_freq_res, random_phase = False, DAC_LUT = False)
            I_dds[m::self.fft_len] = I
            Q_dds[m::self.fft_len] = Q
        return I_dds, Q_dds
    
    def pack_luts(self, freqs, transfunc = False):
    # packs the I and Q look-up-tables into strings of 16-b integers, in preparation to write to the QDR. Returns the string-packed look-up-tables
        if transfunc:
		I_dac, Q_dac = self.freqComb(freqs, self.dac_samp_freq, self.dac_freq_res, random_phase = True, apply_transfunc = True)
        else:
		I_dac, Q_dac = self.freqComb(freqs, self.dac_samp_freq, self.dac_freq_res, random_phase = True)
	I_dds, Q_dds = self.define_DDS_LUT(freqs)
	self.I_dds = I_dds
        I_lut, Q_lut = np.zeros(self.LUTbuffer_len*2), np.zeros(self.LUTbuffer_len*2)
        I_lut[0::4] = I_dac[1::2]         
        I_lut[1::4] = I_dac[0::2]
        I_lut[2::4] = I_dds[1::2]
        I_lut[3::4] = I_dds[0::2]
        Q_lut[0::4] = Q_dac[1::2]         
        Q_lut[1::4] = Q_dac[0::2]
        Q_lut[2::4] = Q_dds[1::2]
        Q_lut[3::4] = Q_dds[0::2]
        I_lut_packed = I_lut.astype('>i2').tostring()
        Q_lut_packed = Q_lut.astype('>i2').tostring()
	return I_lut_packed, Q_lut_packed
        
    def writeQDR(self, freqs, transfunc = False):
    # Writes packed LUTs to QDR
	if transfunc:
		I_lut_packed, Q_lut_packed = self.pack_luts(freqs, transfunc = True)
	else:
		I_lut_packed, Q_lut_packed = self.pack_luts(freqs, transfunc = False)
        self.fpga.write_int(self.regs[np.where(self.regs == 'dac_reset_reg')[0][0]][1],1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'dac_reset_reg')[0][0]][1],0)
        self.fpga.write_int(self.regs[np.where(self.regs == 'start_dac_reg')[0][0]][1],0)
        self.fpga.blindwrite(self.regs[np.where(self.regs == 'qdr0_reg')[0][0]][1],I_lut_packed,0)
        self.fpga.blindwrite(self.regs[np.where(self.regs == 'qdr1_reg')[0][0]][1],Q_lut_packed,0)
        self.fpga.write_int(self.regs[np.where(self.regs == 'start_dac_reg')[0][0]][1],1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'accum_reset_reg')[0][0]][1], 0)
        self.fpga.write_int(self.regs[np.where(self.regs == 'accum_reset_reg')[0][0]][1], 1)
	np.save("last_freq_comb.npy", self.freq_comb)
	self.fpga.write_int(self.regs[np.where(self.regs == 'write_comb_len_reg')[0][0]][1], len(self.freq_comb))
        print 'Done.'
        return 

    def read_accum_snap(self):
	# Reads the avgIQ buffer. Returns I and Q as 32-b signed integers     
	self.fpga.write_int(self.regs[np.where(self.regs == 'accum_snap_ctrl_reg')[0][0]][1], 0)
	self.fpga.write_int(self.regs[np.where(self.regs == 'accum_snap_ctrl_reg')[0][0]][1], 1)
	accum_data = np.fromstring(self.fpga.read(self.regs[np.where(self.regs == 'accum_snap_bram_reg')[0][0]][1], 16*2**9), dtype = '>i').astype('float')
	I = accum_data[0::2]    
	Q = accum_data[1::2]    
	return I, Q    

    def dirfile_all_chan(self, time_interval):
	nchannel = len(self.freq_comb)
	channels = range(nchannel)
	data_path = "./data"
	sub_folder_1 = "meas"
	sub_folder_2 = raw_input("Insert subfolder name (e.g. single_tone): ")
	Npackets = np.int(time_interval * self.accum_freq)
	self.fpga.write_int(self.regs[np.where(self.regs == 'pps_start_reg')[0][0]][1], 1)
        save_path = os.path.join(data_path, sub_folder_1, sub_folder_2)
	if not os.path.exists(save_path):
	    os.makedirs(save_path)
        filename = save_path + '/' + \
                   str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
        # make the dirfile
        d = gd.dirfile(filename,gd.CREAT|gd.RDWR|gd.UNENCODED)
        # add fields
        phase_fields = []
        for chan in range(nchannel):
            phase_fields.append('chP_' + str(chan))
            d.add_spec('chP_' + str(chan) + ' RAW FLOAT64 1')
        d.add_spec('time RAW FLOAT64 1')
        d.add_spec('packet_count RAW UINT32 1')
	d.close()
        d = gd.dirfile(filename,gd.RDWR|gd.UNENCODED)
        #nfo_I = map(lambda x: save_path + "/chI_" + str(x), range(nchannel))
        #nfo_Q = map(lambda y: save_path + "/chQ_" + str(y), range(nchannel))
        nfo_phase = map(lambda z: filename + "/chP_" + str(z), range(nchannel))
	#fo_I = map(lambda x: open(x, "ab"), nfo_I)
	#fo_Q = map(lambda y: open(y, "ab"), nfo_Q)
	fo_phase = map(lambda z: open(z, "ab"), nfo_phase)
        fo_time = open(filename + "/time", "ab")
  	fo_count = open(filename + "/packet_count", "ab")	
	count = 0
	while count < Npackets:
	    ts = time.time()
            packet = self.s.recv(8234) # total number of bytes including 42 byte header
            data = np.fromstring(packet[42:],dtype = '<i').astype('float')
            packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
	    for chan in channels:
	        if (chan % 2) > 0:
                    I = data[1024 + ((chan - 1) / 2)]    
                    Q = data[1536 + ((chan - 1) /2)]    
                else:
                    I = data[0 + (chan/2)]    
                    Q = data[512 + (chan/2)]    
	        #fo_I[chan].write(struct.pack('i',I))
	        #fo_Q[chan].write(struct.pack('i',Q))
	        fo_phase[chan].write(struct.pack('d', np.arctan2([Q],[I])))
	        #fo_I[chan].flush()
	        #fo_Q[chan].flush()
	        fo_phase[chan].flush()
	    count += 1
	    fo_time.write(struct.pack('d', ts))
	    fo_count.write(struct.pack('L',packet_count))
	    fo_time.flush()
	    fo_count.flush()
	for chan in channels:
	    #fo_I[chan].close()
	    #fo_Q[chan].close()
	    fo_phase[chan].close()
	fo_time.close()
	fo_count.close()
        d.close()
        return 

    def plotADC(self):
        # Plots the ADC timestream
        fig = plt.figure(figsize=(10.24,7.68))
        plot1 = fig.add_subplot(211)
        line1, = plot1.plot(np.arange(0,2048), np.zeros(2048), 'r-', linewidth = 2)
        plot1.set_title('I', size = 20)
        plot1.set_ylabel('mV', size = 20)
        plt.xlim(0,1024)
        plt.ylim(-600,600)
        plt.yticks(np.arange(-600, 600, 100))
        plt.grid()
        plot2 = fig.add_subplot(212)
        line2, = plot2.plot(np.arange(0,2048), np.zeros(2048), 'b-', linewidth = 2)
        plot2.set_title('Q', size = 20)
        plot2.set_ylabel('mV', size = 20)
        plt.xlim(0,1024)
        plt.ylim(-600,600)
        plt.yticks(np.arange(-600, 600, 100))
        plt.grid()
        plt.tight_layout()
        plt.show(block = False)
        count = 0
        stop = 1.0e8
        while count < stop:    
            time.sleep(0.1)
            self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_ctrl_reg')[0][0]][1],0)
            time.sleep(0.1)
            self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_ctrl_reg')[0][0]][1],1)
            time.sleep(0.1)
            self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_ctrl_reg')[0][0]][1],0)    
            time.sleep(0.1)
            self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_trig_reg')[0][0]][1],1)    
            time.sleep(0.1)
            self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_trig_reg')[0][0]][1],0)
            time.sleep(0.1)
            adc = (np.fromstring(self.fpga.read(self.regs[np.where(self.regs == 'adc_snap_bram_reg')[0][0]][1],(2**10)*8),dtype='>h')).astype('float')
            adc /= (2**15)
            adc *= 550.
            I = np.hstack(zip(adc[0::4],adc[2::4]))
            Q = np.hstack(zip(adc[1::4],adc[3::4]))
            line1.set_ydata(I)
            line2.set_ydata(Q)
            fig.canvas.draw()
            count += 1
        return

    def rmsVoltageADC(self):
        self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_ctrl_reg')[0][0]][1],0)
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_ctrl_reg')[0][0]][1],1)
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_ctrl_reg')[0][0]][1],0)    
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_trig_reg')[0][0]][1],0)    
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_trig_reg')[0][0]][1],1)    
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'adc_snap_trig_reg')[0][0]][1],0)
        time.sleep(0.1)
        adc = (np.fromstring(self.fpga.read(self.regs[np.where(self.regs == 'adc_snap_bram_reg')[0][0]][1],(2**10)*8),dtype='>h')).astype('float')
        adc /= (2**15 - 1)
        adc *= 550.
        I = np.hstack(zip(adc[0::4],adc[2::4]))
        Q = np.hstack(zip(adc[1::4],adc[3::4]))
        rmsI = np.round(np.sqrt(np.mean(I**2)),2)
        rmsQ = np.round(np.sqrt(np.mean(Q**2)),2)
	peakI = np.abs(np.max(I))
	peakQ = np.abs(np.max(Q))
	crest_factor_I = np.round(20.*np.log10(peakI/rmsI) ,2)
	crest_factor_Q = np.round(20.*np.log10(peakQ/rmsQ), 2)
	return rmsI, rmsQ, crest_factor_I, crest_factor_Q

    def plotAccum(self):
        # Generates a plot stream from read_avgIQ_snap(). To view, run plotAvgIQ.py in a separate terminal
        fig = plt.figure(figsize=(10.24,7.68))
        plt.title('Downsampled |S21|^2, Accum. Frequency = ' + str(self.accum_freq), fontsize=18)
        plot1 = fig.add_subplot(111)
        line1, = plot1.plot(np.arange(1016),np.ones(1016), '#FF4500')
        line1.set_linestyle('None')
        line1.set_marker('.')
        plt.xlabel('Channel #',fontsize = 18)
        plt.ylabel('dB',fontsize = 18)
        plt.xticks(np.arange(0,1016,100))
        plt.xlim(0,1016)
        plt.ylim(-40, 5)
        plt.grid()
        plt.tight_layout()
        plt.show(block = False)
        count = 0 
        stop = 10000
        while(count < stop):
            I, Q = self.read_accum_snap()
            mags =(np.sqrt(I**2 + Q**2))[:1016]
            #mags = np.concatenate((mags[len(mags)/2:],mags[:len(mags)/2]))
            mags = 20*np.log10(mags/np.max(mags))
            line1.set_ydata(mags)
            fig.canvas.draw()
            count += 1
        return
    
    def plotFFT(self):
        fig = plt.figure()
        plot1 = fig.add_subplot(111)
        line1, = plot1.plot(np.arange(0,1024,2), np.zeros(self.fft_len/2), '#FF4500', alpha = 0.8)
        line1.set_marker('.')
        line2, = plot1.plot(np.arange(1,1024,2), np.zeros(self.fft_len/2), 'purple', alpha = 0.8)
        line2.set_marker('.')
        plt.grid()
        plt.ylim(-10, 100)
        plt.tight_layout()
        count = 0 
        stop = 1.0e6
        while(count < stop):
            self.fpga.write_int(self.regs[np.where(self.regs == 'fft_snap_ctrl_reg')[0][0]][1],0)
            self.fpga.write_int(self.regs[np.where(self.regs == 'fft_snap_ctrl_reg')[0][0]][1],1)
	    fft_snap = (np.fromstring(self.fpga.read(self.regs[np.where(self.regs == 'fft_snap_bram_reg')[0][0]][1],(2**9)*8),dtype='>i2')).astype('float')
            I0 = fft_snap[0::4]
            Q0 = fft_snap[1::4]
            I1 = fft_snap[2::4]
            Q1 = fft_snap[3::4]
            mag0 = np.sqrt(I0**2 + Q0**2)
            mag0 = 20*np.log10(mag0)
            mag1 = np.sqrt(I1**2 + Q1**2)
            mags = np.hstack(zip(mag0, mag1))
            mag1 = 20*np.log10(mag1)
            fft_mags = np.hstack(zip(mag0,mag1))
            line1.set_ydata(mag0)
            line2.set_ydata(mag1)
            fig.canvas.draw()
            count += 1
        return

    def plotMixer(self, chan):
        t = np.linspace(0, 16384*2.0e-6, 16384)
        plt.ion()
        fig = plt.figure(figsize=(20,12))
        # I and Q
        plot1 = fig.add_subplot(411)
        #plot1.set_ylabel('mV')
        plt.title('IQ in, Ch ' + str(chan), size = 16)
        line1, = plot1.plot(t, np.zeros(16384), label = 'I in', color = 'red', linewidth = 1)
        line2, = plot1.plot(t, np.zeros(16384), label = 'Q in', color = 'blue', linewidth = 1)
        plt.xlim(0, 0.002)
        plt.ylim((-0.3,0.3))
        plt.grid()
        # DDS I and Q
        plot2 = fig.add_subplot(412)
        #plot2.set_ylabel('mV')
        plt.title('IQ DDC', size = 16)
        line3, = plot2.plot(t, np.zeros(16384), label = 'I ddc', color = 'red', linewidth = 1)
        line4, = plot2.plot(t, np.zeros(16384), label = 'Q ddc', color = 'blue', linewidth = 1)
        plt.xlim(0, 0.002)
        plt.ylim((-3,3))
        plt.grid()
        # Mixer output
        plot3 = fig.add_subplot(413)
        #plot3.set_ylabel('mV')
        plt.title('IQ out', size = 16)
        line5, = plot3.plot(t, np.zeros(16384), label = 'I out', color = 'red', linewidth = 1)
        line6, = plot3.plot(t, np.zeros(16384), label = 'Q out', color = 'blue', linewidth = 1)
        plt.xlim(0, 0.002)
        plt.ylim((-0.3,0.3))
        plt.grid()
        plot4 = fig.add_subplot(414)
        plt.title('FIR out', size = 16)
        line7, = plot4.plot(t, np.zeros(16384), label = 'I fir', color = 'red', linewidth = 1)
        line8, = plot4.plot(t, np.zeros(16384), label = 'Q fir', color = 'blue', linewidth = 1)
        plt.xlim(0,0.002)
        plt.ylim((-0.1,0.1))
        plt.grid()
        plt.tight_layout()
        count = 0
        stop = 10000
        while (count < stop):
            if (chan % 2) > 0:
            	I_in, Q_in, I_dds_in, Q_dds_in, I_out, Q_out, I_lpf, Q_lpf = self.mixer_comp(chan, I0 = False)
            else:
            	I_in, Q_in, I_dds_in, Q_dds_in, I_out, Q_out, I_lpf, Q_lpf = self.mixer_comp(chan)
            line1.set_ydata(I_in/(2**15))
            line2.set_ydata(Q_in/(2**15))
            line3.set_ydata((I_dds_in)/(2**15))
            line4.set_ydata((Q_dds_in)/(2**15))
            line5.set_ydata(I_out/(2**14))
            line6.set_ydata(Q_out/(2**14))
            line7.set_ydata(I_lpf/(2**14))
            line8.set_ydata(Q_lpf/(2**14))
            fig.canvas.draw()
            count += 1
        return


