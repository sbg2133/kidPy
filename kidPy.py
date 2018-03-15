# This software is a work in progress. It is a console interface designed 
# to operate the BLAST-TNG ROACH2 firmware. 
#
# Copyright (C) January, 2018  Gordon, Sam <sbgordo1@asu.edu>
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

import numpy as np
import sys, os
import struct
import casperfpga
import valon_synth9
from socket import *
from roachInterface import roachInterface
from gbeConfig import roachDownlink
import time
import matplotlib.pyplot as plt
from scipy import signal, ndimage, fftpack
import find_kids_interactive as fk
import pygetdata as gd
plt.ion()

################################################################
# Run in IPYTHON as: %run kidPy

# for plotting interface, run as: %run kidPy plot
################################################################

# Load general settings
gc = np.loadtxt("./general_config", dtype = "str")
firmware = gc[np.where(gc == 'FIRMWARE_FILE')[0][0]][1]
vna_savepath = gc[np.where(gc == 'VNA_SAVEPATH')[0][0]][1]
targ_savepath = gc[np.where(gc == 'TARG_SAVEPATH')[0][0]][1]
dirfile_savepath = gc[np.where(gc == 'DIRFILE_SAVEPATH')[0][0]][1]

# Load list of firmware registers (note: must manually update for different versions)
regs = np.loadtxt("./firmware_registers", dtype = "str")

# UDP packet
buf_size = int(gc[np.where(gc == 'buf_size')[0][0]][1])
header_len = int(gc[np.where(gc == 'header_len')[0][0]][1])

# Valon Synthesizer params
CLOCK = int(gc[np.where(gc == 'clock')[0][0]][1])
LO = int(gc[np.where(gc == 'lo')[0][0]][1])
ext_ref = int(gc[np.where(gc == 'ext_ref')[0][0]][1])
lo_step = np.float(gc[np.where(gc == 'lo_step')[0][0]][1])
center_freq = np.float(gc[np.where(gc == 'center_freq')[0][0]][1])

# Optional test frequencies
test_freq = np.float(gc[np.where(gc == 'test_freq')[0][0]][1])
test_freq = np.array([test_freq])
freq_list = gc[np.where(gc == 'freq_list')[0][0]][1]

# Parameters for resonator search
smoothing_scale = np.float(gc[np.where(gc == 'smoothing_scale')[0][0]][1])
peak_threshold = np.float(gc[np.where(gc == 'peak_threshold')[0][0]][1])
spacing_threshold  = np.float(gc[np.where(gc == 'spacing_threshold')[0][0]][1])

def systemInit():
    fpga = getFPGA
    if not fpga:
        print "\nROACH link is down"
        return
    # Valon object
    valon = getValon()
    # Roach PPC object
    fpga = getFPGA()
    # Roach interface 
    ri = roachInterface(fpga, gc, regs, valon)
    if (ri.uploadfpg() < 0):
        print "\nFirmware upload failed"
    time.sleep(0.3)
    # UDP socket
    s = socket(AF_PACKET, SOCK_RAW, htons(3))
    # UDP object
    udp = roachDownlink(ri, fpga, gc, regs, s, ri.accum_freq)
    try:
        initValon(valon)
        print "Valon initiliazed"
    except OSError:
        print '\033[93mValon Synthesizer could not be initialized: Check comm port and power supply\033[93m'
        return
    except IndexError:
        print '\033[93mValon Synthesizer could not be initialized: Check comm port and power supply\033[93m'
        return
    fpga.write_int(regs[np.where(regs == 'accum_len_reg')[0][0]][1], ri.accum_len - 1)
    time.sleep(0.1)
    fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], int(gc[np.where(gc == 'dds_shift')[0][0]][1]))
    time.sleep(0.1)
    #ri.lpf(ri.boxcar)
    if (ri.qdrCal() < 0):
        print '\033[93mQDR calibration failed... Check FPGA clock source\033[93m'
        return
    else:
        fpga.write_int(regs[np.where(regs == 'write_qdr_status_reg')[0][0]][1], 1)
    time.sleep(0.1)
    try:
        udp.configDownlink()
    except AttributeError:
        print "UDP Downlink could not be configured. Check ROACH connection."
        return
    return

def getFPGA():
    """Returns a casperfpga object of the Roach2"""
    try:
        fpga = casperfpga.katcp_fpga.KatcpFpga(gc[np.where(gc == 'roach_ppc_ip')[0][0]][1], timeout = 120.)
    except RuntimeError:
        print "\nNo connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check gc config."
    return fpga

def testConn(fpga):
    """Tests the link to Roach2 PPC, using return from getFPGA()
        inputs:
            casperfpga object fpga: The fpga object
        outputs: the fpga object"""
    if not fpga:
        try:
            fpga = casperfpga.katcp_fpga.KatcpFpga(gc[np.where(gc == 'roach_ppc_ip')[0][0]][1], timeout = 3.)
        except RuntimeError:
            print "\nNo connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check gc config."
    return fpga

def initValon(valon, ref_freq = 10):
    """Configures default parameters for a Valon 5009 Sythesizer
        inputs:
            valon synth object valon: See getValon()
            bool ext_ref: Use external ref?
            int ref_freq: Ext reference freq, MHz"""
    if ext_ref:
        valon.set_reference(ref_freq)
        valon.set_ref_select(1)
    else:
        valon.set_ref_select(0)
    valon.set_refdoubler(CLOCK, 0)
    valon.set_refdoubler(LO, 0)
    valon.set_pfd(CLOCK, 40.)
    valon.set_pfd(LO, 10.)
    valon.set_frequency(LO, center_freq) # LO
    valon.set_frequency(CLOCK, 512.) # Clock
    valon.set_rf_level(CLOCK, 7)
    valon.set_rf_level(LO, 10)
    return

def getValon():
    """Return a valon synthesizer object
       If there's a problem, return None"""
    try:
        valon = valon_synth9.Synthesizer(gc[np.where(gc == 'valon_comm_port')[0][0]][1])
        return valon
    except OSError:
        "Valon could not be initialized. Check comm port and power supply."
    return None

def setValonLevel(valon, chan, dBm):
    """Set the RF power level of a Valon channel
       inputs:
           valon synth object valon: See getValon()
           int chan: LO or CLOCK (see above)
           float dBm: The desired power level in dBm (***calibrate
                      with spectrum analyzer)"""
    valon.set_rf_level(chan, dBm)
    return

def setAtten(inAtten, outAtten):
    """Set the input and output attenuation levels for a RUDAT MCL-30-6000
        inputs:
            float outAtten: The output attenuation in dB
            float inAtten: The input attenuation in dB"""
    command = "sudo ./set_rudats " + str(inAtten) + ' ' + str(outAtten)
    os.system(command)
    return

def readAtten():
    """Read the attenuation levels for both channels of a RUDAT MCL-30-6000
       outputs:
            float outAtten
            float inAtten"""
    os.system("sudo ./read_rudats > rudat.log")
    attens = np.loadtxt('./rudat.log', delimiter = ",")
    inAtten = attens[0][1]
    outAtten = attens[1][1]
    return inAtten, outAtten

# Needs testing
def calibrateADC(target_rms_mv, outAtten, inAtten):
    """Automatically set RUDAT attenuation values to achieve desired ADC rms level
       inputs:
           float target_rms_mv: The target ADC rms voltage level, in mV,
                                for either I or Q channel
           float outAtten: Starting output attenuation, dB
           float inAtten: Starting input attenuation, dB"""
    setAtten(outAtten, inAtten)
    print "Start atten:", outAtten, inAtten
    rmsI, rmsQ, __, __ = ri.rmsVoltageADC()
    avg_rms_0 = (rmsI + rmsQ)/2.
    print "Target RMS:", target_rms_mv, "mV"
    print "Current RMS:", avg_rms_0, "mV"
    if avg_rms_0 < target_rms_mv:
        avg_rms = avg_rms_0
        while avg_rms < target_rms_mv:
            time.sleep(0.1)
            if inAtten > 1:
                inAtten -= 1
            else:
                outAtten -= 1
            if (inAtten == 1) and (outAtten == 1):
                break
            setAtten(outAtten, inAtten)
            rmsI, rmsQ, __, __ = ri.rmsVoltageADC()
            avg_rms = (rmsI + rmsQ)/2.
            outA, inA = readAtten()
            print outA, inA
    if avg_rms_0 > target_rms_mv:
        avg_rms = avg_rms_0
        while avg_rms > target_rms_mv:
            time.sleep(0.1)
            if outAtten < 30:
                outAtten += 1
            else:
                inAtten += 1
            if (inAtten > 30) and (outAtten > 30):
                break
            setAtten(outAtten, inAtten)
            rmsI, rmsQ, __, __ = ri.rmsVoltageADC()
            avg_rms = (rmsI + rmsQ)/2.
            outA, inA = readAtten()
            print outA, inA
    new_out, new_in = readAtten()
    print
    print "Final atten:", new_out, new_in
    print "Current RMS:", avg_rms, "mV"
    return

#######################################################################
# Captions and menu options for terminal interface
caption1 = '\n\t\033[95mKID-PY ROACH2 Readout\033[95m'
caption2 = '\n\t\033[94mThese functions require UDP streaming to be active\033[94m'
captions = [caption1, caption2]
main_opts= ['Test connection to ROACH',\
            'Upload firmware',\
            'Initialize system & UDP conn',\
            'Write test comb (single or multitone)',\
            'Write stored comb',\
            'Apply inverse transfer function',\
            'Calibrate ADC V_rms',\
            'Get system state',\
            'Test GbE downlink',\
            'Print packet info to screen (UDP)',\
            'VNA sweep and plot','Locate freqs from VNA sweep',\
            'Write found freqs',\
            'Target sweep and plot',\
            'Plot channel phase PSD (quick look)',\
            'Save dirfile for range of chan',\
            'Exit']
#########################################################################

def vnaSweep(ri, udp, valon):
    """Does a wideband sweep of the RF band, saves data in vna_savepath
       as .npy files
       inputs:
           roachInterface object ri
           gbeConfig object udp
           valon synth object valon
           bool write: Write test comb before sweeping?
           Navg = Number of data points to average at each sweep step"""
    Navg = np.int(gc[np.where(gc == 'Navg')[0][0]][1])
    if not os.path.exists(vna_savepath):
        os.makedirs(vna_savepath)
    sweep_dir = vna_savepath + '/' + \
       str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
    os.mkdir(sweep_dir)
    np.save("./last_vna_dir.npy", sweep_dir)
    print sweep_dir
    valon.set_frequency(LO, center_freq/1.0e6)
    span = ri.pos_delta
    print "Sweep Span =", 2*np.round(ri.pos_delta,2), "Hz"
    start = center_freq*1.0e6 - (span)
    stop = center_freq*1.0e6 + (span)
    sweep_freqs = np.arange(start, stop, lo_step)
    sweep_freqs = np.round(sweep_freqs/lo_step)*lo_step
    if not np.size(ri.freq_comb):
        ri.makeFreqComb()
    np.save(sweep_dir + '/bb_freqs.npy', ri.freq_comb)
    np.save(sweep_dir + '/sweep_freqs.npy', sweep_freqs)
    Nchan = len(ri.freq_comb)
    if not Nchan:
        Nchan = fpga.read_int(regs[np.where(regs == 'read_comb_len_reg')[0][0]][1])
    for freq in sweep_freqs:
        print 'LO freq =', freq/1.0e6
        valon.set_frequency(LO, freq/1.0e6)
        #print "LO freq =", valon.get_frequency(LO)
        #time.sleep(0.1)
        udp.saveSweepData(Navg, sweep_dir, freq, Nchan)
        #time.sleep(0.1)
    valon.set_frequency(LO, center_freq) # LO
    return

def writeVnaComb(cw = False):                                                             
    # Roach PPC object                                                          
    fpga = getFPGA()                                                            
    if not fpga:                                                                
        print "\nROACH link is down"                                            
        return                                                                  
    # Roach interface                                                           
    ri = roachInterface(fpga, gc, regs, None)                                   
    try:                                                                        
	if cw:
            ri.freq_comb = test_freq    
	else:
	    ri.makeFreqComb()
        if (len(ri.freq_comb) > 400):                                             
            fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)
            time.sleep(0.1)                                                     
        else:                                                                   
            fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)
            time.sleep(0.1)                                                     
        ri.upconvert = np.sort(((ri.freq_comb + (center_freq)*1.0e6))/1.0e6)    
        print "RF tones =", ri.upconvert                                        
        ri.writeQDR(ri.freq_comb, transfunc = False)                            
        np.save("last_freq_comb.npy", ri.freq_comb)                             
        if not (fpga.read_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1])):
            if regs[np.where(regs == 'DDC_mixerout_bram_reg')[0][0]][1] in fpga.listdev():
                shift = ri.return_shift(0)                                      
                if (shift < 0):                                                 
                    print "\nError finding dds shift: Try writing full frequency comb (N = 1000), or single test frequency. Then try again"
                    return                                                       
                else:                                                           
                    fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], shift)
                    print "Wrote DDS shift (" + str(shift) + ")"                
            else:                                                               
                fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], ri.dds_shift)
    except KeyboardInterrupt:                                                   
        return 
    return

def vnaSweepConsole():
    """Does a wideband sweep of the RF band, saves data in vna_savepath
       as .npy files"""
    # UDP socket
    s = socket(AF_PACKET, SOCK_RAW, htons(3))
    # Valon object
    valon = getValon()
    # Roach PPC object
    fpga = getFPGA()
    # Roach interface 
    ri = roachInterface(fpga, gc, regs, valon)
    # UDP object
    udp = roachDownlink(ri, fpga, gc, regs, s, ri.accum_freq)
    udp.configSocket()
    Navg = np.int(gc[np.where(gc == 'Navg')[0][0]][1])
    if not os.path.exists(vna_savepath):
        os.makedirs(vna_savepath)
    sweep_dir = vna_savepath + '/' + \
       str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
    os.mkdir(sweep_dir)
    np.save("./last_vna_dir.npy", sweep_dir)
    print sweep_dir
    valon.set_frequency(LO, center_freq/1.0e6)
    span = ri.pos_delta
    print "Sweep Span =", 2*np.round(ri.pos_delta,2), "Hz"
    start = center_freq*1.0e6 - (span)
    stop = center_freq*1.0e6 + (span)
    sweep_freqs = np.arange(start, stop, lo_step)
    sweep_freqs = np.round(sweep_freqs/lo_step)*lo_step
    if not np.size(ri.freq_comb):
        ri.makeFreqComb()
    np.save(sweep_dir + '/bb_freqs.npy', ri.freq_comb)
    np.save(sweep_dir + '/sweep_freqs.npy', sweep_freqs)
    Nchan = len(ri.freq_comb)
    if not Nchan:
        Nchan = fpga.read_int(regs[np.where(regs == 'read_comb_len_reg')[0][0]][1])
    idx = 0
    while (idx < len(sweep_freqs)):
        print 'LO freq =', sweep_freqs[idx]/1.0e6
        valon.set_frequency(LO, sweep_freqs[idx]/1.0e6)
        time.sleep(0.2)
        #time.sleep(0.1)
        if (udp.saveSweepData(Navg, sweep_dir, sweep_freqs[idx], Nchan) < 0):
            continue
        else:
            idx += 1
        #time.sleep(0.1)
    valon.set_frequency(LO, center_freq) # LO
    return

def targetSweep(ri, udp, valon):
    """Does a sweep centered on the resonances, saves data in targ_savepath
       as .npy files
       inputs:
           roachInterface object ri
           roach UDP object udp
           valon synth object valon
           bool write: Write test comb before sweeping?
           float span: Sweep span, Hz
           Navg = Number of data points to average at each sweep step"""
    span = np.float(gc[np.where(gc == 'targ_span')[0][0]][1])
    Navg = np.float(gc[np.where(gc == 'Navg')[0][0]][1])
    vna_savepath = str(np.load("last_vna_dir.npy"))
    if not os.path.exists(targ_savepath):
        os.makedirs(targ_savepath)
    sweep_dir = targ_savepath + '/' + \
       str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
    os.mkdir(sweep_dir)
    np.save("./last_targ_dir.npy", sweep_dir)
    print sweep_dir
    target_freqs = np.load(vna_savepath + '/bb_targ_freqs.npy')
    np.save(sweep_dir + '/bb_target_freqs.npy', target_freqs)
    start = center_freq*1.0e6 - (span/2.)
    stop = center_freq*1.0e6 + (span/2.) 
    sweep_freqs = np.arange(start, stop, lo_step)
    sweep_freqs = np.round(sweep_freqs/lo_step)*lo_step
    np.save(sweep_dir + '/bb_freqs.npy', target_freqs)
    np.save(sweep_dir + '/sweep_freqs.npy',sweep_freqs)
    for freq in sweep_freqs:
        print 'LO freq =', freq/1.0e6, ' MHz'
        valon.set_frequency(LO, freq/1.0e6)
        #time.sleep(0.1)
        udp.saveSweepData(Navg, sweep_dir, freq, len(target_freqs)) 
        #time.sleep(0.1)
    valon.set_frequency(LO, center_freq)
    return

def openStoredSweep(savepath):
    """Opens sweep data
       inputs:
           char savepath: The absolute path where sweep data is saved
       ouputs:
           numpy array Is: The I values
           numpy array Qs: The Q values"""
    files = sorted(os.listdir(savepath))
    I_list, Q_list = [], []
    for filename in files:
        if filename.startswith('I'):
            I_list.append(os.path.join(savepath, filename))
        if filename.startswith('Q'):
            Q_list.append(os.path.join(savepath, filename))
    Is = np.array([np.load(filename) for filename in I_list])
    Qs = np.array([np.load(filename) for filename in Q_list])
    return Is, Qs

def plotVNASweep(path):
    plt.figure()
    Is, Qs = openStoredSweep(path)
    sweep_freqs = np.load(path + '/sweep_freqs.npy')
    bb_freqs = np.load(path + '/bb_freqs.npy')
    rf_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))
    for chan in range(len(bb_freqs)):
        rf_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    Q = np.reshape(np.transpose(Qs),(len(Qs[0])*len(sweep_freqs)))
    I = np.reshape(np.transpose(Is),(len(Is[0])*len(sweep_freqs)))
    mag = np.sqrt(I**2 + Q**2)
    mag = 20*np.log10(mag/np.max(mag))
    mag = np.concatenate((mag[len(mag)/2:],mag[:len(mag)/2]))
    rf_freqs = np.hstack(rf_freqs)
    rf_freqs = np.concatenate((rf_freqs[len(rf_freqs)/2:],rf_freqs[:len(rf_freqs)/2]))
    plt.plot(rf_freqs, mag)
    plt.title(path, size = 16)
    plt.xlabel('frequency (MHz)', size = 16)
    plt.ylabel('dB', size = 16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(path,'vna_sweep.png'), dpi = 100, bbox_inches = 'tight')
    return
"""
def plotVNASweep(path):
    plt.figure()
    Is, Qs = openStoredSweep(path)
    sweep_freqs = np.load(path + '/sweep_freqs.npy')
    bb_freqs = np.load(path + '/bb_freqs.npy')
    rf_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))
    for chan in range(len(bb_freqs)):
        rf_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    Q = np.reshape(np.transpose(Qs),(len(Qs[0])*len(sweep_freqs)))
    I = np.reshape(np.transpose(Is),(len(Is[0])*len(sweep_freqs)))
    mag = np.sqrt(I**2 + Q**2)
    mag = 20*np.log10(mag/np.max(mag))
    mag = np.hstack(mag)
    rf_freqs = np.hstack(rf_freqs)
    plt.plot(rf_freqs, mag)
    plt.title(path, size = 16)
    plt.xlabel('frequency (MHz)', size = 16)
    plt.ylabel('dB', size = 16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(path,'vna_sweep.png'), dpi = 100, bbox_inches = 'tight')
    return
"""
def plotTargSweep(path):
    """Plots the results of a TARG sweep
       inputs:
           path: Absolute path to where sweep data is saved"""
    plt.figure()
    Is, Qs = openStoredSweep(path)
    sweep_freqs = np.load(path + '/sweep_freqs.npy')
    bb_freqs = np.load(path + '/bb_freqs.npy')
    channels = len(bb_freqs)
    mags = np.zeros((channels, len(sweep_freqs))) 
    chan_freqs = np.zeros((channels, len(sweep_freqs)))
    new_targs = np.zeros((channels))
    for chan in range(channels):
        mags[chan] = np.sqrt(Is[:,chan]**2 + Qs[:,chan]**2)
        mags[chan] = 20*np.log10(mags[chan]/np.max(mags[chan]))
        chan_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    mags = np.concatenate((mags[len(mags)/2:],mags[:len(mags)/2]))
    #bb_freqs = np.concatenate(bb_freqs[len(b_freqs)/2:],bb_freqs[:len(bb_freqs)/2]))
    chan_freqs = np.concatenate((chan_freqs[len(chan_freqs)/2:],chan_freqs[:len(chan_freqs)/2]))
    new_targs = [chan_freqs[chan][np.argmin(mags[chan])] for chan in range(channels)]
    for chan in range(channels):
        plt.plot(chan_freqs[chan],mags[chan])
    plt.title(path, size = 16)
    plt.xlabel('frequency (MHz)', size = 16)
    plt.ylabel('dB', size = 16)
    plt.tight_layout()
    plt.savefig(os.path.join(path,'targ_sweep.png'), dpi = 100, bbox_inches = 'tight')
    return

def plotLastVNASweep():
    plotVNASweep(str(np.load('last_vna_dir.npy')))
    return

def plotLastTargSweep():
    plotTargSweep(str(np.load('last_targ_dir.npy')))
    return

def saveTimestreamDirfile(subfolder, start_chan, end_chan, time_interval):
    """Saves a dirfile containing the I and Q values for a range of channels, streamed
       over a time interval specified by time_interval
       inputs:
           float time_interval: Time interval to integrate over, seconds"""
    # Roach PPC object
    fpga = getFPGA()
    # UDP socket
    s = socket(AF_PACKET, SOCK_RAW, htons(3))
    # Roach interface 
    ri = roachInterface(fpga, gc, regs, None)
    # UDP object
    udp = roachDownlink(ri, fpga, gc, regs, s, ri.accum_freq)
    udp.configSocket()
    chan_range = range(start_chan, end_chan + 1)
    data_path = gc[np.where(gc == 'DIRFILE_SAVEPATH')[0][0]][1] 
    Npackets = int(np.ceil(time_interval * ri.accum_freq))
    udp.zeroPPS()
    save_path = os.path.join(data_path, subfolder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = save_path + '/' + \
               str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
    print filename
    np.save('last_data_path.npy', filename)
    # make the dirfile
    d = gd.dirfile(filename,gd.CREAT|gd.RDWR|gd.UNENCODED)
    # add fields
    I_fields = []
    Q_fields = []
    for chan in chan_range:
        I_fields.append('I_' + str(chan))
        Q_fields.append('Q_' + str(chan))
        d.add_spec('I_' + str(chan) + ' RAW FLOAT64 1')
        d.add_spec('Q_' + str(chan) + ' RAW FLOAT64 1')
    d.close()
    d = gd.dirfile(filename,gd.RDWR|gd.UNENCODED)
    nfo_I = map(lambda z: filename + "/I_" + str(z), chan_range)
    nfo_Q = map(lambda z: filename + "/Q_" + str(z), chan_range)
    fo_I = map(lambda z: open(z, "ab"), nfo_I)
    fo_Q = map(lambda z: open(z, "ab"), nfo_Q)
    fo_time = open(filename + "/time", "ab")
    fo_count = open(filename + "/packet_count", "ab")
    count = 0
    while count < Npackets:
        ts = time.time()
        try:
            packet, data, header, saddr = udp.parsePacketData()
            if not packet:
                continue
        except TypeError:
            continue
        packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
        idx = 0
        for chan in range(start_chan, end_chan + 1):
            I, Q, __ = udp.parseChanData(chan, data)
            fo_I[idx].write(struct.pack('d', I))
            fo_Q[idx].write(struct.pack('d', Q))
            fo_I[idx].flush()
            fo_Q[idx].flush()
            idx += 1
        fo_count.write(struct.pack('L',packet_count))
        fo_count.flush()
        fo_time.write(struct.pack('d', ts))
        fo_time.flush()
        count += 1
    for idx in range(len(fo_I)):
         fo_I[idx].close()
         fo_Q[idx].close()
    fo_time.close()
    fo_count.close()
    d.close()
    return

def getSystemState(fpga, ri, udp, valon):
    """Displays current firmware configuration
       inputs:
           casperfpga object fpga
           roachInterface object ri
           gbeConfig object udp
           valon synth object valon"""
    print
    print "Current system state:"
    print "DDS shift:", fpga.read_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1])
    print "FFT shift:", fpga.read_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1])
    print "Number of tones:", fpga.read_int(regs[np.where(regs == 'read_comb_len_reg')[0][0]][1])
    print "QDR Cal status:", fpga.read_int(regs[np.where(regs == 'read_qdr_status_reg')[0][0]][1])
    print
    print "Data downlink:"
    print "Stream status: ", fpga.read_int(regs[np.where(regs == 'read_stream_status_reg')[0][0]][1])
    print "Data rate: ", ri.accum_freq, "Hz", ", " + str(np.round(buf_size * ri.accum_freq / 1.0e6, 2)) + " MB/s"
    #print "UDP source IP,port:", inet_ntoa(struct.pack(">i", fpga.read_int(regs[np.where(regs == 'udp_srcip_reg')[0][0]][1]))),":", fpga.read_int(regs[np.where(regs == 'udp_srcport_reg')[0][0]][1]) 
    #print "UDP dest IP,port:", inet_ntoa(struct.pack(">i", fpga.read_int(regs[np.where(regs == 'udp_destip_reg')[0][0]][1]))),":", fpga.read_int(regs[np.where(regs == 'udp_destport_reg')[0][0]][1])
    print
    print "ADC and attenuator levels:"
    inAtten, outAtten = readAtten()
    rmsI, rmsQ, crest_factor_I, crest_factor_Q = ri.rmsVoltageADC()
    print "out atten:", outAtten, "dB"
    print "in atten:", inAtten, "dB"
    print "ADC V_rms (I,Q):", rmsI, "mV", rmsQ, "mV"
    print "Crest factor (I,Q):", crest_factor_I, "dB", crest_factor_Q, "dB"
    print
    print "Valon state:"
    #print "Reference:"
    print "LO center freq:", center_freq, "MHz"
    print "Current LO freq:", valon.get_frequency(LO), "MHz"
    print "Current LO power:", valon.get_rf_level(LO), "dBm"
    print "Current clock freq:", valon.get_frequency(CLOCK), "MHz"
    print "Current clock power:", valon.get_rf_level(CLOCK), "dBm"
    return

def plotPhasePSD(chan, udp, ri, time_interval):
    """Plots a channel phase noise power spectral density using Welch's method
       inputs:
           int chan: Detector channel
           gbeConfig object udp
           roachInterface object ri
           float time_interval: The integration time interval, seconds"""
    plt.ion()
    I, Q, phases = udp.streamChanPhase(chan, time_interval)
    f, Sii = signal.welch(I, ri.accum_freq, nperseg=len(I)/4)
    f, Sqq = signal.welch(Q, ri.accum_freq, nperseg=len(Q)/4)
    f, Spp = signal.welch(phases, ri.accum_freq, nperseg=len(phases)/4)
    Spp = 10*np.log10(Spp[1:]) 
    Sii = 10*np.log10(Sii[1:]) 
    Sqq = 10*np.log10(Sqq[1:]) 
    #plt.figure(figsize = (10.24, 7.68))
    #plt.title(r' $S_{\phi \phi}$', size = 18)
    plt.suptitle('Chan ' + str(chan))
    plt.subplot(3,1,1)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylabel('dBc/Hz', size = 16)
    #ax.set_xlabel('log Hz', size = 16)
    ax.set_ylim((np.min(Sii) - 10, np.max(Sii) + 10))
    ax.plot(f[1:], Sii, linewidth = 1, label = 'I', alpha = 0.7)
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.subplot(3,1,2)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylabel('dBc/Hz', size = 16)
    #ax.set_xlabel('log Hz', size = 16)
    ax.set_ylim((np.min(Sqq) - 10, np.max(Sqq) + 10))
    ax.plot(f[1:], Sqq, linewidth = 1, label = 'Q', alpha = 0.7)
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.subplot(3,1,3)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylabel('dBc/Hz', size = 16)
    ax.set_xlabel('log Hz', size = 16)
    ax.set_ylim((np.min(Spp) - 10, np.max(Spp) + 10))
    ax.plot(f[1:], Spp, linewidth = 1, label = 'Phase', alpha = 0.7)
    plt.legend(loc = 'upper right')
    plt.grid()
    return

def plotAllPSD(dirfile): 
    if dirfile == None:
        dirfile = str(np.load('last_data_path.npy'))
    firstframe = 0
    firstsample = 0
    d = gd.dirfile(dirfile, gd.RDWR|gd.UNENCODED)
    print "Number of frames in dirfile =", d.nframes
    nframes = d.nframes
    vectors = d.field_list()
    ifiles = [i for i in vectors if i[0] == "I"]
    qfiles = [q for q in vectors if q[0] == "Q"]
    ifiles.remove("INDEX")
    wn = []
    plt.figure()
    plt.title(r' $S_{\phi \phi}$', size = 16)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylabel('dBc/Hz', size = 16)
    ax.set_xlabel('log Hz', size = 16)
    for n in range(len(ifiles)):
        ivals = d.getdata(ifiles[n], gd.FLOAT32, first_frame = firstframe, first_sample = firstsample, num_frames = nframes)
        qvals = d.getdata(qfiles[n], gd.FLOAT32, first_frame = firstframe, first_sample = firstsample, num_frames = nframes)
        ivals = ivals[~np.isnan(ivals)]
        Qvals = qvals[~np.isnan(qvals)]
        f, Spp = signal.welch(np.arctan2(qvals,ivals), 488.28125)
        Spp = Spp[Spp != 0.]
        if not np.size(Spp):
            mean_wn = np.nan
            pass
        else:
            Spp = 10*np.log10(Spp) 
        mean_wn = np.mean(Spp[3*len(Spp)/4:])
    	ax.plot(f, Spp, linewidth = 1)
        wn.append(mean_wn)
    plt.grid()
    plt.tight_layout()
    d.close()
    wn = np.array(wn)
    plt.figure()
    plt.plot(wn)
    plt.scatter(range(len(wn)), wn)
    plt.xlabel('Chan', size = 18)
    plt.ylabel('dBc/Hz', size = 18)
    plt.grid()
    plt.tight_layout()
    return

def filter_trace(path, bb_freqs, sweep_freqs):
    """Loads RF frequencies and magnitudes from TARG sweep data
       inputs:
           char path: Absolute path to sweep data
           bb_freqs: Array of baseband frequencies used during sweep
           sweep_freqs: Array of LO frequencies used during sweep
       outputs:
           array chan_freqs: Array of RF frequencies covered by each channel
           array mags: Magnitudes, in dB, of each channel sweep"""
    chan_I, chan_Q = openStoredSweep(path)
    channels = np.arange(np.shape(chan_I)[1])
    mag = np.zeros((len(bb_freqs),len(sweep_freqs)))
    chan_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))
    for chan in channels:
        mag[chan] = (np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2))
        chan_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    mags = 20*np.log10(mag/np.max(mag))
    mags = np.hstack(mags)
    chan_freqs = np.hstack(chan_freqs)
    return chan_freqs, mags

def lowpass_cosine(y, tau, f_3db, width, padd_data=True):
    """Applies a raised cosine low-pass filter to the sweep data
       ***Code/inner comments provided by Sean Bryan***
       inputs:
           float y: array of input data to operate on
           float tau: frequency step size of sweep, 
           f_3db: 1/smoothing scale (3 dB cutoff) specified in general config
           width: Scaling factor for f_3dB
           bool padd_data: See inner comment below
       outputs:
           filtered: filtered sweep data"""
    # padd_data = True means we are going to symmetric copies of the data to the start and stop
    # to reduce/eliminate the discontinuities at the start and stop of a dataset due to filtering
    # False means we're going to have transients at the start and stop of the data
    # kill the last data point if y has an odd length
    if np.mod(len(y),2):
        y = y[0:-1]
    # add the weird padd
    # so, make a backwards copy of the data, then the data, then another backwards copy of the data
    if padd_data:
        y = np.append( np.append(np.flipud(y),y) , np.flipud(y) )
    # take the FFT
    ffty = fftpack.fft(y)
    ffty = fftpack.fftshift(ffty)
    # make the companion frequency array
    delta = 1.0/(len(y)*tau)
    nyquist = 1.0/(2.0*tau)
    freq = np.arange(-nyquist,nyquist,delta)
    # turn this into a positive frequency array
    pos_freq = freq[(len(ffty)/2):]
    # make the transfer function for the first half of the data
    i_f_3db = min( np.where(pos_freq >= f_3db)[0] )
    f_min = f_3db - (width/2.0)
    i_f_min = min( np.where(pos_freq >= f_min)[0] )
    f_max = f_3db + (width/2);
    i_f_max = min( np.where(pos_freq >= f_max)[0] )
    transfer_function = np.zeros(len(y)/2)
    transfer_function[0:i_f_min] = 1
    transfer_function[i_f_min:i_f_max] = (1 + np.sin(-np.pi * ((freq[i_f_min:i_f_max] - freq[i_f_3db])/width)))/2.0
    transfer_function[i_f_max:(len(freq)/2)] = 0
    # symmetrize this to be [0 0 0 ... .8 .9 1 1 1 1 1 1 1 1 .9 .8 ... 0 0 0] to match the FFT
    transfer_function = np.append(np.flipud(transfer_function),transfer_function)
    # apply the filter, undo the fft shift, and invert the fft
    filtered=np.real(fftpack.ifft(fftpack.ifftshift(ffty*transfer_function)))
    # remove the padd, if we applied it
    if padd_data:
        filtered = filtered[(len(y)/3):(2*(len(y)/3))]
    # return the filtered data
    return filtered

def findFreqs(path, plot = False):
    """Open target sweep data stored at path and identify resonant frequencies
       inputs:
           char path: Absolute path to sweep data
           bool plot: Option to plot results"""
    bb_freqs = np.load(path + '/bb_freqs.npy')
    sweep_freqs = np.load(path + '/sweep_freqs.npy')
    chan_freqs, mags = filter_trace(path, bb_freqs, sweep_freqs)
    chan_freqs *= 1.0e6
    filtermags = lowpass_cosine(mags, lo_step, 1./smoothing_scale, 0.1 * (1.0/smoothing_scale))
    ilo = np.where((mags-filtermags) < -1.0*peak_threshold)[0]
    iup = np.where( (mags-filtermags) > -1.0*peak_threshold)[0]
    new_mags = mags - filtermags
    new_mags[iup] = 0
    labeled_image, num_objects = ndimage.label(new_mags)
    indices = ndimage.measurements.minimum_position(new_mags,labeled_image,np.arange(num_objects)+1)
    kid_idx = np.array(indices, dtype = 'int')
    del_idx = []
    for i in range(len(kid_idx) - 1):
        spacing = (chan_freqs[kid_idx[i + 1]] - chan_freqs[kid_idx[i]])
        if (spacing < spacing_threshold):
            if (new_mags[kid_idx[i + 1]] < new_mags[kid_idx[i]]):
                del_idx.append(i)
            else:
                del_idx.append(i + 1)

    del_idx = np.array(del_idx)
    kid_idx = np.delete(kid_idx, del_idx)

    del_again = []
    for i in range(len(kid_idx) - 1):
        spacing = (chan_freqs[kid_idx[i + 1]] - chan_freqs[kid_idx[i]])
        if (spacing < spacing_threshold):
            if (new_mags[kid_idx[i + 1]] < new_mags[kid_idx[i]]):
                del_again.append(i)
            else:
                del_again.append(i + 1)

    del_again = np.array(del_again)
    kid_idx = np.delete(kid_idx, del_again)
    # list of kid frequencies
    rf_target_freqs = np.array(chan_freqs[kid_idx])
    bb_target_freqs = ((rf_target_freqs*1.0e6) - center_freq)

    if len(bb_target_freqs) > 0:
        bb_target_freqs = np.roll(bb_target_freqs, - np.argmin(np.abs(bb_target_freqs)) - 1)
        np.save(path + '/bb_targ_freqs.npy', bb_target_freqs)
        print len(rf_target_freqs), "KIDs found:\n"
        print rf_target_freqs
    else:
        print "No freqs found..."

    if plot:
        plt.figure(1)
        plt.plot(chan_freqs, mags,'b', label = 'no filter')
        plt.plot(chan_freqs, filtermags,'g', label = 'filtered')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('dB')
        plt.legend()
        plt.figure(2)
        plt.plot(chan_freqs, mags - filtermags, 'b')
        plt.plot(chan_freqs[ilo],mags[ilo]-filtermags[ilo],'r*')
        plt.figure(4)
        plt.plot(chan_freqs, mags, 'b')
        plt.plot(chan_freqs[kid_idx], mags[kid_idx], 'r*')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('dB')
    return

def menu(captions, options):
    """Creates menu for terminal interface
       inputs:
           list captions: List of menu captions
           list options: List of menu options
       outputs:
           int opt: Integer corresponding to menu option chosen by user"""
    print '\t' + captions[0] + '\n'
    for i in range(len(options)):
        if (i < 9):
            print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
    print '\t' + captions[1] + '\n'
    for i in range(len(options)):
        if (i >= 9):
            print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
    opt = input()
    return opt

def main_opt(fpga, ri, udp, valon, upload_status):
    """Creates terminal interface
       inputs:
           casperfpga object fpga
           roachInterface object ri
           gbeConfig object udp
           valon synth object valon
           int upload_status: Integer indicating whether or not firmware is uploaded
        outputs:
          int  upload_status"""
    while 1:
        if not fpga:
            print '\n\t\033[93mROACH link is down: Check PPC IP & Network Config\033[93m'
        else:
            print '\n\t\033[92mROACH link is up\033[92m'
        if not upload_status:
            print '\n\t\033[93mNo firmware onboard. If ROACH link is up, try upload option\033[93m'
        else:
            print '\n\t\033[92mFirmware uploaded\033[92m'
        opt = menu(captions,main_opts)
        if opt == 0:
            result = testConn(fpga)
            if not result:
                break
            else:
                fpga = result
                print "\n Connection is up"
        if opt == 1:
            if not fpga:
                print "\nROACH link is down"
                break
            if (ri.uploadfpg() < 0):
                print "\nFirmware upload failed"
            else:
                upload_status = 1
        if opt == 2:
            if not fpga:
                print "\nROACH link is down"
                break
            os.system('clear')
            try:
                initValon(valon)
                print "Valon initiliazed"
            except OSError:
                print '\033[93mValon Synthesizer could not be initialized: Check comm port and power supply\033[93m'
                break
            except IndexError:
                print '\033[93mValon Synthesizer could not be initialized: Check comm port and power supply\033[93m'
                break
            fpga.write_int(regs[np.where(regs == 'accum_len_reg')[0][0]][1], ri.accum_len - 1)
            time.sleep(0.1)
            fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], int(gc[np.where(gc == 'dds_shift')[0][0]][1]))
            time.sleep(0.1)
            #ri.lpf(ri.boxcar)
            if (ri.qdrCal() < 0):
                print '\033[93mQDR calibration failed... Check FPGA clock source\033[93m'
                break
            else:
                fpga.write_int(regs[np.where(regs == 'write_qdr_status_reg')[0][0]][1], 1)
            time.sleep(0.1)
            try:
                udp.configDownlink()
            except AttributeError:
                print "UDP Downlink could not be configured. Check ROACH connection."
                break
        if opt == 3:
             prompt = raw_input('Full test comb? y/n ')
             if prompt == 'y':
		 writeVnaComb()
             else:
                 writeVnaComb(cw = True)
        if opt == 4:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                freq_comb = np.load(freq_list)
                freq_comb = freq_comb[freq_comb != 0]
                freq_comb = np.roll(freq_comb, - np.argmin(np.abs(freq_comb)) - 1)
                ri.freq_comb = freq_comb
                ri.upconvert = np.sort(((ri.freq_comb + (ri.center_freq)*1.0e6))/1.0e6)
                print "RF tones =", ri.upconvert
                if len(ri.freq_comb) > 400:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)
                    time.sleep(0.1)
                else:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)
                    time.sleep(0.1)
                ri.writeQDR(ri.freq_comb)
                #setAtten(27, 17)
                np.save("last_freq_comb.npy", ri.freq_comb)
            except KeyboardInterrupt:
                pass
        if opt == 5:
            if not fpga:
                print "\nROACH link is down"
                break
            if not np.size(ri.freq_comb):
                try:
                    ri.freq_comb = np.load("last_freq_comb.npy")
                except IOError:
                   print "\nFirst need to write a frequency comb with length > 1"
                   break
            try:
                ri.writeQDR(ri.freq_comb, transfunc = True)
                fpga.write_int(regs[np.where(regs == 'write_comb_len_reg')[0][0]][1], len(ri.freq_comb))
            except ValueError:
                print "\nClose Accumulator snap plot before calculating transfer function"
        if opt == 6:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                calibrateADC(83., 20, 20)
            except KeyboardInterrupt:
                pass
        if opt == 7:
            if not fpga:
                print "\nROACH link is down"
                break
            getSystemState(fpga, ri, udp, valon)
        if opt == 8:
            if not fpga:
                print "\nROACH link is down"
                break
            if (udp.testDownlink(5) < 0):
                print "Error receiving data. Check ethernet configuration."
            else:
                print "OK"
                fpga.write_int(regs[np.where(regs == 'write_stream_status_reg')[0][0]][1], 1)
        if opt == 9:
            if not fpga:
                print "\nROACH link is down"
                break
            time_interval = input('\nNumber of seconds to stream? ' )
            chan = input('chan = ? ')
            try:
                udp.printChanInfo(chan, time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 10:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                vnaSweep(ri, udp, valon)
                plotVNASweep(str(np.load("last_vna_dir.npy")))
            except KeyboardInterrupt:
                pass
        if opt == 11:
            try:
                path = str(np.load("last_vna_dir.npy"))
                print "Sweep path:", path
                fk.main(path, center_freq, lo_step, smoothing_scale, peak_threshold, spacing_threshold)
                #findFreqs(str(np.load("last_vna_dir.npy")), plot = True)
            except KeyboardInterrupt:
                break
        if opt == 12:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                freq_comb = np.load(os.path.join(str(np.load('last_vna_dir.npy')), 'bb_targ_freqs.npy'))
                freq_comb = freq_comb[freq_comb != 0]
                freq_comb = np.roll(freq_comb, - np.argmin(np.abs(freq_comb)) - 1)
                ri.freq_comb = freq_comb
                print ri.freq_comb
                #ri.upconvert = np.sort(((ri.freq_comb + (center_freq)*1.0e6))/1.0e6)
                #print "RF tones =", ri.upconvert
                if len(ri.freq_comb) > 400:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**5 -1)    
                    time.sleep(0.1)
                else:
                    fpga.write_int(regs[np.where(regs == 'fft_shift_reg')[0][0]][1], 2**9 -1)    
                    time.sleep(0.1)
                ri.writeQDR(ri.freq_comb)
                #setAtten(27, 17)
                np.save("last_freq_comb.npy", ri.freq_comb)
            except KeyboardInterrupt:
                pass
        if opt == 13:
            if not fpga:
                print "\nROACH link is down"
                break
            try:
                targetSweep(ri, udp, valon)
                plotTargSweep(str(np.load("last_targ_dir.npy")))
            except KeyboardInterrupt:
                pass
        if opt == 14:
            if not fpga:
                print "\nROACH link is down"
                break
            chan = input('Channel number = ? ')
            time_interval = input('Time interval (s) ? ')
            try:
                plotPhasePSD(chan, udp, ri, time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 15:
            if not fpga:
                print "\nROACH link is down"
                break
            time_interval = input('Time interval (s) ? ')
            try:
                #udp.saveDirfile_chanRange(time_interval)
                udp.saveDirfile_chanRangeIQ(time_interval)
                #udp.saveDirfile_adcIQ(time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 16:
            sys.exit()
        return upload_status

############################################################################
# Interface for snap block plotting
plot_caption = '\n\t\033[95mKID-PY ROACH2 Snap Plots\033[95m'
plot_opts= ['I & Q ADC input',\
            'Firmware FFT',\
            'Digital Down Converter Time Domain',\
            'Downsampled Channel Magnitudes']
#############################################################################

def makePlotMenu(prompt,options):
    """Menu for plotting interface
       inputs:
           char prompt: a menu caption
           list options: List of menu options
       outputs:
           int opt: Integer corresponding to chosen option"""
    print '\t' + prompt + '\n'
    for i in range(len(options)):
        print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
    opt = input()
    return opt

def plot_opt(ri):
    """Creates terminal interface for plotting snap blocks
       inputs:
           roachInterface object ri"""
    while 1:
        opt = makePlotMenu(plot_caption, plot_opts)
        if opt == 0:
            try:
                ri.plotADC()
            except KeyboardInterrupt:
                #fig = plt.gcf()
                #plt.close(fig)
                pass
        if opt == 1:
            try:
                ri.plotFFT()
            except KeyboardInterrupt:
                fig = plt.gcf()
                plt.close(fig)
        if opt == 2:
            chan = input('Channel = ? ')
            try:
                ri.plotMixer(chan, fir = False)
            except KeyboardInterrupt:
                fig = plt.gcf()
                plt.close(fig)
        if opt == 3:
            try:
                ri.plotAccum()
            except KeyboardInterrupt:
                fig = plt.gcf()
                plt.close(fig)
    return

def main():
    s = None
    try:
        fpga = casperfpga.katcp_fpga.KatcpFpga(gc[np.where(gc == 'roach_ppc_ip')[0][0]][1], timeout = 120.)
    except RuntimeError:
        fpga = None
    # UDP socket
    s = socket(AF_PACKET, SOCK_RAW, htons(3))

    # Valon synthesizer instance
    try:
        valon = valon_synth9.Synthesizer(gc[np.where(gc == 'valon_comm_port')[0][0]][1])
    except OSError:
        "Valon could not be initialized. Check comm port and power supply."

    # Roach interface
    ri = roachInterface(fpga, gc, regs, valon)

    # GbE interface
    udp = roachDownlink(ri, fpga, gc, regs, s, ri.accum_freq)
    udp.configSocket()
    os.system('clear')
    while 1:
        try:
            upload_status = 0
            if fpga:
                if fpga.is_running():
                    #firmware_info = fpga.get_config_file_info()
                    upload_status = 1
            time.sleep(0.1)
            upload_status = main_opt(fpga, ri, udp, valon, upload_status)
        except TypeError:
            pass
    return 

def plot_main():
    try:
        fpga = casperfpga.katcp_fpga.KatcpFpga(gc[np.where(gc == 'roach_ppc_ip')[0][0]][1], timeout = 3.)
    except RuntimeError:
        fpga = None
    # Roach interface
    ri = roachInterface(fpga, gc, regs, None)
    while 1:
        plot_opt(ri)
    return 

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_main()
    else:
        main()
