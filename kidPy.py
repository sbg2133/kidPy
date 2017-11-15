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
from sean_psd import amplitude_and_power_spectrum as sean_psd
from scipy import signal, ndimage, fftpack
plt.ion()

# load general settings
gc = np.loadtxt("./general_config", dtype = "str")
firmware = gc[np.where(gc == 'FIRMWARE_FILE')[0][0]][1]
vna_savepath = gc[np.where(gc == 'VNA_SAVEPATH')[0][0]][1] 
targ_savepath = gc[np.where(gc == 'TARG_SAVEPATH')[0][0]][1] 
dirfile_savepath = gc[np.where(gc == 'DIRFILE_SAVEPATH')[0][0]][1] 

# load list of firmware registers (note: must manually update for different versions)
regs = np.loadtxt("./firmware_registers", dtype = "str")

# load list of network parameters
network = np.loadtxt("./network_config", dtype = "str")

buf_size = int(network[np.where(network == 'buf_size')[0][0]][1])
header_len = int(network[np.where(network == 'header_len')[0][0]][1])

# Valon channels
CLOCK = 1
LO = 2

lo_step = np.float(gc[np.where(gc == 'lo_step')[0][0]][1])
center_freq = np.float(gc[np.where(gc == 'center_freq')[0][0]][1])
test_freq = np.float(gc[np.where(gc == 'test_freq')[0][0]][1])
test_freq = np.array([test_freq])

# parameters for freq search
smoothing_scale = np.float(gc[np.where(gc == 'smoothing_scale')[0][0]][1])
peak_threshold = np.float(gc[np.where(gc == 'peak_threshold')[0][0]][1])
spacing_threshold  = np.float(gc[np.where(gc == 'spacing_threshold')[0][0]][1])

def testConn(fpga):
    if not fpga:
        try:
	    fpga = casperfpga.katcp_fpga.KatcpFpga(network[np.where(network == 'roach_ppc_ip')[0][0]][1], timeout = 120.)
        except RuntimeError:
	    print "\nNo connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check network config."
    return fpga

# Initialize Valon settings
def initValon(valon, ext_ref = False, ref_freq = 10):
    if ext_ref:
        valon.set_reference(ref_freq)
        valon.set_ref_select(1)
    valon.set_refdoubler(CLOCK, 0)
    valon.set_refdoubler(LO, 0)
    valon.set_pfd(CLOCK, 40.)
    valon.set_pfd(LO, 40.)
    valon.set_frequency(LO, center_freq) # LO
    valon.set_frequency(CLOCK, 512.) # Clock
    valon.set_rf_level(CLOCK, 6)
    valon.set_rf_level(LO, 7)
    return

# For setting MCL RUDAT attenuators
def setAtten(outAtten, inAtten):
    """
    Set input and outputs attenuators (RUDAT MCL-30-6000)
    """
    command = "sudo ./set_rudats " + str(outAtten) + ' ' + str(inAtten)
    os.system(command)
    return

# For reading attenuators
def readAtten():
    os.system("sudo ./read_rudats > rudat.log")
    attens = np.loadtxt('./rudat.log', delimiter = ",")
    outAtten = attens[0][1]
    inAtten = attens[1][1]
    return outAtten, inAtten

# calibrate ADC input level, in millivolts
def calibrateADC(target_rms_mv, outAtten, inAtten):
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

caption1 = '\n\t\033[95mKID-PY ROACH2 Readout\033[95m'
caption2 = '\n\t\033[94mThese functions require UDP streaming to be active\033[94m'
captions = [caption1, caption2]
main_opts= ['Test connection to ROACH', 'Upload firmware', 'Initialize system & UDP conn','Write test comb (single or multitone)', 'Write stored comb', 'Apply inverse transfer function', 'Calibrate ADC V_rms', 'Get system state','Test GbE downlink', 'Print packet info to screen (UDP)','VNA sweep and plot','Locate resonances','Target sweep and plot', 'Plot channel phase PSD (quick look)', 'Save dirfile for range of chan (phase)','Exit'] 

def vnaSweep(ri, udp, valon, write = False, Navg = 50):
    if not os.path.exists(vna_savepath):
        os.makedirs(vna_savepath)
    sweep_dir = vna_savepath + '/' + \
       str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
    os.mkdir(sweep_dir)
    np.save("./last_vna_dir.npy", sweep_dir)
    valon.set_frequency(LO, center_freq/1.0e6)
    span = ri.pos_delta
    start = center_freq*1.0e6 - (span/2.)
    stop = center_freq*1.0e6 + (span/2.) 
    sweep_freqs = np.arange(start, stop, lo_step)
    sweep_freqs = np.round(sweep_freqs/lo_step)*lo_step
    if write:
        ri.makeFreqComb()
        ri.writeQDR(ri.freq_comb)
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
        time.sleep(0.1)
    	udp.saveSweepData(Navg, sweep_dir, freq, Nchan) 
	time.sleep(0.1)
    valon.set_frequency(LO, center_freq) # LO
    return 

def targetSweep(ri, udp, valon, write = False, span = 150.0e3, Navg = 50):
    # span = Hz 
    vna_savepath = str(np.load("last_vna_dir.npy"))
    if not os.path.exists(targ_savepath):
        os.makedirs(targ_savepath)
    sweep_dir = targ_savepath + '/' + \
       str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
    os.mkdir(sweep_dir)
    np.save("./last_targ_dir.npy", sweep_dir)
    # uncomment once find_kids works
    #target_freqs = np.load(vna_savepath + '/target_freqs.npy')
    # until find_kids works, just use bb_freqs
    target_freqs = np.load(vna_savepath + "/bb_freqs.npy")
    np.save(sweep_dir + '/target_freqs.npy', target_freqs)
    bb_target_freqs = target_freqs
    # uncomment when find_kids works
    #bb_target_freqs = ((self.target_freqs*1.0e6) - center_freq)
    bb_target_freqs = np.roll(bb_target_freqs, - np.argmin(np.abs(bb_target_freqs)) - 1)
    upconvert = np.sort((bb_target_freqs + center_freq*1.0e6)/1.0e6)
    print "RF tones =", upconvert
    start = center_freq*1.0e6 - (span/2.)
    stop = center_freq*1.0e6 + (span/2.) 
    sweep_freqs = np.arange(start, stop, lo_step)
    sweep_freqs = np.round(sweep_freqs/lo_step)*lo_step
    np.save(sweep_dir + '/bb_freqs.npy', bb_target_freqs)
    np.save(sweep_dir + '/sweep_freqs.npy',sweep_freqs)
    if write:
        ri.writeQDR(bb_target_freqs)
    for freq in sweep_freqs:
        print 'LO freq =', freq/1.0e6
	time.sleep(0.1)
        valon.set_frequency(LO, freq/1.0e6)
	time.sleep(0.2)
    	udp.saveSweepData(Navg, sweep_dir, freq, len(bb_target_freqs)) 
    valon.set_frequency(LO, center_freq)
    return

def openStoredSweep(savepath):
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

def plotTargSweep(path):
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

def getSystemState(fpga, ri, udp, valon):
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
    print "UDP source IP,port:", network[np.where(network == 'udp_source_ip')[0][0]][1], ":", network[np.where(network == 'udp_source_port')[0][0]][1]
    print "UDP dest IP,port:", inet_ntoa(struct.pack(">i", fpga.read_int(regs[np.where(regs == 'udp_destip_reg')[0][0]][1]))), ":", fpga.read_int(regs[np.where(regs == 'udp_destport_reg')[0][0]][1])
    print
    print "ADC and attenuator levels:"
    outAtten, inAtten = readAtten()
    rmsI, rmsQ, crest_factor_I, crest_factor_Q = ri.rmsVoltageADC()
    print "in atten:", inAtten, "dB"
    print "out atten:", outAtten, "dB"
    print "ADC V_rms (I,Q):", rmsI, "mV", rmsQ, "mV"
    print "Crest factor (I,Q):", crest_factor_I, "dB", crest_factor_Q, "dB"
    print
    print "Valon state:"
    #print "Reference:"
    print "LO center freq:", center_freq, "MHz"
    print "Current LO freq:", valon.get_frequency(LO), "MHz"
    print "Current LO power:", np.abs(valon.get_rf_level(LO)), "dBm"
    print "Current clock freq:", valon.get_frequency(CLOCK), "MHz"
    print "Current clock power:", np.abs(valon.get_rf_level(CLOCK)), "dBm"
    return

def plotPhasePSD(chan, udp, ri, time_interval):
    plt.ion()
    #plt.figure(figsize = (10.24, 7.68), dpi = 100)
    plt.title(r' $S_{\phi \phi}$', size = 18)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_ylabel('dBc/Hz', size = 18)
    ax.set_xlabel('log Hz', size = 18)
    phases = udp.streamChanPhase(chan, time_interval)
    f, Spp = signal.welch(phases, ri.accum_freq, nperseg=len(phases)/2)
    #f, Spp = signal.periodogram(phases, fs = 488.28125)
    #f, Spp = sean_psd(phases, 1/self.accum_freq)
    Spp = 10*np.log10(Spp[1:])
    print "MIN =", np.min(Spp), "dBc/Hz"
    print "MAX =", np.max(Spp), "dBc/Hz"
    ax.set_ylim((np.min(Spp) - 10, np.max(Spp) + 10))
    ax.plot(f[1:], Spp, linewidth = 1, label = 'chan ' + str(chan), alpha = 0.7)
    plt.legend(loc = 'upper right')
    plt.grid()
    return

def filter_trace(path, bb_freqs, sweep_freqs):
    chan_I, chan_Q = openStoredSweep(path)
    channels = np.arange(np.shape(chan_I)[1])
    mag = np.zeros((len(bb_freqs),len(sweep_freqs)))
    chan_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))
    for chan in channels:
    	mag[chan] = (np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2))
    	chan_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    mag = np.concatenate((mag[len(mag)/2:], mag[:len(mag)/2]))
    mags = np.hstack(mag)
    mags = 20*np.log10(mags/np.max(mags))
    chan_freqs = np.hstack(chan_freqs)
    chan_freqs = np.concatenate((chan_freqs[len(chan_freqs)/2:],chan_freqs[:len(chan_freqs)/2]))
    return chan_freqs, mags

def lowpass_cosine(y, tau, f_3db, width, padd_data=True):
    # padd_data = True means we are going to symmetric copies of the data to the start and stop
    # to reduce/eliminate the discontinuities at the start and stop of a dataset due to filtering
    #
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
            print spacing, spacing_threshold
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
    	np.save(path + '/last_bb_targ_freqs.npy', bb_target_freqs)
    	print len(rf_target_freqs), "KIDs found:\n"
        print rf_target_freqs
    else:
        print "No freqs found..."

    if plot:
        plt.figure(1)
        plt.plot(chan_freqs, mags,'b', label = 'no filter')
        plt.plot(chan_freqs, filtermags,'g', label = 'filtered')
        plt.xlabel('frequency (MHz)')
        plt.ylabel('dB')
        plt.legend()
        plt.figure(2)
        plt.plot(chan_freqs, mags - filtermags, 'b')
        plt.plot(chan_freqs[ilo],mags[ilo]-filtermags[ilo],'r*')
        plt.figure(4)
        plt.plot(chan_freqs, mags, 'b')
        plt.plot(chan_freqs[kid_idx], mags[kid_idx], 'r*')
        plt.xlabel('frequency (MHz)')
        plt.ylabel('dB')
    return

def menu(captions, options):
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

def main_opt(fpga, ri, udp, valon, upload_status, name, build_time):
    while 1:
        if not fpga:
            print '\n\t\033[93mROACH link is down: Check PPC IP & Network Config\033[93m'
        else:
            print '\n\t\033[92mROACH link is up\033[92m'
        if not upload_status:
            print '\n\t\033[93mNo firmware onboard. If ROACH link is up, try upload option\033[93m'
        else:
            #print '\n\t\033[92mFirmware: \033[92m' + ' ' + name + ' ' + build_time
            print '\n\t\033[92mFirmware:[92m' + firmware_file
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
            ri.lpf(ri.boxcar)
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
	    if not fpga:
	        print "\nROACH link is down"
		break
            try:
	        prompt = raw_input('Full test comb? y/n ')
                if prompt == 'y':
                    ri.makeFreqComb()
		    setAtten(30, 20)
                else:
                    ri.freq_comb = test_freq 
		    setAtten(30, 30)
                if len(ri.freq_comb) > 400:
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
			    break
                        else:
			    fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], shift)
                            print "Wrote DDS shift (" + str(shift) + ")"
                    else:
                        fpga.write_int(regs[np.where(regs == 'dds_shift_reg')[0][0]][1], ri.dds_shift)
            except KeyboardInterrupt:
                pass
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
		setAtten(27, 17)
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
            prompt = raw_input('Write test comb (required if first sweep) ? (y/n) ')
            if prompt == 'y':
                try:
                    vnaSweep(ri, udp, valon, write = True)
                except KeyboardInterrupt:
                    pass
            if prompt == 'n':
                try:
                    vnaSweep(ri, udp, valon)
                except KeyboardInterrupt:
                    pass
            plotVNASweep(str(np.load("last_vna_dir.npy")))
        if opt == 11:
            try:
	        findFreqs(str(np.load("last_vna_dir.npy")), plot = True)
            except KeyboardInterrupt:
	        pass
        if opt == 12:
	    if not fpga:
	        print "\nROACH link is down"
		break
            prompt = raw_input('Write tones (recommended for first time)? (y/n) ')
            if prompt == 'y':
	        try:
                    targetSweep(ri, udp, valon, write = True)
                except KeyboardInterrupt:
                    pass
            if prompt == 'n':
                try:
                    targetSweep(ri, udp, valon)
                except KeyboardInterrupt:
                    pass
            plotTargSweep(str(np.load("last_targ_dir.npy")))
        if opt == 13:
	    if not fpga:
	        print "\nROACH link is down"
		break
            chan = input('Channel number = ? ')
            time_interval = input('Time interval (s) ? ')
            try:
                plotPhasePSD(chan, udp, ri, time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 14:
	    if not fpga:
	        print "\nROACH link is down"
		break
            time_interval = input('Time interval (s) ? ')
            try:
                udp.saveDirfile_chanRange(time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 15:
            sys.exit()
        return upload_status
    
plot_caption = '\n\t\033[95mKID-PY ROACH2 Snap Plots\033[95m'                
plot_opts= ['I & Q ADC input','Firmware FFT','Digital Down Converter Time Domain','Downsampled Channel Magnitudes']

def makePlotMenu(prompt,options):
    print '\t' + prompt + '\n'
    for i in range(len(options)):
        print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
    opt = input()
    return opt

def plot_opt(ri):
    while 1:
        opt = makePlotMenu(plot_caption, plot_opts)
        if opt == 0:
            try:
                ri.plotADC()
            except KeyboardInterrupt:
                pass
        if opt == 1:
            try:
                ri.plotFFT()
            except KeyboardInterrupt:
                pass
        if opt == 2:
            chan = input('Channel = ? ')
            try:
                ri.plotMixer(chan)
            except KeyboardInterrupt:
                pass
        if opt == 3:
            try:
                ri.plotAccum()
            except KeyboardInterrupt:
                pass
    return

def main():
    s = None
    try:
        fpga = casperfpga.katcp_fpga.KatcpFpga(network[np.where(network == 'roach_ppc_ip')[0][0]][1], timeout = 1.)
    except (RuntimeError, AttributeError):
        fpga = None

    # UDP socket
    s = socket(AF_PACKET, SOCK_RAW, htons(3))

    # Valon synthesizer instance
    try:
        valon = valon_synth9.Synthesizer(network[np.where(network == 'valon_comm_port')[0][0]][1]) 
    except OSError:
        "Valon could not be initialized. Check comm port and power supply."

    # Roach interface
    ri = roachInterface(fpga, gc, regs, valon)

    # GbE interface
    udp = roachDownlink(fpga, regs, network, s, ri.accum_freq)
    udp.configSocket()

    os.system('clear')
    while 1:
        try:
            upload_status = 0
            name = ''
            build_time = ''
	    if fpga:
	        if fpga.is_running():
                    #firmware_info = fpga.get_config_file_info()
                    #name = firmware_info['name']
                    #build_time = firmware_info['build_time']
                    upload_status = 1
            time.sleep(0.1)
	    upload_status = main_opt(fpga, ri, udp, valon, upload_status, name, build_time)
        except TypeError:
	    pass
    return

def plot_main():
    try:
        fpga = casperfpga.katcp_fpga.KatcpFpga(network[np.where(network == 'roach_ppc_ip')[0][0]][1], timeout = 120.)
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
