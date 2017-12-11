import numpy as np
import sys, os
import errno
import struct
import casperfpga
import socket
from roachInterface import roachInterface
from gbeConfig import roachDownlink
import time
import matplotlib.pyplot as plt
from sean_psd import amplitude_and_power_spectrum as sean_psd
from scipy import signal
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

# UDP packet 
buf_size = int(network[np.where(network == 'buf_size')[0][0]][1])
header_len = int(network[np.where(network == 'header_len')[0][0]][1])

# Valon channels
CLOCK = 1
LO = 2
valon_init_message = "python /home/pi/device_control/kidpy/init_valon.py"
valon_read_message = "python /home/pi/device_control/kidpy/read_valon.py"

center_freq = np.float(gc[np.where(gc == 'center_freq')[0][0]][1])
test_freq = np.float(gc[np.where(gc == 'test_freq')[0][0]][1])
test_freq = np.array([test_freq])

# Pi ports
FC2 = 12346
FC1 = 12345

def piSocket(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    server_address = (network[np.where(network == 'pi_ip')[0][0]][1], port)
    #print >> sys.stderr, 'connecting to %s port %s' % server_address
    try:
        sock.connect(server_address)
        return sock
    except socket.error, v:
        errorcode = v[0]
	if errorcode==errno.ECONNREFUSED:
	    pass
    return
    
def piSendRecv(sock, message, recv = True):
    try:
        message += '\n'
        #print >>sys.stderr, 'sending "%s"' % message
        sock.sendall(message)
        if not recv:
	    return
	else:
	    return sock.recv(256)
    except KeyboardInterrupt:
        return None

def testPiConn(sock):
    message = 'whoami\n'
    response = piSendRecv(sock, message)
    if response:
        print "Connection OK"
    else: 
        print "No response from Pi"
    return

def testRoachConn(fpga):
    if not fpga:
        try:
	    fpga = casperfpga.katcp_fpga.KatcpFpga(network[np.where(network == 'roach_ppc_ip')[0][0]][1], timeout = 120.)
        except RuntimeError:
	    print "\nNo connection to ROACH. If booting, wait 30 seconds and retry. Otherwise, check network config."
    return fpga

def initValonPi(sock):
    response = piSendRecv(sock, valon_init_message)
    r = response.splitlines()
    print r[0]
    return

def readValonPi(sock):
    response = piSendRecv(sock, valon_read_message)
    r = response.splitlines()
    f_clock = r[0]
    pow_clock = r[1]
    f_lo = r[2]
    pow_lo = r[3]
    return f_clock, pow_clock, f_lo, pow_lo

def setValonPi(sock, lo_freq):
    valon_set_message = 'python /home/pi/device_control/kidpy/set_lo.py ' + str(lo_freq)
    response = piSendRecv(sock, valon_set_message)
    freq = response.splitlines()[0]
    if freq != 0:
        print freq
    else:
        print "Error setting Valon"
    return

# For setting MCL RUDAT attenuators with direct USB connection
def setAttenPi(sock, outAtten, inAtten):
    message = 'sudo ./set_rudats %g %g' % (outAtten, inAtten)
    piSendRecv(sock, message, recv = False)
    return

# For reading attenuators
def readAttenPi(sock):
    set_message = "sudo ./read_rudats > rudat.log"
    piSendRecv(sock, set_message, recv = False)
    read_message = "cat rudat.log"
    response = piSendRecv(sock, read_message).splitlines()
    outAtten = response[1]
    inAtten = response[3]
    return outAtten, inAtten

# calibrate ADC input level, in millivolts
def calibrateADC(target_rms_mv, pi, outAtten, inAtten):
    setAttenPi(pi, outAtten, inAtten)
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
	    setAttenPi(pi, outAtten, inAtten)
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
	    setAttenPi(pi, outAtten, inAtten)
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
main_opts= ['Test connection to Pi', 'Test connection to Roach', 'Upload firmware', 'Initialize system & UDP conn','Write test comb (single or multitone)', 'Write stored comb', 'Apply inverse transfer function', 'Calibrate ADC V_rms', 'Get system state','Test GbE downlink', 'Print packet info to screen (UDP)','VNA sweep and plot','Locate resonances','Target sweep and plot', 'Plot channel phase PSD (quick look)', 'Save dirfile for range of chan (phase)','Exit'] 

def vnaSweep(ri, udp, pi, write = False, Navg = 50):
    if not os.path.exists(vna_savepath):
        os.makedirs(vna_savepath)
    sweep_dir = vna_savepath + '/' + \
       str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
    os.mkdir(sweep_dir)
    np.save("./last_vna_dir.npy", sweep_dir)
    setValonPi(pi, center_freq)
    span = ri.pos_delta
    start = center_freq*1.0e6 - (span/2.)
    stop = center_freq*1.0e6 + (span/2.) 
    sweep_freqs = np.arange(start, stop, ri.lo_step)
    sweep_freqs = np.round(sweep_freqs/ri.lo_step)*ri.lo_step
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
	time.sleep(0.25)
        setValonPi(pi, freq/1.0e6)
    	udp.saveSweepData(Navg, sweep_dir, freq, Nchan) 
    setValonPi(pi, center_freq)
    return 

def targetSweep(ri, udp, pi, write = False, span = 150.0e3, Navg = 50):
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
    sweep_freqs = np.arange(start, stop, ri.lo_step)
    sweep_freqs = np.round(sweep_freqs/ri.lo_step)*ri.lo_step
    np.save(sweep_dir + '/bb_freqs.npy', bb_target_freqs)
    np.save(sweep_dir + '/sweep_freqs.npy',sweep_freqs)
    if write:
        ri.writeQDR(bb_target_freqs)
    for freq in sweep_freqs:
	time.sleep(0.01)
        setValonPi(pi, freq/1.0e6)
	time.sleep(0.1)
    	udp.saveSweepData(Navg, sweep_dir, freq, len(bb_target_freqs)) 
    setValonPi(pi, center_freq)
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

def getSystemState(fpga, ri, udp, pi):
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
    print "UDP dest IP,port:", socket.inet_ntoa(struct.pack(">i", fpga.read_int(regs[np.where(regs == 'udp_destip_reg')[0][0]][1]))), ":", fpga.read_int(regs[np.where(regs == 'udp_destport_reg')[0][0]][1])
    print
    print "ADC and attenuator levels:"
    #outAtten, inAtten = readAtten()
    rmsI, rmsQ, crest_factor_I, crest_factor_Q = ri.rmsVoltageADC()
    try:
        outAtten, inAtten = readAttenPi(pi)
        print "in atten:", inAtten, "dB"
        print "out atten:", outAtten, "dB"
        print "Valon state:"
        #print "Reference:"
        f_clock, p_clock, f_lo, p_lo = readValonPi(pi)
        print "LO center freq:", center_freq, "MHz"
        print "Current LO freq:", f_lo, "MHz"
        print "Current LO power:", p_lo, "dBm"
        print "Current clock freq:", f_clock, "MHz"
        print "Current clock power:", p_clock, "dBm"
    except AttributeError:
        print "Pi socket not available. Try restarting kidPy"
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

def menu(captions, options):
    print '\t' + captions[0] + '\n'
    for i in range(len(options)):
        if (i < 10):
            print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
    print '\t' + captions[1] + '\n'
    for i in range(len(options)):
        if (i >= 10):
            print '\t' +  '\033[32m' + str(i) + ' ..... ' '\033[0m' +  options[i] + '\n'
    opt = input()
    return opt

def main_opt(fpga, ri, udp, pi, upload_status, name, build_time):
    while 1:
        if not fpga:
            print '\n\t\033[93mROACH link is down: Check PPC IP & Network Config\033[93m'
        else:
            print '\n\t\033[92mROACH link is up\033[92m'
        if not upload_status:
            print '\n\t\033[93mNo firmware onboard. If ROACH link is up, try upload option\033[93m'
        else:
            print '\n\t\033[92mFirmware: \033[92m' + ' ' + name + ' ' + build_time
	if not pi:
	    print '\n\t\033[93mPi link is down: Try restarting\033[93m'
	else:
	    print '\n\t\033[92mPi link is up\033[92m'
        opt = menu(captions,main_opts)
	if opt == 0:
	    try:
	        testPiConn(pi)
	    except AttributeError:
	        break
        if opt == 1:
	    result = testRoachConn(fpga)
	    if not result:
	        break
	    else:
	        fpga = result
		print "\n Connection is up"
	if opt == 2:
	    if not fpga:
	        print "\nROACH link is down"
		break
	    if (ri.uploadfpg() < 0):
		print "\nFirmware upload failed"
            else:
	        upload_status = 1
        if opt == 3:
	    if not fpga:
	        print "\nROACH link is down"
		break
            os.system('clear')
            try:
	        initValonPi(pi)
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
	if opt == 4:
	    if not fpga:
	        print "\nROACH link is down"
		break
            try:
	        prompt = raw_input('Full test comb (N = 1000) ? y/n) ')
                if prompt == 'y':
                    ri.makeFreqComb()
		    setAttenPi(pi, 30, 20)
                else:
                    ri.freq_comb = test_freq
		    setAttenPi(pi, 30, 30)
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
	if opt == 5:
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
		setAttenPi(pi, 27, 17)
                np.save("last_freq_comb.npy", ri.freq_comb)
	    except KeyboardInterrupt:
	        pass
        if opt == 6:
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
	if opt == 7:
	    if not fpga:
	        print "\nROACH link is down"
		break
	    try:
                calibrateADC(83., 20, 20)
	    except KeyboardInterrupt:
	        pass
        if opt == 8:
	    if not fpga:
	        print "\nROACH link is down"
		break
	    getSystemState(fpga, ri, udp, pi)
	if opt == 9:
	    if not fpga:
	        print "\nROACH link is down"
		break
	    if (udp.testDownlink(5) < 0):
	        print "Error receiving data. Check ethernet configuration."
            else:
	        print "OK"
	        fpga.write_int(regs[np.where(regs == 'write_stream_status_reg')[0][0]][1], 1)
	if opt == 10:
	    if not fpga:
	        print "\nROACH link is down"
		break
            time_interval = input('\nNumber of seconds to stream? ' )
            chan = input('chan = ? ')
            try:
                udp.printChanInfo(chan, time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 11:
	    if not fpga:
	        print "\nROACH link is down"
		break
            prompt = raw_input('Write test comb (required if first sweep) ? (y/n) ')
            if prompt == 'y':
                try:
                    vnaSweep(ri, udp, pi, write = True)
                except KeyboardInterrupt:
                    pass
            if prompt == 'n':
                try:
                    vnaSweep(ri, udp, pi)
                except KeyboardInterrupt:
                    pass
            plotVNASweep(str(np.load("last_vna_dir.npy")))
        if opt == 12:
	    if not fpga:
	        print "\nROACH link is down"
		break
            path = ""
            try:
	        fk.main(path)
            except KeyboardInterrupt:
	        pass
        if opt == 13:
	    if not fpga:
	        print "\nROACH link is down"
		break
            prompt = raw_input('Write tones (recommended for first time)? (y/n) ')
            if prompt == 'y':
	        try:
                    targetSweep(ri, udp, pi, write = True)
                except KeyboardInterrupt:
                    pass
            if prompt == 'n':
                try:
                    targetSweep(ri, udp, pi)
                except KeyboardInterrupt:
                    pass
            plotTargSweep(str(np.load("last_targ_dir.npy")))
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
                udp.saveDirfile_chanRange(time_interval)
            except KeyboardInterrupt:
                pass
        if opt == 16:
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
    pi = None
    s = None
    try:
        fpga = casperfpga.katcp_fpga.KatcpFpga(network[np.where(network == 'roach_ppc_ip')[0][0]][1], timeout = 120.)
    except RuntimeError:
        fpga = None
    
    # UDP socket
    s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))

    # Roach interface
    ri = roachInterface(fpga, gc, regs)

    # GbE interface
    udp = roachDownlink(fpga, regs, network, s, ri.accum_freq)
    udp.configSocket()
    
    #os.system('clear')
    while 1:
        try:
            upload_status = 0
            name = ''
            build_time = ''
            if fpga:
	        if fpga.is_running():
                    firmware_info = fpga.get_config_file_info()
                    name = firmware_info['name']
                    build_time = firmware_info['build_time']
                    upload_status = 1
	    time.sleep(0.1)
            pi = piSocket(FC2)
            upload_status = main_opt(fpga, ri, udp, pi, upload_status, name, build_time)
	    if pi:
	        pi.shutdown(1)
	        pi.close()
        except KeyboardInterrupt:
	    if pi:
	        pi.shutdown(1)
	        pi.close()
	    break
    return

def plot_main():
    try:
        fpga = casperfpga.katcp_fpga.KatcpFpga(network[np.where(network == 'roach_ppc_ip')[0][0]][1], timeout = 120.)
    except RuntimeError:
        fpga = None
    # Roach interface
    ri = roachInterface(fpga, gc, regs)
    while 1:
        plot_opt(ri)
    return

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_main()
    else:
       main()
