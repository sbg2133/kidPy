import pygetdata as gd
import matplotlib.pyplot as plt
import numpy as np
import struct
from scipy import signal
from sean_psd import amplitude_and_power_spectrum as sean_psd

plt.ion()

#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/fivetwenty/1509742860-Nov-03-2017-14-01-00.dir"
#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/fivetwenty_2/1509744242-Nov-03-2017-14-24-02.dir"
#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/fivetwenty_3/1509745244-Nov-03-2017-14-40-44.dir"
#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/fivetwenty_180/1509747308-Nov-03-2017-15-15-08.dir"
#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509830608-Nov-04-2017-14-23-28.dir"
#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509831884-Nov-04-2017-14-44-44.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509832677-Nov-04-2017-14-57-57.dir"
#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509833004-Nov-04-2017-15-03-24.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509833749-Nov-04-2017-15-15-49.dir"
#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509834609-Nov-04-2017-15-30-09.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509837025-Nov-04-2017-16-10-25.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509837337-Nov-04-2017-16-15-37.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509837715-Nov-04-2017-16-21-55.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509838008-Nov-04-2017-16-26-48.dir" 
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509840413-Nov-04-2017-17-06-53.dir" 
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509841384-Nov-04-2017-17-23-04.dir"  # external referenc, 60 sece
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509842036-Nov-04-2017-17-33-56.dir"  # external ref, Rudin-Shapir, 60 seco
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509842422-Nov-04-2017-17-40-22.dir"  # external ref, Rudin-Sh, 60 sec
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509842712-Nov-04-2017-17-45-12.dir"  # external ref, 60 sec
#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492/1509842712-Nov-04-2017-17-45-12.dir"  # external ref, 60 sec, fir = ones
#dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/1000/1509846373-Nov-04-2017-18-46-13.dir" # external ref, 60 sec, fir = ones, [1-500]/1000
dirfile = "/home/user1/blastFall2017/blastSummer2017/data/meas/492_pfb1/1509920230-Nov-05-2017-15-17-10.dir" # internal ref, 60 sec, fir = ones, PFB1

# LOOPBACK TESTS, NOVEMBER 2017
dirfile = "/home/user1/blastFall2017/blastSummer2017/kidPy/data/R1_492_60/1510338244-Nov-10-2017-11-24-04.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/kidPy/data/R1_492_30/1510338848-Nov-10-2017-11-34-08.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/kidPy/data/R1_492_30_nt/1510339113-Nov-10-2017-11-38-33.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/kidPy/data/R2_492_60_tf/1510342329-Nov-10-2017-12-32-09.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/kidPy/data/R3_492_60_tf/1510345172-Nov-10-2017-13-19-32.dir"
dirfile = "/home/user1/blastFall2017/blastSummer2017/kidPy/data/R4_492_60_nt/1510347740-Nov-10-2017-14-02-20.dir"

dirfile = "/media/user1/FREE/eee505_project_data/nofir_300s_500outof1000/1511898326-Nov-28-2017-12-45-26.dir"
#dirfile = "/media/user1/FREE/eee505_project_data/nofir_300s_499to999/1511899221-Nov-28-2017-13-00-21.dir"
dirfile = "/media/user1/FREE/eee505_project_data/300s_500_heatsinks/1511908164-Nov-28-2017-15-29-24.dir"
dirfile = "/media/user1/FREE/eee505_project_data/100s_500/1511909093-Nov-28-2017-15-44-53.dir"
dirfile = "/media/user1/FREE/eee505_project_data/100s_500_extclock/1511909809-Nov-28-2017-15-56-49.dir"
dirfile = "/media/user1/FREE/eee505_project_data/100s_500_extclock/1511910213-Nov-28-2017-16-03-33.dir"
dirfile = "/media/user1/FREE/eee505_project_data/100s_500_extclock/1511910847-Nov-28-2017-16-14-07.dir"
# 4 fans
dirfile = "/media/user1/FREE/eee505_project_data/1chan_4fans/1512342995-Dec-03-2017-16-16-35.dir" 
dirfile = "/media/user1/FREE/eee505_project_data/1chan_4fans/1512343235-Dec-03-2017-16-20-35.dir" 
dirfile = "/media/user1/FREE/eee505_project_data/1chan_4fans/1512343740-Dec-03-2017-16-29-00.dir" 
dirfile = "/media/user1/FREE/eee505_project_data/1chan_4fans/1512344729-Dec-03-2017-16-45-29.dir" 
dirfile = "/media/user1/FREE/eee505_project_data/1chan_4fans/1512345186-Dec-03-2017-16-53-06.dir" 
# 3 fans
#dirfile = "/media/user1/FREE/eee505_project_data/1chan_3fans/1512344231-Dec-03-2017-16-37-11.dir" 



firstframe = 0
firstsample = 0
d = gd.dirfile(dirfile, gd.RDWR|gd.UNENCODED)
print "Number of frames in dirfile =", d.nframes
nframes = d.nframes

vectors = d.field_list()
pfiles = [s for s in vectors if s[0] == "c"]

def phase_PSD(plot = False):
    wn = []
    if plot:
        plt.figure()
        plt.suptitle(r' $S_{\phi \phi}$', size = 18)
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_ylabel('dBc/Hz', size = 18)
        ax.set_xlabel('log Hz', size = 18)
        plt.grid()
    for i in pfiles:
        d = gd.dirfile(dirfile, gd.RDWR|gd.UNENCODED)
        pvals = d.getdata(i, gd.FLOAT32, first_frame = firstframe, first_sample = firstsample, num_frames = nframes)
        pvals = pvals[~np.isnan(pvals)]
	f, Spp = signal.welch(pvals, 488.28125, nperseg=len(pvals)/2)
        #f, Spp = signal.periodogram(pvals, fs = 488.28125)
        #f, Spp = sean_psd(pvals, 1/488.28125)
        #Spp = 10*np.log10(Spp[1:]) 
	Spp = Spp[Spp != 0.]
	if not np.size(Spp):
	    #mean_wn = np.nan
	    pass
	else:
            Spp = 10*np.log10(Spp) 
	    mean_wn = np.mean(Spp[3*len(Spp)/4:])
	    if plot:
	         ax.plot(f, Spp, linewidth = 1)
            wn.append(mean_wn)
	    plt.tight_layout()
    d.close()
    return np.array(wn)

wn = phase_PSD(plot = True)
plt.figure()
plt.plot(wn)
plt.scatter(range(len(wn)), wn)
plt.xlabel('Chan', size = 18)
plt.ylabel('dBc/Hz', size = 18)
plt.grid()
plt.tight_layout()
