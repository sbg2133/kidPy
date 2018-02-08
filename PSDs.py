import pygetdata as gd
import matplotlib.pyplot as plt
import numpy as np
import struct
from scipy import signal

plt.ion()

def allPSD(dirfile): 
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
