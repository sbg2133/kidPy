import numpy as np
import matplotlib.pyplot as plt


center_freq = np.float(gc[np.where(gc == 'center_freq')[0][0]][1])
span = np.float(gc[np.where(gc == 'targ_span')[0][0]][1])
start = center_freq*1.0e6 - (span/2.)
stop = center_freq*1.0e6 + (span/2.)
lo_step_targ  = np.float(gc[np.where(gc == 'lo_step')[0][0]][1])
res_index = np.int(raw_input("What resonator index would you like to look at?"))
comb = np.load("last_freq_comb.npy")
plt.ion()



sweep_freqs = np.arange(start, stop, lo_step_targ)
sweep_freqs = np.round(sweep_freqs/lo_step_targ)*lo_step_targ
res_freqs = sweep_freqs + comb[res_index]



first = True

I_arr = np.zeros(len(sweep_freqs))
Q_arr = np.zeros(len(sweep_freqs))
i = 0

while True:
    for freq in sweep_freqs:
        print(freq/10**6)
        #time1 = time.time()
        valon.set_frequency(LO, freq/1.0e6)
        #time2 = time.time()
        #print(time2-time1)
        I,Q = udp.returnSweepData(1, freq, len(comb),skip_packets = 0)
        #time3 = time.time()
        #print(time3-time2)
        I_arr[i] = I[res_index]
        Q_arr[i] = Q[res_index]
        if first == False:
            l2.set_data(res_freqs,10*np.log10(I_arr**2+Q_arr**2))
            p1.set_data(res_freqs[i],10*np.log10(I_arr[i]**2+Q_arr[i]**2))
            ax.relim()
            ax.autoscale()
            fig.canvas.draw()
        i = i +1
    if first:
        i = 0
        fig = plt.figure(1,figsize = (10,6))
        ax = fig.add_subplot(111)
        ax.set_ylabel("Power (dB)")
        ax.set_xlabel("Frequecy (Hz)")
        l1, = ax.plot(res_freqs,10*np.log10(I_arr**2+Q_arr**2))
        l2, = ax.plot(res_freqs,10*np.log10(I_arr**2+Q_arr**2))
        p1, = ax.plot(res_freqs[i],10*np.log10(I_arr[i]**2+Q_arr[i]**2),"o",mec = "k")
        plt.show(block = False)
        fig.canvas.draw()
        first = False
    else:
    #    l2.set_data(res_freqs,10*np.log10(I_arr**2+Q_arr**2))
    #    ax.relim()
    #    ax.autoscale()
    #    fig.canvas.draw()
        i = 0

    

    
