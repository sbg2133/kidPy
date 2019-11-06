import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

center_freq = np.float(gc[np.where(gc == 'center_freq')[0][0]][1])
span = np.float(gc[np.where(gc == 'targ_span')[0][0]][1])
start = center_freq*1.0e6 - (span/2.)
stop = center_freq*1.0e6 + (span/2.)
lo_step_targ  = np.float(gc[np.where(gc == 'lo_step')[0][0]][1])
res_index = np.int(raw_input("What resonator index would you like to look at? "))
comb = np.load("last_freq_comb.npy")
plt.ion()



sweep_freqs = np.arange(start, stop, lo_step_targ)
sweep_freqs = np.round(sweep_freqs/lo_step_targ)*lo_step_targ
res_freqs = sweep_freqs + comb[res_index]



first = True

I_arr = np.zeros(len(sweep_freqs))
Q_arr = np.zeros(len(sweep_freqs))
i = 0


for freq in sweep_freqs:
    print(freq/10**6)
    #time1 = time.time()
    valon.set_frequency(LO, freq/1.0e6)
    #time2 = time.time()
    #print(time2-time1)
    I,Q,phase = udp.streamChanPhase(res_index,2.4,)
    #time3 = time.time()
    #print(time3-time2)
    I_arr[i] = np.mean(I)
    Q_arr[i] = np.mean(Q)
    i = i+1
    
valon.set_frequency(LO, center_freq)
print("0")

fig = plt.figure(1,figsize = (10,6))
ax = fig.add_subplot(121,aspect = "equal")
ax2 = fig.add_subplot(122)
ax.set_ylabel("Q")
ax.set_xlabel("I")
l1, = ax.plot(I_arr,Q_arr)
l2, = ax.plot(I_arr,Q_arr,'o',mec = "k")

print("1")
shift_kHz = np.zeros(100)
time_arr = np.zeros(100)
print("2")
ax2.set_ylabel("shift (kHz)")
ax2.set_xlabel("time (s)")
l1_2, = ax2.plot(time_arr,shift_kHz)
p1_2, = ax2.plot(time_arr[99],shift_kHz[99],'o',mec = "k")
fig.canvas.draw()
print("3")
plt.show(block = False)
print("4")
interp = interpolate.interp2d(I_arr,Q_arr,res_freqs)
print("5")
start_time = time.time()
print("6")
first = True
while True:
    I,Q,phase = udp.streamChanPhase(res_index,0.7,)
    if first:
        f0 = interp(np.mean(I),np.mean(Q))[0]
        p0, = ax.plot(np.mean(I),np.mean(Q),"+",color = "k",markersize = 15)
        p1, = ax.plot(np.mean(I),np.mean(Q),"*",mec = "k",markersize = 15)
        first = False
    p1.set_data(np.mean(I),np.mean(Q))
    f = interp(np.mean(I),np.mean(Q))[0]
    if (f-f0)>(res_freqs[-1]-f0):
        f = res_freqs[-1]
    elif (f-f0)<(res_freqs[0]-f0):
        f = res_freqs[0]
    shift = (f-f0)/1000
    shift_kHz[99] = shift
    time_arr[99] = time.time()-start_time
    l1_2.set_data(time_arr,shift_kHz)
    p1_2.set_data(time_arr[99],shift_kHz[99])
    ax2.relim()
    ax2.autoscale()
    fig.canvas.draw()
    shift_kHz = np.roll(shift_kHz,-1)
    time_arr = np.roll(time_arr,-1)

    

    
