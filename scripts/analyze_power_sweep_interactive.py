import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from multitone_kidPy import analyze # see below
from scipy import interpolate
import numpy as np
from multitone_kidPy import read_multitone as rm #see below
from KIDs import resonance_fitting as resf #see below
import copy

# This program relays on code from https://github.com/GlennGroupUCB/submm_python_routines

# new version of tranfer function creating program that is interactive allowing for
# human overriding becuase fitting of resonators fails when the situation is not optimal
# i.e. when there are a bunch of collisions but you are still desperate to do your best
# to readout every single resonator.

# I was putting this in a directory called power sweeps so that it doesn't pollute
# some other directory with a bunch of output files

# note this doesn't work on mac becuase the backend doesn't recognize the key event for shift

# if you have already fit all the data select false i.e. used analyze on the go in power sweep
# fitting the data takes a while consider doing it on a server in parallel
fit_scans = False

# select the level of non-liearity paramter (a) you would like to normalize power to
bif_level = 0.5

#read in the names of the scan in the power sweep
fine_names = np.load("2018-09-0715:09:47_fine_names.npy")
gain_names = np.load("2018-09-0715:09:47_gain_names.npy")
attn_levels = np.load("2018-09-0715:09:47_attn_levels.npy")

# would be nice to parralize this statement
if fit_scans:
    for i in range(0,len(fine_names)):
        print("fitting scan "+str(i))
        analyze.fit_fine_gain_std(fine_names[i],gain_names[i])


#some stuff to run it on a seperature computer where data is storted in data dir
'''
fine_names2 = []
gain_names2 = []

for i in range(0,len(fine_names)):
    fine_names2.append("data/"+fine_names[i].decode().split("/")[-1])
    gain_names2.append("data/"+gain_names[i].decode().split("/")[-1])

fine_names = fine_names2
gain_names = gain_names2
'''

#grab the first set to intialize some arrays to store data.
fine = rm.read_iq_sweep(fine_names[0],load_std = True)
gain = rm.read_iq_sweep(gain_names[0],load_std = True)


#load in the first data set
mag_fits = np.load(fine_names[0]+"/all_fits_mag.npy")
iq_fits = np.load(fine_names[0] + "/all_fits_iq.npy")


#intialize arrays to hold all of the fit results
all_fits_mag = np.zeros((mag_fits.shape[0],mag_fits.shape[1],len(attn_levels)))
all_fits_iq = np.zeros((iq_fits.shape[0],iq_fits.shape[1],len(attn_levels)))
all_fine_f = np.zeros((fine['freqs'].shape[0],fine['freqs'].shape[1],len(attn_levels)))
all_fine_z = np.zeros((fine['freqs'].shape[0],fine['freqs'].shape[1],len(attn_levels)),dtype = 'complex')
all_fine_z_fit_mag = np.zeros((fine['freqs'].shape[0],fine['freqs'].shape[1],len(attn_levels)))
all_fine_z_fit_iq = np.zeros((fine['freqs'].shape[0],fine['freqs'].shape[1],len(attn_levels)),dtype = 'complex')
#all_gain_f = np.zeros((gain['freqs'].shape[0],gain['freqs'].shape[1],len(attn_levels)))
#all_gain_z = np.zeros((gain['freqs'].shape[0],gain['freqs'].shape[1],len(attn_levels)),dtype = 'complex')

# collect all of the fit results and data
# nested for loop with funciton calls it takes a little while to read in the data
for i in range(0,len(fine_names)):
    print(i)
    fine = rm.read_iq_sweep(fine_names[i])
    gain = rm.read_iq_sweep(gain_names[i])
    all_fine_f[:,:,i] = fine['freqs']
    all_fine_z[:,:,i] = fine['I']+1.j*fine['Q']
    #all_gain_f[:,:,i] = gain['freqs']
    #all_gain_z[:,:,i] = gain['I'][:,i]+1.j*gain['Q'][:,i]
    all_fits_mag[:,:,i] = np.load(fine_names[i] + "/all_fits_mag.npy")
    all_fits_iq[:,:,i] = np.load(fine_names[i]+"/all_fits_iq.npy")
    for j in range(0,all_fits_mag.shape[1]):
        if all_fits_mag[0,j,i] != 0:
            all_fine_z_fit_mag[:,j,i] = resf.nonlinear_mag(all_fine_f[:,j,i]*10**6,all_fits_mag[0,j,i],all_fits_mag[1,j,i],all_fits_mag[2,j,i],all_fits_mag[3,j,i],all_fits_mag[4,j,i],all_fits_mag[5,j,i],all_fits_mag[6,j,i],all_fits_mag[7,j,i])
        if all_fits_iq[0,j,i] != 0:
            all_fine_z_fit_iq[:,j,i] = resf.nonlinear_iq(all_fine_f[:,j,i]*10**6,all_fits_iq[0,j,i],all_fits_iq[1,j,i],all_fits_iq[2,j,i],
                                                         all_fits_iq[3,j,i],all_fits_iq[4,j,i],all_fits_iq[5,j,i],all_fits_iq[6,j,i],all_fits_iq[7,j,i],all_fits_iq[8,j,i])


class interactive_plot(object):

    def __init__(self,Is,Qs,z_fit_mag,Is_fit_iq,Qs_fit_iq,chan_freqs,all_fits_mag,all_fits_iq,attn_levels,bif_level):
        self.attn_levels = attn_levels
        self.bif_level = bif_level
        self.Is = Is
        self.Qs = Qs
        self.z_fit_mag = z_fit_mag
        self.Is_fit_iq = Is_fit_iq
        self.Qs_fit_iq = Qs_fit_iq
        self.all_fits_mag = all_fits_mag
        self.all_fits_iq = all_fits_iq
        self.chan_freqs = chan_freqs
        self.targ_size = chan_freqs.shape[0]
        self.plot_index = 0
        self.power_index = 0
        self.res_index_overide = np.asarray((),dtype = np.int16)
        self.overide_freq_index = np.asarray((),dtype = np.int16)
        self.shift_is_held = False
        self.bif_levels_mag = np.zeros(all_fits_mag.shape[1])
        self.bif_levels_iq = np.zeros(all_fits_mag.shape[1])
        # here we compute our best guess for the proper power level from the fit data
        for i in range(0,self.all_fits_mag.shape[1]):
            try:
                first_bif_mag = np.where(all_fits_mag[4,i,:]>bif_level)[0][0]
            except:
                first_bif_mag = all_fits_mag.shape[2]-1
            try:
                first_bif_iq = np.where(all_fits_iq[4,i,:]>bif_level)[0][0]
            except:
                first_bif_iq = self.all_fits_iq.shape[2]-1
            if first_bif_mag == 0:
                first_bif_mag =1
            if first_bif_iq == 0:
                first_bif_iq = 1

            interp_mag  = interpolate.interp1d(-attn_levels[0:first_bif_mag+1],all_fits_mag[4,i,:][0:first_bif_mag+1] )
            interp_iq  = interpolate.interp1d(-attn_levels[0:first_bif_iq+1],all_fits_iq[4,i,:][0:first_bif_iq+1] )

            powers_mag = np.linspace(-attn_levels[0]+.01,-attn_levels[first_bif_mag]-0.01,1000)
            powers_iq = np.linspace(-attn_levels[0]+.01,-attn_levels[first_bif_iq]-0.01,1000)

            bifurcation_mag = powers_mag[np.argmin(np.abs(interp_mag(powers_mag)-np.ones(len(powers_mag))*bif_level))]
            bifurcation_iq = powers_iq[np.argmin(np.abs(interp_iq(powers_iq)-np.ones(len(powers_iq))*bif_level))]
            self.bif_levels_mag [i] = bifurcation_mag
            self.bif_levels_iq [i] = bifurcation_iq

        self.bif_levels = copy.deepcopy(self.bif_levels_mag) 
        self.power_index = np.argmin(np.abs(self.bif_levels[self.plot_index]+self.attn_levels))
        self.fig = plt.figure(1,figsize = (13,10))
        self.ax = self.fig.add_subplot(221)
        self.ax.set_ylabel("Power (dB)")
        self.ax.set_xlabel("Frequecy (MHz)")
        self.ax2 = self.fig.add_subplot(222)
        self.ax2.set_ylabel("Q")
        self.ax2.set_xlabel("I")
        self.ax3 = self.fig.add_subplot(212)
        self.ax3.set_xlabel("Power level")
        self.ax3.set_ylabel("Nolinearity parameter a")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.l1_fit, = self.ax.plot(self.chan_freqs[:,self.plot_index],10*np.log10(self.z_fit_mag[:,self.plot_index,self.power_index]),label = "Fit")
        self.l1, = self.ax.plot(self.chan_freqs[:,self.plot_index],10*np.log10(self.Is[:,self.plot_index,self.power_index]**2+self.Qs[:,self.plot_index,self.power_index]**2),'o',mec = "k",label = "Data")
        self.ax.legend()
        self.l2_fit, = self.ax2.plot(self.Is_fit_iq[:,self.plot_index,self.power_index],Qs_fit_iq[:,self.plot_index,self.power_index],label = "Fit")
        self.l2, = self.ax2.plot(self.Is[:,self.plot_index,self.power_index],Qs[:,self.plot_index,self.power_index],'o',mec = "k",label = "Data")
        self.ax2.legend()
        self.l3, = self.ax3.plot(-self.attn_levels,self.all_fits_mag[4,self.plot_index,:],color = 'r',label = "Mag fit")
        self.l4, = self.ax3.plot(-self.attn_levels,self.all_fits_iq[4,self.plot_index,:],color = 'g',label = "IQ fit")
        self.l8, = self.ax3.plot((self.bif_levels[self.plot_index],self.bif_levels[self.plot_index]),(0,1),"--",color = 'm',linewidth = 3,label = "Power pick")
        self.l5, = self.ax3.plot((-self.attn_levels[self.power_index],-self.attn_levels[self.power_index]),(0,1),"--",color = 'k',label = "Current power")
        self.l6, = self.ax3.plot((self.bif_levels_mag[self.plot_index],self.bif_levels_mag[self.plot_index]),(0,self.bif_level),"--",color = 'r')
        self.l7, = self.ax3.plot((self.bif_levels_iq[self.plot_index],self.bif_levels_iq[self.plot_index]),(0,self.bif_level),"--",color = 'g')
        self.ax3.legend()
        self.ax3.set_ylim(0,1)
        self.ax.set_title("Resonator index "+str(self.plot_index))
        self.ax2.set_title("Power level "+str(-self.attn_levels[self.power_index]))
        print("")
        print("Interactive Power Tuning Activated")
        print("Use left and right arrows to switch between resonators")
        print("Use the up and down arrows to change between power levels")
        print("Hold shift and right click on the bottom plot to overide picked power level")
        print("or hold shift and press enter to overide picked power level to the current plotted power level")
        plt.show(block = True)

    def refresh_plot(self):
        self.l1.set_data(self.chan_freqs[:,self.plot_index],10*np.log10(self.Is[:,self.plot_index,self.power_index]**2+self.Qs[:,self.plot_index,self.power_index]**2))
        self.l1_fit.set_data(self.chan_freqs[:,self.plot_index],10*np.log10(self.z_fit_mag[:,self.plot_index,self.power_index]))
        self.ax.relim()
        self.ax.autoscale()
        self.ax.set_title("Resonator index "+str(self.plot_index))
        self.ax2.set_title("Power level "+str(-self.attn_levels[self.power_index]))
        self.l2.set_data((self.Is[:,self.plot_index,self.power_index],self.Qs[:,self.plot_index,self.power_index]))
        self.l2_fit.set_data((self.Is_fit_iq[:,self.plot_index,self.power_index],self.Qs_fit_iq[:,self.plot_index,self.power_index]))
        self.l3.set_data(-self.attn_levels,self.all_fits_mag[4,self.plot_index,:])
        self.l4.set_data(-self.attn_levels,self.all_fits_iq[4,self.plot_index,:])        
        self.l5.set_data((-self.attn_levels[self.power_index],-self.attn_levels[self.power_index]),(0,1))
        self.l6.set_data((self.bif_levels_mag[self.plot_index],self.bif_levels_mag[self.plot_index]),(0,self.bif_level))
        self.l7.set_data((self.bif_levels_iq[self.plot_index],self.bif_levels_iq[self.plot_index]),(0,self.bif_level))
        self.l8.set_data((self.bif_levels[self.plot_index],self.bif_levels[self.plot_index]),(0,1))
        self.ax2.relim()
        self.ax2.autoscale()
        plt.draw()
        
    def on_key_press(self, event):
        #print(event.key) #for debugging
        if event.key == 'right':
           if self.plot_index != self.chan_freqs.shape[1]-1:
               self.plot_index = self.plot_index +1
               #snap to the automated choice in power level
               self.power_index = np.argmin(np.abs(self.bif_levels[self.plot_index]+self.attn_levels))
               self.refresh_plot()

        if event.key == 'left':
           if self.plot_index != 0:
               self.plot_index = self.plot_index -1
               #snap to the automated choice in power level
               self.power_index = np.argmin(np.abs(self.bif_levels[self.plot_index]+self.attn_levels))
               self.refresh_plot()

        if event.key == 'up':
            if self.power_index != self.Is.shape[2]-1:
                self.power_index = self.power_index +1
            self.refresh_plot()

        if event.key == 'down':
            if self.power_index != 0:
               self.power_index = self.power_index -1
            self.refresh_plot()

        if event.key == 'shift':
                self.shift_is_held = True
                #print("shift pressed") #for debugging

        if event.key == 'enter':
                if self.shift_is_held:
                    self.bif_levels[self.plot_index] = -self.attn_levels[self.power_index]
                    self.refresh_plot()
                    

    def on_key_release(self,event):
        if event.key == "shift":
            self.shift_is_held = False
            #print("shift released") #for debugging

    def onClick(self,event):
        if event.button == 3:
            if self.shift_is_held:
                print("overiding point selection",event.xdata)
                self.bif_levels[self.plot_index] = event.xdata
                self.refresh_plot()


# Call the interactive plot
ip = interactive_plot(np.real(all_fine_z),np.imag(all_fine_z),all_fine_z_fit_mag,np.real(all_fine_z_fit_iq),np.imag(all_fine_z_fit_iq),all_fine_f[:,:,0],all_fits_mag,all_fits_iq,attn_levels,bif_level)
            

#plot the resultant picked power levels and the corresponding transfer function
plt.figure(1,figsize = (12,10))
plt.subplot(211)
plt.title("Bifurcation power")
plt.plot(ip.bif_levels+attn_levels[-1],'o',color = 'g',label = "Picked Valuse")
plt.xlabel("Resonator index")
plt.ylabel("Power at a = "+ str(bif_level))
plt.legend(loc = 4)

plt.subplot(212)
plt.title("Transfer function")
plt.plot(np.sqrt(10**((ip.bif_levels+attn_levels[-1])/10)/np.min(10**((ip.bif_levels+attn_levels[-1])/10))),'o')
plt.xlabel("Resonator index")
plt.ylabel("Voltage factor")

plt.savefig("power_sweep_results_"+fine_names[0].split('/')[-1]+".pdf")

# save the output
np.savetxt("bif_levels_mag_"+fine_names[0].split('/')[-1]+".csv",ip.bif_levels_mag+attn_levels[-1])
np.savetxt("bif_levels_iq_"+fine_names[0].split('/')[-1]+".csv",ip.bif_levels_iq+attn_levels[-1])
np.save("trans_func_"+fine_names[0].split('/')[-1],np.sqrt(10**((ip.bif_levels+attn_levels[-1])/10)/np.min(10**((ip.bif_levels+attn_levels[-1])/10))))

plt.show()
