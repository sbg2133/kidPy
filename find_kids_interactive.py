import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy import signal, ndimage, fftpack

class interactive_plot(object): # will eventually be deleted saved in case of problems

    def __init__(self):
	self.fig = plt.figure(4,figsize = (16,6))
	self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.shift_is_held = False
        self.control_is_held = False
        self.add_list = []
        self.delete_list = []
	print "please hold either the shift or control key while right clicking to add or remove points"
	print "close all plots when done"
        
    def on_key_press(self, event):
        #print event.key
	#has to be shift and ctrl because remote viewers only forward
	#certain key combinations
	#print event.key == 'd'
        if event.key == 'shift':
           self.shift_is_held = True
        if event.key == 'control':
           self.control_is_held = True

    def on_key_release(self, event):
       if event.key == 'shift':
           self.shift_is_held = False
       if event.key == 'control':
           self.control_is_held = False

    def onClick(self, event):
        if event.button == 3:
            if self.shift_is_held:
                print "adding point", event.xdata
                #self.add_list.append(lambda n=event.xdata: n)
                self.add_list.append(event.xdata)
                plt.plot(event.xdata,event.ydata,"*", markersize = 20)
            elif self.control_is_held:
                print "removing point", event.xdata
                #self.delete_list.append(lambda n=event.xdata: n)
                self.delete_list.append(event.xdata)
                plt.plot(event.xdata,event.ydata,"x",markersize = 20,mew = 5)
            else:
                print "please hold either the shift or control key while right clicking to add or remove points"


class interactive_plot_2(object):

    def __init__(self,chan_freqs,data,kid_idx):
	plt.rcParams['keymap.forward'] = ['v']
	plt.rcParams['keymap.back'] = ['c','backspace']# remove arrows from back and forward on plot
        self.chan_freqs = chan_freqs
        self.data = data
        self.kid_idx = kid_idx
	self.lim_shift_factor = 0.1
	self.zoom_factor = 0.1 #no greater than 0.5
        self.kid_idx_len = len(kid_idx)
        self.fig = plt.figure(4,figsize = (16,6))
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.l1, = self.ax.plot(self.chan_freqs,self.data)
        self.p1, = self.ax.plot(self.chan_freqs[self.kid_idx],self.data[self.kid_idx],"r*",markersize = 8)
        self.text_dict = {}
        for i in range(0,len(self.kid_idx)):
            self.text_dict[i] = plt.text(self.chan_freqs[self.kid_idx][i], self.data[self.kid_idx][i], str(i))
            
        self.shift_is_held = False
        self.control_is_held = False
        self.add_list = []
        self.delete_list = []
        print("please hold either the shift or control key \n while right clicking to add or remove points")
        print("You can use the arrow keys to pan around")
	print("You can use z and x keys to zoom in and out")
        print("close all plots when finished")
        plt.xlabel('frequency (MHz)')
        plt.ylabel('dB')
        plt.show(block = True)
        
    def on_key_press(self, event):
        #print event.key
	#has to be shift and ctrl because remote viewers only forward
	#certain key combinations
	#print event.key == 'd'
        if event.key == 'shift':
           self.shift_is_held = True

        if event.key == 'control':
           self.control_is_held = True

        if event.key == 'right': #pan right
		xlim_left, xlim_right = self.ax.get_xlim() 
		xlim_size = xlim_right-xlim_left
		self.ax.set_xlim(xlim_left+self.lim_shift_factor*xlim_size,xlim_right+self.lim_shift_factor*xlim_size)
		plt.draw()       

        if event.key == 'left': #pan left
		xlim_left, xlim_right = self.ax.get_xlim() 
		xlim_size = xlim_right-xlim_left
		self.ax.set_xlim(xlim_left-self.lim_shift_factor*xlim_size,xlim_right-self.lim_shift_factor*xlim_size)
		plt.draw()

        if event.key == 'up': #pan up
		ylim_left, ylim_right = self.ax.get_ylim() 
		ylim_size = ylim_right-ylim_left
		self.ax.set_ylim(ylim_left+self.lim_shift_factor*ylim_size,ylim_right+self.lim_shift_factor*ylim_size)
		plt.draw()       

        if event.key == 'down': #pan down
		ylim_left, ylim_right = self.ax.get_ylim() 
		ylim_size = ylim_right-ylim_left
		self.ax.set_ylim(ylim_left-self.lim_shift_factor*ylim_size,ylim_right-self.lim_shift_factor*ylim_size)
		plt.draw()

        if event.key == 'z': #zoom in
		xlim_left, xlim_right = self.ax.get_xlim() 
		ylim_left, ylim_right = self.ax.get_ylim() 
		xlim_size = xlim_right-xlim_left
		ylim_size = ylim_right-ylim_left
		self.ax.set_xlim(xlim_left+self.zoom_factor*xlim_size,xlim_right-self.zoom_factor*xlim_size)
		self.ax.set_ylim(ylim_left+self.zoom_factor*ylim_size,ylim_right-self.zoom_factor*ylim_size)
		plt.draw() 

        if event.key == 'x': #zoom out
		xlim_left, xlim_right = self.ax.get_xlim() 
		ylim_left, ylim_right = self.ax.get_ylim() 
		xlim_size = xlim_right-xlim_left
		ylim_size = ylim_right-ylim_left
		self.ax.set_xlim(xlim_left-self.zoom_factor*xlim_size,xlim_right+self.zoom_factor*xlim_size)
		self.ax.set_ylim(ylim_left-self.zoom_factor*ylim_size,ylim_right+self.zoom_factor*ylim_size)
		plt.draw()


    def on_key_release(self, event):
       if event.key == 'shift':
           self.shift_is_held = False
       if event.key == 'control':
           self.control_is_held = False

    def onClick(self, event):
        if event.button == 3:
            if self.shift_is_held: # add point
                print "adding point", event.xdata
                self.kid_idx = np.hstack((self.kid_idx,np.argmin(np.abs(self.chan_freqs-event.xdata))))
                self.kid_idx = self.kid_idx[np.argsort(self.kid_idx)]
                self.refresh_plot()
            elif self.control_is_held: #delete point
                print "removing point", event.xdata
    	        delete_index = np.argmin(np.abs(self.chan_freqs[self.kid_idx]-event.xdata))
                self.kid_idx = np.delete(self.kid_idx,delete_index)
                self.refresh_plot()
                #self.delete_list.append(event.xdata)
                #plt.plot(event.xdata,event.ydata,"x",markersize = 20,mew = 5)
            else:
                print "please hold either the shift or control key while right clicking to add or remove points"

    def refresh_plot(self):
        self.p1.set_data(self.chan_freqs[self.kid_idx],self.data[self.kid_idx])
        for i in range(0,self.kid_idx_len):
            self.text_dict[i].set_text("")# clear all of the texts
        self.text_dict = {}
        for i in range(0,len(self.kid_idx)):
            self.text_dict[i] = plt.text(self.chan_freqs[self.kid_idx][i], self.data[self.kid_idx][i], str(i))
        self.kid_idx_len = len(self.kid_idx)
        plt.draw()


class interactive_threshold_plot(object):

    def __init__(self,chan_freqs,data,peak_threshold):
        self.peak_threshold = peak_threshold
        self.chan_freqs = chan_freqs
        self.data = data
        self.fig = plt.figure(2,figsize = (16,6))
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.l1, = self.ax.plot(self.chan_freqs,self.data)
        self.ilo = np.where( self.data < -1.0*self.peak_threshold)[0]     
        self.p1, = self.ax.plot(self.chan_freqs[self.ilo],self.data[self.ilo],"r*")            
        print "Press up or down to change the threshold by 0.1 dB or press t to enter a custom threshold value."
        print "Close all plots when finished"
        plt.xlabel('frequency (MHz)')
        plt.ylabel('dB')
        self.ax.set_title("Peak Threshold "+str(self.peak_threshold))
        plt.show(block = True)
        
    def on_key_press(self, event):
        #print event.key
	#has to be shift and ctrl because remote viewers only forward
	#certain key combinations
	#print event.key == 'd'
        if event.key == 'up':
           self.peak_threshold = self.peak_threshold +0.1
           self.refresh_plot()
        if event.key == 'down':
           self.peak_threshold = self.peak_threshold - 0.1
           self.refresh_plot()
        if event.key == 't':
           self.peak_threshold = np.float(input("What threshold would you like in dB? "))
           self.refresh_plot()


    def refresh_plot(self):
        self.ilo = np.where( self.data < -1.0*self.peak_threshold)[0] 
        self.p1.set_data(self.chan_freqs[self.ilo],self.data[self.ilo])
        self.ax.set_title("Peak Threshold "+str(self.peak_threshold))
        plt.draw()


                
#bb_freqs = np.load(os.path.join(path,'bb_freqs.npy'))
#lo_freqs = np.load(os.path.join(path,'sweep_freqs.npy'))
accum_len = 2**19
def openStored(path):
	files = sorted(os.listdir(path))
	I_list = [os.path.join(path, filename) for filename in files if filename.startswith('I')]
	Q_list = [os.path.join(path, filename) for filename in files if filename.startswith('Q')]
	chan_I = np.array([np.load(filename) for filename in I_list])
	chan_Q = np.array([np.load(filename) for filename in Q_list])
	return chan_I, chan_Q

def compute_dI_and_dQ(I,Q,freq=None,filterstr='SG',do_deriv=True):
	#Given I,Q,freq arrays
	#input filterstr = 'SG' for sav-gol filter with builtin gradient, 'SGgrad' savgol then apply gradient to filtered
	#do_deriv: if want to look at filtered non differentiated data
    if freq==None:
        df=1.0
    else:
        df = freq[1]-freq[0]
    dI=filtered_differential(I, df,filtertype=filterstr,do_deriv=do_deriv)
    dQ=filtered_differential(Q, df,filtertype=filterstr,do_deriv=do_deriv)
    return dI,dQ
   
def filtered_differential(data,df,filtertype=None,do_deriv=True):
    '''take 1d array data with spacing df. return filtered version of data depending on filterrype'''
    if filtertype==None:
        out = np.gradient(data,df)
    window=13; n=3
    if filtertype=='SG':
        if do_deriv==True:  
            out = savgol_filter(data, window, n, deriv=1, delta=df)
        else:
            out = savgol_filter(data, window, n, deriv=0, delta=df)
    if filtertype=='SGgrad':    
        tobegrad = savgol_filter(data, window, n)
        out = np.gradient(tobegrad,df)
    return out

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

def filter_trace(path, bb_freqs, sweep_freqs):
    chan_I, chan_Q = openStoredSweep(path)
    channels = np.arange(np.shape(chan_I)[1])
    mag = np.zeros((len(bb_freqs),len(sweep_freqs)))
    chan_freqs = np.zeros((len(bb_freqs),len(sweep_freqs)))
    for chan in channels:
    	mag[chan] = (np.sqrt(chan_I[:,chan]**2 + chan_Q[:,chan]**2))
    	chan_freqs[chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
    #mag = np.concatenate((mag[len(mag)/2:], mag[0:len(mag)/2]))
    mags = 20*np.log10(mag/np.max(mag))
    mags = np.hstack(mags)
    #chan_freqs = np.concatenate((chan_freqs[len(chan_freqs)/2:],chan_freqs[0:len(chan_freqs)/2]))
    chan_freqs = np.hstack(chan_freqs)
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

def main(path, center_freq, sweep_step, smoothing_scale, peak_threshold, spacing_threshold):
    bb_freqs = np.load(os.path.join(path,'bb_freqs.npy'))
    lo_freqs = np.load(os.path.join(path,'sweep_freqs.npy'))
    
    chan_freqs,mags = filter_trace(path, bb_freqs, lo_freqs)
    filtermags = lowpass_cosine( mags, sweep_step, 1./smoothing_scale, 0.1 * (1.0/smoothing_scale))
    #plt.ion()
    plt.figure(1)
    #plt.clf()
    plt.plot(chan_freqs,mags,'b',label='#nofilter')
    plt.plot(chan_freqs,filtermags,'g',label='Filtered')
    plt.xlabel('frequency (MHz)')
    plt.ylabel('dB')
    plt.legend()
    
    #plt.figure(2)
    #plt.clf()
    #plt.plot(chan_freqs,mags-filtermags,'b')
    ipt = interactive_threshold_plot(chan_freqs,(mags-filtermags),peak_threshold)
    peak_threshold = ipt.peak_threshold
    ilo = np.where( (mags-filtermags) < -1.0*peak_threshold)[0]
    #plt.plot(chan_freqs[ilo],mags[ilo]-filtermags[ilo],'r*')
    #plt.xlabel('frequency (MHz)')
    
    iup = np.where( (mags-filtermags) > -1.0*peak_threshold)[0]
    new_mags = mags - filtermags
    new_mags[iup] = 0
    labeled_image, num_objects = ndimage.label(new_mags)
    indices = ndimage.measurements.minimum_position(new_mags,labeled_image,np.arange(num_objects)+1)
    kid_idx = np.array(indices, dtype = 'int')
    
    print len(kid_idx)
    print kid_idx
    del_idx = []
    for i in range(len(kid_idx) - 1):
    	spacing = (chan_freqs[kid_idx[i + 1]] - chan_freqs[kid_idx[i]]) * 1.0e3
    	if (spacing < spacing_threshold):
    		print spacing, spacing_threshold
    		print "Spacing conflict"
    		if (new_mags[kid_idx[i + 1]] < new_mags[kid_idx[i]]):
    			del_idx.append(i)
    		else: 
    			del_idx.append(i + 1)
    
    del_idx = np.array(del_idx) 
    print "Removing " + str(len(del_idx)) + " KIDs"
    print
    kid_idx = np.delete(kid_idx, del_idx)
    
    del_again = []
    for i in range(len(kid_idx) - 1):
    	spacing = (chan_freqs[kid_idx[i + 1]] - chan_freqs[kid_idx[i]]) * 1.0e3
    	if (spacing < spacing_threshold):
    		print "Spacing conflict"
    		print spacing, spacing_threshold
    		if (new_mags[kid_idx[i + 1]] < new_mags[kid_idx[i]]):
    			del_again.append(i)
    		else: 
    			del_again.append(i + 1)
    
    del_again = np.array(del_again) 
    print "Removing " + str(len(del_again)) + " KIDs"
    print
    kid_idx = np.delete(kid_idx, del_again)
    
    #fig = plt.figure(4,figsize = (18,6))
    #plt.clf()
    #ax = fig.add_subplot(111)
    # Changed this part up to be an interactive plot that lets you
    # delete and add resonators	
    ip = interactive_plot_2(chan_freqs,(mags-filtermags),kid_idx)
    #plt.plot(chan_freqs, mags-filtermags,'b')
    #plt.plot(chan_freqs[kid_idx], (mags-filtermags)[kid_idx], 'r*')
    print len(kid_idx)
    #for i in range(0,len(kid_idx)):
    #	plt.text(chan_freqs[kid_idx][i], (mags-filtermags)[kid_idx][i], str(i))
    # list of kid frequencies
    #print "remove list", ip.delete_list
    #print "add list", ip.add_list
    kid_idx = ip.kid_idx
    
    '''
    # delete the manualy deleted reesonators
    delete_index = []
    for i in range(0,len(ip.delete_list)):
    	delete_index.append(np.argmin(np.abs(chan_freqs[kid_idx]-ip.delete_list[i])))
    #print delete_index
    kid_idx = np.delete(kid_idx,delete_index)
    # add in the manually added index
    add_index = []
    for i in range(0,len(ip.add_list)):
    	add_index.append(np.argmin(np.abs(chan_freqs-ip.add_list[i])))
    #print add_index
    if len(add_index)>0:
    	kid_idx = np.hstack((kid_idx,add_index))
    kid_idx = kid_idx[np.argsort(kid_idx)]
    '''
    #print kid_idx
    plt.figure(5,figsize = (16,6))
    plt.plot(chan_freqs, mags-filtermags,'b')
    plt.plot(chan_freqs[kid_idx], (mags-filtermags)[kid_idx], 'r*')
    print len(kid_idx)
    for i in range(0,len(kid_idx)):
    	plt.text(chan_freqs[kid_idx][i], (mags-filtermags)[kid_idx][i], str(i))
    plt.xlabel('frequency (MHz)')
    plt.ylabel('dB')
    rf_target_freqs = chan_freqs[kid_idx]
    bb_target_freqs = ((rf_target_freqs*1.0e6) - center_freq*1.0e6)
    print bb_target_freqs
    bb_target_freqs = np.roll(bb_target_freqs, - np.argmin(np.abs(bb_target_freqs)) - 1)
    print len(rf_target_freqs), "pixels found"
    print "Freqs =", chan_freqs[kid_idx]
    np.save(path + '/bb_targ_freqs.npy', bb_target_freqs)
    return
