import matplotlib.pyplot as plt
import numpy as np

class interactive_plot(object):

    def __init__(self,Is,Qs,chan_freqs):
	self.Is = Is
	self.Qs = Qs
	self.chan_freqs = chan_freqs
        self.fig = plt.figure(1,figsize = (13,6))
        self.ax = self.fig.add_subplot(121)
	self.ax.set_ylabel("Power (dB)")
	self.ax.set_xlabel("Frequecy (MHz)")
        self.ax2 = self.fig.add_subplot(122)
	self.ax2.set_ylabel("Q")
	self.ax2.set_xlabel("I")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        #self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        #self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        self.plot_index = 0
	self.l1, = self.ax.plot(chan_freqs[self.plot_index,:],10*np.log10(Is[:,self.plot_index]**2+Qs[:,self.plot_index]**2),'o')
	self.l2, = self.ax2.plot(Is[:,self.plot_index],Qs[:,self.plot_index],'o')
	self.ax.set_title("Resonator Index "+str(self.plot_index))
	print("")
        print("Interactive Target Sweep Plotting Activated")
        print("Use left and right arrows to switch between resonators")
	plt.show(block = True)

    def refresh_plot(self):
        self.l1.set_data(self.chan_freqs[self.plot_index,:],10*np.log10(self.Is[:,self.plot_index]**2+self.Qs[:,self.plot_index]**2))
        self.ax.relim()
        self.ax.autoscale()
        self.ax.set_title("Resonator Index "+str(self.plot_index))
	self.l2.set_data(self.Is[:,self.plot_index],self.Qs[:,self.plot_index])
        self.ax2.relim()
        self.ax2.autoscale()
        plt.draw()
        
    def on_key_press(self, event):
        #print event.key
        if event.key == 'right':
           if self.plot_index != self.chan_freqs.shape[0]-1:
    		self.plot_index = self.plot_index +1
		self.refresh_plot()

        if event.key == 'left':
           if self.plot_index != 0:
    		self.plot_index = self.plot_index -1
		self.refresh_plot()


