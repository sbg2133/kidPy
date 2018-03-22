import numpy as np
from kidPy import openStoredSweep
import os

# script for retuning the kids either to the minimum of the resonance
# or to the max seperation in the iq loop (best place for streaming noise)
# To Do add look around points option so that you can say only look for the min
# in the nearest 20 poins or so

filename = str(np.load("last_targ_dir.npy"))#"../data/targ/1521158106-Mar-15-2018-17-55-06.dir"

# reads in an iq sweep and stores i, q, and the frequencies in a dictionary
def read_iq_sweep(filename):
	I, Q = openStoredSweep(filename)
	sweep_freqs = np.load(filename + '/sweep_freqs.npy')
	bb_freqs = np.load(filename + '/bb_freqs.npy')
	channels = len(bb_freqs)
	mags = np.zeros((channels, len(sweep_freqs))) 
	chan_freqs = np.zeros((len(sweep_freqs),channels))
	for chan in range(channels):
        	chan_freqs[:,chan] = (sweep_freqs + bb_freqs[chan])/1.0e6
	iq_dict = {'I': I, 'Q': Q, 'freqs': chan_freqs}
	return iq_dict

def tune_kids(filename,ri,regs,fpga,min = True):
	iq_dict = read_iq_sweep(filename)
	if min: #fine the minimum
		print("centering on minimum")
		min_indx = np.argmin(iq_dict['I']**2+iq_dict['Q']**2,axis = 0)
		new_freqs = iq_dict['freqs'][(min_indx,np.arange(0,iq_dict['freqs'].shape[1]))]
	else: # find the max of dIdQ
		print("centering on max dIdQ")
		pos_offset_I = np.roll(iq_dict['I'],1,axis = 0)
		neg_offset_I = np.roll(iq_dict['I'],-1,axis = 0)
		pos_offset_Q = np.roll(iq_dict['Q'],1,axis = 0)
		neg_offset_Q = np.roll(iq_dict['Q'],-1,axis = 0)
		pos_dist= np.sqrt((iq_dict['I']-pos_offset_I)**2+(iq_dict['Q']-pos_offset_Q)**2)
		neg_dist= np.sqrt((iq_dict['I']-neg_offset_I)**2+(iq_dict['Q']-neg_offset_Q)**2)
		ave_dist = (pos_dist - neg_dist)/2.
		#zero out the last and first values
		ave_dist[0,:] = 0
		ave_dist[ave_dist.shape[0]-1,:] = 0
		new_freqs = iq_dict['freqs'][(np.argmax(ave_dist,axis =0),np.arange(0,iq_dict['freqs'].shape[1]))]
	new_bb_freqs = (new_freqs*10**6-ri.center_freq*10**6)
	if not np.size(ri.freq_comb):
		try:
			ri.freq_comb = np.load("last_freq_comb.npy")
		except IOError:
			print "\nFirst need to write a frequency comb with length > 1"
			return
	#new_bb_freqs = np.roll(new_bb_freqs,-np.argmin(new_bb_freqs)-1)
	ri.freq_comb = new_bb_freqs
	print("Old frequencys -> New frequencies")
	for i in range(0,iq_dict['freqs'].shape[1]):
		print(str(iq_dict['freqs'][iq_dict['freqs'].shape[0]/2,i])[0:7]+"->"+str(new_freqs[i])[0:7])
	print("writing qdr")
	#ri.writeQDR(ri.freq_comb)
	try:
		ri.writeQDR(new_bb_freqs,transfunc = True,transfunc_filename = "last_transfunc.npy")
		fpga.write_int(regs[np.where(regs == 'write_comb_len_reg')[0][0]][1], len(ri.freq_comb))
	except:
		ri.writeQDR(new_bb_freqs)
		print("WARNING Tranfer function was not applied")
	vna_savepath = str(np.load("last_vna_dir.npy"))
	np.save(vna_savepath + '/bb_targ_freqs.npy', new_bb_freqs)




