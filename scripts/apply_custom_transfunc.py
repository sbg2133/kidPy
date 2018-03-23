import numpy as np
# this script applys a tranfer function specified by a filename
# this could for instance be the output of analyze_power_sweep.py

transfunc_filename = '../transfuncs/trans_func_1521733375-Mar-22-2018-09-42-55.dir.npy'
#transfunc_filename = 'last_transfunc.npy'

if not np.size(ri.freq_comb): #if kidpy exits and it can't remember what freqcomb is
	try:
		ri.freq_comb = np.load("last_freq_comb.npy")
	except IOError:
		print "\nFirst need to write a frequency comb with length > 1"

#transfunc = np.linspace(1,4,len(ri.freq_comb)) # a test transfer function
transfunc = np.load(transfunc_filename)
for i in range(0,len(ri.freq_comb)):
	print(ri.freq_comb[i],transfunc[i])
#np.save("transfunc_custom",transfunc) # save test transfer function to file

np.save("last_transfunc",transfunc) #save it as the last used transfunc
if len(transfunc) != len(ri.freq_comb):
	print("Your transfer fuction length does not match the length of the tone comb")
	print("Try writing found freqs again")

#ri.writeQDR(ri.freq_comb, transfunc = True, transfunc_filename = "transfunc_custom.npy") # load the test transfer function
ri.writeQDR(ri.freq_comb, transfunc = True, transfunc_filename = transfunc_filename)
fpga.write_int(regs[np.where(regs == 'write_comb_len_reg')[0][0]][1], len(ri.freq_comb))

print("finished")
