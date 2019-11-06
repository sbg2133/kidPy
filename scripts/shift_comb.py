import numpy as np
# tihs script can be useful if you are doing temperature sweeps
# and all of the resonators shift by the same df/f

if not np.size(ri.freq_comb):
    try:
        ri.freq_comb = np.load("last_freq_comb.npy")
    except IOError:
        print "\nFirst need to write a frequency comb with length > 1"

center_freq = np.float(gc[np.where(gc == 'center_freq')[0][0]][1])

#print(center_freq)
#print(ri.freq_comb)
#print((center_freq*10**6+ri.freq_comb)/10**6)

df_over_f = input("Enter df/f to shift comb by use negaitve nubmers to shift to lower frequencies\n")

#print(df_over_f)

#convert df/f to frequency shift for each resonator
shift = df_over_f*(center_freq*10**6+ri.freq_comb)
#print(shift)
new_bb_freqs = ri.freq_comb+shift
ri.freq_comb = new_bb_freqs
print("writing qdr")
try:
    ri.writeQDR(new_bb_freqs,transfunc = True,transfunc_filename = "last_transfunc.npy")
    fpga.write_int(regs[np.where(regs == 'write_comb_len_reg')[0][0]][1], len(ri.freq_comb))
except:
    ri.writeQDR(new_bb_freqs)
    print("WARNING Tranfer function was not applied")
vna_savepath = str(np.load("last_vna_dir.npy"))
np.save(vna_savepath + '/bb_targ_freqs.npy', new_bb_freqs)
