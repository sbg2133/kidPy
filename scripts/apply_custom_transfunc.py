import numpy as np

transfunc = np.linspace(1,4,len(ri.freq_comb))
np.save("transfunc_custom",transfunc)

ri.writeQDR(ri.freq_comb, transfunc = True, transfunc_filename = 'transfunc_custom.npy')
fpga.write_int(regs[np.where(regs == 'write_comb_len_reg')[0][0]][1], len(ri.freq_comb))
