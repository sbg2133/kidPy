import tune_kids
import numpy as np

#Tunes the frequencies of the kids to the minimum 
#of the resonator. Run target sweep before this

filename = str(np.load("last_targ_dir.npy"))
tune_kids.tune_kids(filename,ri,regs,fpga,interactive = True,look_around = 25)
