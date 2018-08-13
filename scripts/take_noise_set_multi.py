import tune_kids
import numpy as np


sets = 6
gain_names = []
fine_names = []
fine_names_2 = []
stream_names = []

for i in range(0,sets):
    print("Starting set "+ str(i))
    targetSweep(ri, udp, valon,span = 2.0e6,lo_step = 2.0e6/50.)
    gain_name = str(np.load("last_targ_dir.npy"))
    gain_names.append(gain_name)
    #plotTargSweep(gain_name)
    #fine scan
    targetSweep(ri, udp, valon,span = (30.*1000),lo_step = 0.3e3)
    fine_name = str(np.load("last_targ_dir.npy"))
    fine_names.append(fine_name)
    #plotTargSweep(fine_name)

    #plt.close('all')
    print("The fine scan is: " + fine_name)
    print("The gain scan is: " + gain_name)
    file_name = "noise_"+time.strftime("%Y%m%d_%H%M%S")
    stream_names.append(file_name)
    udp.saveDirfile_chanRangeIQ(600,start_chan = 0,end_chan = 165,sub_folder = file_name)
    targetSweep(ri, udp, valon,span = (30.*1000),lo_step = 0.3e3)
    fine_name_2 = str(np.load("last_targ_dir.npy"))
    fine_names_2.append(fine_name_2)
    #recenter the tones
    tune_kids.tune_kids(fine_name_2,ri,regs,fpga,find_min = False,interactive = False,look_around = 10)

file_prefix = "../../noise_sets/"
time_str = time.strftime("%Y%m%d_%H%M%S")
np.save(file_prefix+"noise_fine_names"+time_str,fine_names)
np.save(file_prefix+"noise_gain_names"+time_str,gain_names)
np.save(file_prefix+"noise_fine_2_names"+time_str,fine_names_2)
np.save(file_prefix+"noise_stream_names"+time_str,stream_names)
                      
        
