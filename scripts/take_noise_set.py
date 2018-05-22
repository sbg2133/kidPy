
targetSweep(ri, udp, valon,span = 2.0e6,lo_step = 2.0e6/50.)
gain_name = str(np.load("last_targ_dir.npy"))
#plotTargSweep(gain_name)
#fine scan
targetSweep(ri, udp, valon,span = (50.*1000),lo_step = 0.5e3)
fine_name = str(np.load("last_targ_dir.npy"))
#plotTargSweep(fine_name)

#plt.close('all')
print("The fine scan is: " + fine_name)
print("The gain scan is: " + gain_name)
file_name = "noise_"+time.strftime("%Y%m%d_%H%M%S")

udp.saveDirfile_chanRangeIQ(600,start_chan = 0,end_chan = 165,sub_folder = file_name)
