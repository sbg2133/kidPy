# this script takes a fine scan and gain scan combo
#then analyzes it if you like
# below module found at https://github.com/GlennGroupUCB/submm_python_routines.git
from multitone_kidPy import analyze 
fit_scans = True

print("Running fine scan and gain scan combo")

#gain scan 
#run gain scan first so you don't acidentially recenter
#on the gain scan afterwords instead of the fine scan
targetSweep(ri, udp, valon,span = 1.26e6,lo_step = 1.26e6/90.)
gain_name = str(np.load("last_targ_dir.npy"))
plotTargSweep(gain_name)
#fine scan
targetSweep(ri, udp, valon,span = (100*.4*1000),lo_step = .4e3)
fine_name = str(np.load("last_targ_dir.npy"))
plotTargSweep(fine_name)


plt.close('all')
print("The fine scan is: " + fine_name)
print("The gain scan is: " + gain_name)

if fit_scans:
	plt.ioff() # this causes some error to print to screen but it is better that figures continuously poping up
	analyze.fit_fine_gain(fine_name,gain_name)
	plt.ion()







