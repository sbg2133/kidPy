# this script takes a fine scan and gain scan combo
#then analyzes it if you like
# below module found at https://github.com/GlennGroupUCB/submm_python_routines.git
import subprocess
import sys
from multitone_kidPy import analyze 
fit_scans = True

print("Running fine scan and gain scan combo")

#gain scan 
#run gain scan first so you don't acidentially recenter
#on the gain scan afterwords instead of the fine scan
targetSweep(ri, udp, valon,span = 2.0e6,lo_step = 2.0e6/50.)
gain_name = str(np.load("last_targ_dir.npy"))
#plotTargSweep(gain_name)
#fine scan
targetSweep(ri, udp, valon,span = (50.*1000),lo_step = 0.5e3)
fine_name = str(np.load("last_targ_dir.npy"))
#plotTargSweep(fine_name)


plt.close('all')
print("The fine scan is: " + fine_name)
print("The gain scan is: " + gain_name)

if fit_scans:
	plt.ioff() # this causes some error to print to screen but it is better that figures continuously poping up
	#analyze.fit_fine_gain(fine_name,gain_name)
        p = subprocess.Popen([sys.executable, './scripts/analyze_fine_gain.py',str(fine_name),str(gain_name)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	plt.ion()







