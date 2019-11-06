# this script takes a fine scan and gain scan combo
#then analyzes it if you like
# below module found at https://github.com/GlennGroupUCB/submm_python_routines.git
import subprocess
import sys
from multitone_kidPy import analyze 
fit_scans = True

print("Running fine scan and gain scan combo")

gain_span = 1.0e6 #1e6
gain_pts = 50
fine_span = 1.0e5 #7.5e4
fine_pts = 100 #75
#gain scan 
#run gain scan first so you don't acidentially recenter
#on the gain scan afterwords instead of the fine scan
targetSweep(ri, udp, valon,span = gain_span,lo_step = gain_span/gain_pts)
gain_name = str(np.load("last_targ_dir.npy"))
#plotTargSweep(gain_name)
#fine scan
targetSweep(ri, udp, valon,span = fine_span,lo_step = fine_span/fine_pts)
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







