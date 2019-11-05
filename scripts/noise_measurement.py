# this script takes a fine scan and gain scan for calibration, then a time
#stream for a noise PSD, then analyzes it if you like
# below module found at https://github.com/GlennGroupUCB/submm_python_routines.git
import subprocess
import sys
import os
from multitone_kidPy import analyze 
run_analysis = True

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

#start time stream
time_interval = input('Time interval (s) ? ')
try:
    #udp.saveDirfile_chanRange(time_interval)
    stream_name = udp.saveDirfile_chanRangeIQ(time_interval)
    #udp.saveDirfile_adcIQ(time_interval)
except KeyboardInterrupt:
    stream_name = None
    pass

plt.close('all')
fine_name = os.path.abspath(fine_name)
gain_name = os.path.abspath(gain_name)
print("The fine scan is: " + fine_name)
print("The gain scan is: " + gain_name)
print("The stream file is : " + stream_name)
stream_dir = stream_name + '/../' #running process in this
    #directory so the results and plots are saved there
cwd = os.getcwd()

if run_analysis:
    plt.ioff() #Prevents plot popups as this runs in the background
    p = subprocess.Popen([sys.executable, cwd + '/scripts/analyze_noise.py',
        str(fine_name),str(gain_name), str(stream_name)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=stream_dir)
    plt.ion()

