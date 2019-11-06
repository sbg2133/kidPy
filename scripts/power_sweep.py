# this is a script for doing a power sweep so that you can find bifurcation powers
# currently uses a lab brick as the attenuator
# should try to make it compatiable with a rudat attenuators as well
# or I should just buy a rudat

#before starting center tones on greatest attenutaiton
from datetime import datetime
import subprocess
import sys
import os

analyze_on_the_go = True #will use subproccess to start analyzing the data as soon as you take it
# for Pete becuase he likes things to be fast

from lab_brick import core #this module can be found https://github.com/GlennGroupUCB/submm_python_routines.git
power_sweep_dir = "../../data/kidpy_multitone/power_sweeps/"

#you will need to edit the last number below to be the serial number of your lab brick
if atten is None:
    attn = core.Attenuator(0x041f,0x1208,"01784")
    disconnect = True
else:
    attn = atten.instr
    disconnect = False
max_attn = 25.
min_attn = 10.
attn_step = 2.5
n_attn_levels = np.int((max_attn-min_attn)/attn_step)+1

attn_levels = np.linspace(max_attn,min_attn,n_attn_levels)

fine_step = 1e3
fine_span = 250e3
gain_span = 2.0e6
gain_step = gain_span/50.

fine_names = []
gain_names = []

for i in range(0,n_attn_levels):
	print("starting scan "+str(i+1))
	attn.set_attenuation(attn_levels[i]) #change power level
	#take gain scan
	targetSweep(ri, udp, valon,span = gain_span,lo_step = gain_step)
	gain_name = str(np.load("last_targ_dir.npy")) 
	gain_names.append(os.path.abspath(gain_name))
	#take fine scan
	targetSweep(ri, udp, valon,span = fine_span,lo_step = fine_step)
	fine_name = str(np.load("last_targ_dir.npy"))
	fine_names.append(os.path.abspath(fine_name))
	#here you might consider recentering tones
	if analyze_on_the_go: #analyze on the fly to make things faster
		p = subprocess.Popen([sys.executable, './scripts/analyze_fine_gain.py',str(fine_name),str(gain_name)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	
if disconnect:
    attn.connected = False
time_str = str(datetime.now())[0:10]+str(datetime.now())[11:19]
np.save(power_sweep_dir+ time_str+"_fine_names",fine_names)
np.save(power_sweep_dir+ time_str+"_gain_names",gain_names)
np.save(power_sweep_dir+ time_str+"_attn_levels",attn_levels)
#save the names of the fine and gain scans for analyze_power_sweep.py



