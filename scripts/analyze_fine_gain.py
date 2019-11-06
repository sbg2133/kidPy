from multitone_kidPy import analyze #founc at https://github.com/GlennGroupUCB/submm_python_routines.git
import sys
# this is just a script to be called by subprocess from power_sweep.py
analyze.fit_fine_gain_std(sys.argv[1],sys.argv[2])

