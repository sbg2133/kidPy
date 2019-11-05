import sys
import numpy as np
import matplotlib.pyplot as plt
from multitone_kidPy import analyze

skip_beginning = 10 #skip first few streaming points

def main(fine_filename, gain_filename, stream_filename):
    cal_dict = analyze.calibrate_multi(fine_filename,gain_filename,
            stream_filename,skip_beginning = skip_beginning)
    psd_dict = analyze.noise_multi(cal_dict)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('fine scan, gain scan, and stream directories are required '
                'arguments, in that order')
        sys.exit(1)
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
