import pygetdata as gd
import socket as sock
import casperfpga
import numpy as np
import time
from PyQt4.QtCore import QTimer
from PyQt4.QtGui import QApplication
import sys
from gbeConfig import roachDownlink

# load general settings
gc = np.loadtxt("./general_config", dtype = "str")
# load list of firmware registers (note: must manually update for different versions)
regs = np.loadtxt("./firmware_registers", dtype = "str")
# load list of network parameters
network = np.loadtxt("./network_config", dtype = "str")

try:
    fpga = casperfpga.katcp_fpga.KatcpFpga(network[np.where(network == 'roach_ppc_ip')[0][0]][1], timeout = 120.)
except RuntimeError:
    fpga = None

# UDP socket
s = sock.socket(sock.AF_PACKET, sock.SOCK_RAW, sock.htons(3))
# GbE interface
udp = roachDownlink(fpga, regs, network, s, 488.2815)
udp.configSocket()

start_chan = input("Start chan # ? ")
end_chan = input("End chan # ? ")
chan_range = range(start_chan, end_chan + 1)

for Nrows in range(5, 20):
    try:
	chan_range = np.reshape(chan_range, (Nrows, -1))
	break
    except ValueError:
	Nrows += 1

Nrows = np.shape(chan_range)[0]
Ncols = np.shape(chan_range)[1]

print "Nrows =", Nrows
print "Ncols =", Ncols

log_dir = './data/array'

# make the filename
filename = log_dir + '/' + str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'

# open a dirfile for writing
df = gd.dirfile(filename,gd.CREAT|gd.RDWR|gd.UNENCODED)

# make a time field
df.add_spec('time RAW FLOAT64 1')
for i in xrange(Nrows):
    for j in xrange(Ncols):
	fieldname = 'r'+str(i).zfill(3)+'c'+str(j).zfill(3)
	df.add_spec(fieldname+' RAW FLOAT64 1')
df.flush()

# function to append data to the dirfile
def callback():
    try:
        #last_phase = np.zeros((Nrows, Ncols))
        phase = np.zeros((Nrows, Ncols))
        last_phase = np.zeros((Nrows, Ncols))
        packet, data, header, saddr = udp.parsePacketData()
        packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
	for chan in range(start_chan, end_chan + 1):
            i = np.where(chan_range == chan)[0][0]
            j = np.where(chan_range == chan)[1][0]
	    __, __, last_phase[i,j] = udp.parseChanData(chan, data)
        packet, data, header, saddr = udp.parsePacketData()
        packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
	for chan in range(start_chan, end_chan + 1):
            i = np.where(chan_range == chan)[0][0]
            j = np.where(chan_range == chan)[1][0]
	    __, __, phase[i,j] = udp.parseChanData(chan, data)
        for i in xrange(Nrows):
	    for j in xrange(Ncols):
		fieldname = 'r'+str(i).zfill(3)+'c'+str(j).zfill(3)
		df.putdata(fieldname,np.array([phase[i,j] - last_phase[i,j]]),gd.FLOAT64,callback.thisframe)
	df.putdata('time',np.array([time.time()]),gd.FLOAT64,callback.thisframe)
	callback.thisframe+=1
	df.flush()
    except TypeError:
        print "Stream error"
    except KeyboardInterrupt:
        sys.exit()

# frame counter
callback.thisframe = 0

# set up Qt
app = QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)

# make and start the timer
timer = QTimer()
timer.timeout.connect(callback)
timer.start(10)

app.exec_()
