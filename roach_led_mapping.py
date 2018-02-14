import kidPy

"""Upload the system and initialize the registers and UDP socket.
Only do this once per measurement, or if there are connection/data issues"""
kidPy.systemInit()

"""Write the tone comb specified in general_config"""
kidPy.writeVnaComb()

"""Do a VNA sweep"""
kidPy.vnaSweepConsole()

"""Plot the last VNA sweep
To plot an arbitrary sweep, do:
kidPy.plotVNASweep('path_to_data')"""
#kidPy.plotLastVNASweep()

"""Save the I/Q timestreams in the subfolder 'test',
for channels 0 - 100, for 60 seconds"""
kidPy.saveTimestreamDirfile('test', 0, 100, 60)

"""Plot the PSDs from the file saved above
To plot from a different file, add the path as an argument in quotations,
#Instead of None"""
#kidPy.allPSD(None)
