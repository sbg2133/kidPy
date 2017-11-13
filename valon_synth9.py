
"""
Provides a serial interface to the Valon 5009.
"""

# Python modules
import serial
from time import sleep, time

__author__ = "Brad Dober"
__license__ = "GPL"
__version__ = "1.0"

# Handy aliases
SYNTH_A = 1
SYNTH_B = 2

INT_REF = 0
EXT_REF = 1

class Synthesizer:
    """A simple interface to the Valon 5009 synthesizer."""
    def __init__(self, port):
        self.conn = serial.Serial(None, 9600, serial.EIGHTBITS,
                                  serial.PARITY_NONE, serial.STOPBITS_ONE, timeout = 0.500)
        self.conn.setPort(port)

    def get_frequency(self, synth):
        """
        Returns the current output frequency for the selected synthesizer.

        @param synth : synthesizer this command affects (1 or 2).
        @type  synth : int

        @return: the frequency in MHz (float)
        """
        self.conn.open()
        
        data = 'f'+str(synth)+'?\r'
        self.conn.write(data)
        self.conn.flush()
        t = time()
	data = ''
        while (len(data) < 40 and time()-t < 1): data += self.conn.read(40)
	self.conn.close()
        data = data.split(' Act ')[1]
        data = data.split(' ')[0]
        return float(data) #in MHz#

    def set_frequency(self, synth, freq, chan_spacing = 10.):
        """
        Sets the synthesizer to the desired frequency

        Sets to the closest possible frequency, depending on the channel spacing.
        Range is determined by the minimum and maximum VCO frequency.

        @param synth : synthesizer this command affects (1 or 2).
        @type  synth : int

        @param freq : output frequency
        @type  freq : float

        @param chan_spacing : deprecated
        @type  chan_spacing : float

        @return: True if success (bool)
        """
        self.conn.open()
        data = 's'+str(synth)+';f'+str(freq)+'\r'
        self.conn.write(data)
        self.conn.flush()
        self.conn.flush()
        t = time()
	data = ''
        while (len(data) < 40 and time()-t < 1): data += self.conn.read(40)
        self.conn.close()
        data = data.split(' Act ')[1]
        data = data.split(' ')[0]
        return float(data)==float(freq)

    def get_reference(self):
        """
        Get reference frequency in MHz
        """
        self.conn.open()
        data = 'REF?\r'
        self.conn.write(data)
        self.conn.flush()
        data = self.conn.read(100)
        self.conn.close()
        #print data
        freq=data.split(' ')[1]
        return float(freq) #in MHz

    def set_reference(self, freq):
        """
        Set reference frequency in MHz

        @param freq : frequency in MHz
        @type  freq : float

        @return: True if success (bool)
        """
        self.conn.open()
        data = 'REF '+str(freq)+'M\r'
        self.conn.write(data)
        self.conn.flush()
        data = self.conn.read(100)
        self.conn.close()
        #print data
        ack=data.split(' ')[2]
        #print ack
        return ack == str(freq)
        
    def set_refdoubler(self, synth, enable):
        """
        Set reference doubler
        
        @param synth : synthesizer this command affects (1 or 2).
        @type  synth : int

        @param enable : turn on or off the reference doubler
        @type  enable : bool

        @return: True if success (bool)
        """
        if enable:
            self.conn.open()
            data = 's'+str(synth)+';REFDB E\r'
            self.conn.write(data)
            self.conn.flush()
        else:
            self.conn.open()
            data = 's'+str(synth)+';REFDB D\r'
            self.conn.write(data)            
            self.conn.flush()
        data = self.conn.read(100)
        self.conn.close()
        #print data
        ack=data.split(' ')[2]
        ack=ack.split(';')[0]
        return int(ack) == 0
        
    def get_refdoubler(self, synth):
        """
        Get reference doubler
        
        @param synth : synthesizer this command affects (1 or 2).
        @type  synth : int

        @return: True if on, False if off (bool)
        """
        
        self.conn.open()
        data = 'REFDB'+str(synth)+'?\r'
        self.conn.write(data)            
        self.conn.flush()
        data = self.conn.read(100)
        self.conn.close()
        #print data
        ack=data.split('REFDB')[2]
        ack=ack.split(';')[0]
        return int(ack)

    def get_rf_level(self, synth):
        """
        Returns RF level in dBm

        @param synth : synthesizer address, 1 or 2
        @type  synth : int

        @return: dBm (int)
        """
        self.conn.open()
        data = 's'+str(synth)+';Att?\r'
        self.conn.write(data)
        self.conn.flush()
        data = self.conn.read(100)
        self.conn.close()
        #print data
        data=data.split('ATT ')[1]
        data=data.split(';')[0]
        rf_level=float(data)-15
        return rf_level

    def set_rf_level(self, synth, rf_level):
        """
        Set RF level

        @param synth : synthesizer address, 1 or 2
        @type  synth : int

        @param rf_level : RF power in dBm
        @type  rf_level : float

        @return: True if success (bool)
        """
        "15 dB is equal to 0 dB ouput power"
        "can be set from 0 (+15) to 31.5 (-16.5)"
        if -16.5<=rf_level<=15:
            atten=-rf_level+15
            #print atten
            data='s'+str(synth)+';att'+str(atten)+'\r'
            self.conn.open()            
            self.conn.write(data)
            self.conn.flush()
            data = self.conn.read(1000)
            self.conn.close()
	    ack=data.split('ATT ')[1]
            ack=ack.split(';')[0]
            #print ack
            return float(ack) == float(atten)
        else:
            return False
            
    def set_pfd(self, synth, freq):
        """
        Sets the synthesizer's phase/frequency detector to the desired frequency

        @param synth : synthesizer this command affects (1 or 2).
        @type  synth : int

        @param freq : pfd frequency
        @type  freq : float

        @return: True if success (bool)
        """
        self.conn.open()
        data = 's'+str(synth)+';PFD'+str(freq)+'M\r'
        self.conn.write(data)
        self.conn.flush()
        data=self.conn.read(100)
        self.conn.close()
        #print data
        data = data.split('PFD ')[1]
	#print data
        data = data.split(' MHz')[0]
	#print data
        return int(data)==int(freq)
    
    def get_pfd(self, synth):
        """
        Gets the synthesizer's phase/frequency detector to the desired frequency

        @param synth : synthesizer this command affects (1 or 2).
        @type  synth : int

        @return: True if success (bool)
        """
        self.conn.open()
        data = 'PFD'+str(synth)+'?\r'
        self.conn.write(data)
        self.conn.flush()
        data=self.conn.read(100)
        self.conn.close()
        #print data
        data = data.split('PFD ')[1]
        data = data.split(' MHz')[0]
        return float(data)    

    def get_ref_select(self):
        """
        Returns the currently selected reference clock.
        
        Returns 1 if the external reference is selected, 0 otherwise.
        """
        self.conn.open()
        data = 'REFS?\r'
        self.conn.write(data)
        self.conn.flush()
        data = self.conn.read(100)
        self.conn.close()
        data=data.split(' ')[1]
        data=data.split(';')[0]
        return int(data)

    def set_ref_select(self, e_not_i = 1):
        """
        Selects either internal or external reference clock.

        @param e_not_i : 1 (external) or 0 (internal); default 1
        @type  e_not_i : int

        @return: True if success (bool)
        """
        data='REFS'+str(e_not_i)+'\r'
        self.conn.open()
        self.conn.write(data)
        self.conn.flush()
        data = self.conn.read(100)
        self.conn.close()
        ack=data.split(' ')[1]
        ack=ack.split(';')[0]
        return ack == str(e_not_i)

    def get_vco_range(self, synth):
        """
        Returns (min, max) VCO range tuple.

        @param synth : synthesizer base address
        @type  synth : int

        @return: min,max in MHz
        """
        self.conn.open()
        data = struct.pack('>B', 0x83 | synth)
        self.conn.write(data)
        self.conn.flush()
        data = self.conn.read(4)
        checksum = self.conn.read(1)
        self.conn.close()
        #_verify_checksum(data, checksum)
        return struct.unpack('>HH', data)

    def set_vco_range(self, synth, low, high):
        """
        Sets VCO range.

        @param synth : synthesizer base address
        @type  synth : int

        @param min : minimum VCO frequency
        @type  min : int

        @param max : maximum VCO frequency
        @type  max : int

        @return: True if success (bool)
        """
        self.conn.open()
        data = struct.pack('>BHH', 0x03 | synth, low, high)
        checksum = _generate_checksum(data)
        self.conn.write(data + checksum)
        self.conn.flush()
        data = self.conn.read(1)
        self.conn.close()
        ack = struct.unpack('>B', data)[0]
        return ack == ACK

    def get_phase_lock(self, synth):
        """
        Get phase lock status

        @param synth : synthesizer base address
        @type  synth : int

        @return: True if locked (bool)
        """
        self.conn.open()
        data = "LOCK"+str(synth)+"\r"
        self.conn.write(data)
        self.conn.flush()
        data = self.conn.read(100)
        self.conn.close()
        if "locked" in data:
            return True
        else:
            return False

    def flash(self):
        """
        Flash current settings for both synthesizers into non-volatile memory.

        @return: True if success (bool)
        """
        self.conn.open()
        data = 'SAV\r'
        self.conn.write(data)
        self.conn.flush()
        ack = self.conn.read(100)
        self.conn.close()
        if "No" in ack:
            return False
        else:
            return True
    
    def reset(self):
        """
        Resets the Valon to factory settings
        
        @return: True if success (bool)
        """
        self.conn.open()
        data = 'RST\r'
        self.conn.write(data)
        self.conn.flush()
        data = self.conn.read(100)
        self.conn.close()
        print data
        return True
