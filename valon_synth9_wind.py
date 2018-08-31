"""This is a wrapper around the windfreaksynth to expose an interface
 similar to the valon_synth9.
"""

import windfreaksynth as wfs

class Synthesizer(wfs.SynthHDDevice):
    def __init__(self, port):
        wfs.SynthHDDevice.__init__(self)
	self.setReferenceSelect(1)
	print(self.conn)
        for channel in [0, 1]:
            self.setControlChannel(channel)
            self.setAMRunContinuously(0)
            self.setFMContinuousMode(0)
            self.setRFAmpOn(True)
            self.setPLLPowerOn(True)
    
    def set_rf_level(self, channel, dBm):
        #valon has channels 1 and 2, where windfreak is 0 and 1, thus the channel-1
        self.setControlChannel(channel - 1)
        self.setPower(dBm)

    def set_frequency(self, channel, center_freq):
        #valon uses MHz and windfreak Hz
        self.setControlChannel(channel - 1)
        self.setFrequency(center_freq * 1e6)

    def set_reference(self, ref_freq):
        self.setPLLReferenceFrequency(ref_freq * 1e6)
    
    def set_ref_select(self, value):
        if value == 0:
            self.setReferenceSelect(1) #Use internal reference
        elif value == 1:
            self.setReferenceSelect(0) #Use external ref
        else:
            raise ValueError('Not Valid Reference Value!')

    def set_refdoubler(self, channel, value):
        pass #Feature not available in Windfreak

    def set_pfd(self, channel, value):
        pass #I think there is not an equivalent for the windfreak
