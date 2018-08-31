# Windfreaktech SynthHD python controller
# Sam Rowe, Cardiff University, 2018

# NOTE: all frequencies should be given in Hz, and powers in dBm.

import serial
import serial.tools.list_ports
import time

#serial port snrs for device identification
hwid_snrs = ['4294967295']
dev_snrs = ['633']


class SynthHDDevice(object):
	def __init__(self, dev_snr=dev_snrs[0], open_connection=True):
		print 'Connecting to WFT SynthHD Synthesiser...'
		self.serialNumber = dev_snr
		self.FREQMIN      = 53e6
		self.FREQMAX      = 14000e6
		self.POWMIN       = -60.0
		self.POWMAX       = +20.0
		self.serialPort   = self._findSerialPort()
		self.conn         = serial.Serial(None, timeout=0.01)
		self.conn.setPort(self.serialPort)
		if open_connection:
			self._openConnection()
			self.getStatus()
		print 'OK :)'
		
	def _findSerialPort(self):
		comports = serial.tools.list_ports.comports()
		for port, desc, hwid in comports:
			if hwid.find(hwid_snrs[dev_snrs.index(self.serialNumber)])>0:
				return port
		print 'Error, device not found'
		return None
	
	def _openConnection(self):
		self.conn.open()
	
	def _closeConnection(self):
		self.conn.close()
		
	def _clearSerialBuffer(self):
		self.conn.readlines()
	
	def getStatus(self):
		self.status = 'Not implemented yet'
		return self.status
	
	def sendCommand(self,command,clearbuf=True,getResponse=True):
		if clearbuf: self._clearSerialBuffer()
		self.conn.write(command)
		if getResponse:
			readlines = self.conn.readlines()
			if len(readlines)==0:
				return
			elif len(readlines)==1:
				return readlines[0].strip()
			else:
				message = ''
				for i in readlines:
					message += i.strip()+'\n'
				return message


	def sendCommandFast(self,command,dwell=0.0):
		self.sendCommand(command,clearbuf=False,getResponse=False)
		time.sleep(dwell)

	def getHelp(self):
		print self.sendCommand('?')
	
	def getControlChannel(self):
		return self.sendCommand('C?')
	def setControlChannel(self,value):
		value = int(value)
		assert value in [0,1], 'value error'
		return self.sendCommand('C%d'%value)
	
	def getFrequency(self):
		return float(self.sendCommand('f?'))*1e6
	def setFrequency(self,value):
		assert value>=self.FREQMIN, 'value too low'
		assert value<=self.FREQMAX, 'value too high'
		return self.sendCommand('f%.8f'%(value/1e6))
	def setFrequencyFast(self,value):
		assert value>=self.FREQMIN, 'value too low'
		assert value<=self.FREQMAX, 'value too high'
		self.sendCommandFast('f%.8f'%(value/1e6))
		
	def getPower(self):
		return float(self.sendCommand('W?'))
	def setPower(self,value):
		assert value>=self.POWMIN, 'value too low'
		assert value<=self.POWMAX, 'value too high'
		self.sendCommand('W%.3f'%(value))

	def getCalibrationSuccess(self):
		return bool(int(self.sendCommand('V?')))

	def getTemperatureCompensationSetting(self):
		# (0=none, 1=on set, 2=1sec, 3=10sec)
		return int(self.sendCommand('Z?'))
	def setTemperatureCompensationSetting(self,value):
		# (0=none, 1=on set, 2=1sec, 3=10sec)
		assert value in (0,1,2,3)
		self.sendCommand('Z%d'%value)

	def getVGADACSetting(self):
		return int(self.sendCommand('a?'))
	def setVGADACSetting(self,value):
		assert value in range(0,45001)
		self.sendCommand('a%d'%value)

	def getPhaseStep(self):
		return float(self.sendCommand('~?'))
	def setPhaseStep(self,value):
		assert value>=0.0 and value<=360.0
		self.sendCommand('~%.4f'%value)

	def getRFMuteOff(self):
		return bool(int(self.sendCommand('h?')))
	def setRFMuteOff(self,value):
		assert value in (True,False)
		self.sendCommand('h%d'%(int(value)))

	def getRFAmpOn(self):
		return bool(int(self.sendCommand('r?')))
	def setRFAmpOn(self,value):
		assert value in (True,False)
		self.sendCommand('r%d'%(int(value)))

	def getPLLPowerOn(self):
		return bool(int(self.sendCommand('E?')))
	def setPLLPowerOn(self,value):
		assert value in (True,False)
		self.sendCommand('E%d'%(int(value)))

	#I) PLL output power 2, 2
	#U) PLL charge pump current 3, 3
	#d) PLL mute till LD 1, 1
	#m) Muxout function 6, 6
	#T) Autocal On(1) or Off(0) 1, 1
	#b) Feedback select Fundamental(1) or Divided(0) 0, 0
	#i) Mathematical spacing (Hz) 100.000, 100.000
	
	def getVersionFW(self):
		return self.sendCommand('v0')
	def getVersionHW(self):
		return self.sendCommand('v1')
	
	def _programAllSettingsToEEPROM(self):
		self.sendCommand('e')

	def getReferenceSelect(self):
		# y=0=external, y=1=internal 27MHz, y=2=internal 10MHz
		return int(self.sendCommand('x?'))
	def setReferenceSelect(self,value):
		assert value in (0,1,2)
		self.sendCommand('x%d'%(int(value)))

	def getTriggerConnectorFunctions(self):
		# 0) No Triggers
		# 1) Trigger full frequency sweep
		# 2) Trigger single frequency step
		# 3) Trigger 'stop all' which pauses sequencing through all functions of the SynthHD
		# 4) Trigger digital RF ON/OFF - Could be used for External Pulse Modulation
		# 5) Remove Interrupts (Makes modulation have less jitter - use carefully)
		# 6) Reserved
		# 7) Reserved
		# 8) External AM modulation input (requires AM Internal modulation LUT set to ramp)
		# 9) External FM modulation input (requires FM Internal modulation set to chirp)
		return int(self.sendCommand('w?'))
	def setTriggerConnectorFunctions(self,value):
		assert value in range(10)
		self.sendCommand('w%d'%(int(value)))

	def getSweepLowerFreq(self):
		return float(self.sendCommand('l?'))*1e6
	def setSweepLowerFreq(self,value):
		assert value>=self.FREQMIN, 'value too low'
		assert value<=self.FREQMAX, 'value too high'
		self.sendCommand('l%.8f'%(value/1e6))
	
	def getSweepUpperFreq(self):
		return float(self.sendCommand('u?'))*1e6
	def setSweepUpperFreq(self,value):
		assert value>=self.FREQMIN, 'value too low'
		assert value<=self.FREQMAX, 'value too high'
		self.sendCommand('u%.8f'%(value/1e6))

	def getSweepStepSize(self):
		return float(self.sendCommand('s?'))*1e6
	def setSweepStepSize(self,value):
		self.sendCommand('s%.8f'%(value/1e6))
	
	def getSweepStepTime(self):
		return float(self.sendCommand('t?'))*1e3
	def setSweepStepTime(self,value):
		assert value>=0.004 and value<=10.000 
		self.sendCommand('t%.3f'%(value/1e3))
		
	def getSweepAmplitudeLow(self):
		return float(self.sendCommand('[?'))
	def setSweepAmplitudeLow(self,value):
		assert value>=-60 and value<=20 
		self.sendCommand('[%.3f'%(value))
		
	def getSweepAmplitudeHigh(self):
		return float(self.sendCommand(']?'))
	def setSweepAmplitudeHigh(self,value):
		assert value>=-60 and value<=20 
		self.sendCommand(']%.3f'%(value))
		
	def getSweepDirection(self):
		#(up=1 / down=0)
		return int(self.sendCommand('^?'))
	def setSweepDirection(self,value):
		#(up=1 / down=0)
		assert value in [1,0] 
		self.sendCommand('^%d'%(value))
	
	def getSweepDifferentialFrequencySeparation(self):
		return float(self.sendCommand('k?'))*1e6
	def setSweepDifferentialFrequencySeparation(self,value):
		self.sendCommand('k%.8f'%(value/1e6))
	
	def getSweepDifferentialMethod(self):
		#(0=off, 1=ChA-DiffFreq, 2=ChA+DiffFreq)
		return int(self.sendCommand('n?'))
	def setSweepDifferentialMethod(self,value):
		#(0=off, 1=ChA-DiffFreq, 2=ChA+DiffFreq)
		assert value in [0,1,2]
		self.sendCommand('n%d'%(value))
	
	def getSweepType(self):
		#(linear=0 / tabular=1)
		return int(self.sendCommand('X?'))
	def setSweepType(self,value):
		#(linear=0 / tabular=1)
		assert value in [0,1]
		self.sendCommand('X%d'%value)
	
	def getSweepRun(self):
		#(on=1 / off=0)
		return int(self.sendCommand('g?'))
	def setSweepRun(self,value):
		#(on=1 / off=0)
		assert value in [0,1]
		self.sendCommand('g%d'%value)
		
	def getSweepContinuous(self):
		#(on=1 / off=0)
		return int(self.sendCommand('c?'))
	def setSweepContinuous(self,value):
		#(on=1 / off=0)
		assert value in [0,1]
		self.sendCommand('c%d'%value)
	
	def getAMStepTime(self):
		return float(self.sendCommand('F?'))/1e6
	def setAMStepTime(self,value):
		self.sendCommand('F%d'%(value*1e6))
	
	def getAMNumberRepetitions(self):
		return int(self.sendCommand('q?'))
	def setAMNumberRepetitions(self,value):
		self.sendCommand('q%d'%(value))
	
	def getAMRunContinuously(self):
		#(on=1 / off=0)
		return int(self.sendCommand('A?'))
	def setAMRunContinuously(self,value):
		#(on=1 / off=0)
		assert value in [1,0]
		self.sendCommand('A%d'%(value))
	
	def getAMLookupTable(self):
		#always 100 samples long but samples ignored if value==-75.0
		return [float(self.sendCommand('@%da?'%(j))) for j in range(100)]
	def setAMLookupTable(self,table):
		#always 100 samples long but samples ignored if value==-75.0
		assert len(table)==100
		cmd = ''
		for j in range(100):
			cmd+='@%da%.2f'%(j,table[j])
		self.sendCommand(cmd)
	
#     Program a Spot in the AM Lookup Table in dBm (Command @):

# Program Frequency Sweep table? and FM table?
	
	def getPulseOnTime(self):
		return float(self.sendCommand('P?'))/1e6
	def setPulseOnTime(self,value):
		assert value>=1e-6 and value<=10 
		self.sendCommand('P%d'%(value*1e6))
	
	def getPulseOffTime(self):
		return float(self.sendCommand('O?'))/1e6
	def setPulseOffTime(self,value):
		assert value>=2e-6 and value<=10 
		self.sendCommand('O%d'%(value*1e6))

	def getPulseNumberRepetitions(self):
		return float(self.sendCommand('R?'))
	def setPulseNumberRepetitions(self,value):
		assert value>=1 and value<=65500 
		self.sendCommand('R%d'%(value))

	def setPulseRunBurst(self):
		self.sendCommand('G')
	
	def getPulseContinuous(self):
		return int(self.sendCommand('j?'))
	def setPulseContinuous(self,value):
		assert value in [0,1]
		self.sendCommand('j%d'%value)
		
	def getPulseDualChannelMode(self):
		return int(self.sendCommand('D?'))
	def setPulseDualChannelMode(self):
		assert value in [0,1]
		self.sendCommand('D%d'%value)
	
	def getFMFrequency(self):
		return int(self.sendCommand('<?'))
	def setFMFrequency(self,value):
		assert value>=1 and value<=5000
		self.sendCommand('<%d'%(value))
	
	def getFMDeviation(self):
		return int(self.sendCommand('>?'))
	def setFMDeviation(self,value):
		assert abs(value)>=10
		current_freq = self.getFrequency()
		if   current_freq<=106.250e6: assert abs(value)<=160000
		elif current_freq<=212.500e6: assert abs(value)<=312000
		elif current_freq<=425.000e6: assert abs(value)<=625000
		elif current_freq<=850.000e6: assert abs(value)<=1250000
		elif current_freq<=1700.00e6: assert abs(value)<=2500000
		elif current_freq<=3400.00e6: assert abs(value)<=5000000
		elif current_freq<=6800.00e6: assert abs(value)<=10000000
		elif current_freq<=14000.0e6: assert abs(value)<=20000000
		self.sendCommand('>%d'%(value))
	
	def getFMNumberRepetitions(self):
		return int(self.sendCommand(',?'))
	def setFMNumberRepetitions(self,value):
		self.sendCommand(',%d'%(value))
	
	def getFMType(self):
		#(sinusoid=1 / chirp=0)
		return int(self.sendCommand(';?'))
	def setFMType(self,value):
		#(sinusoid=1 / chirp=0)
		assert value in [0,1]
		self.sendCommand(';%d'%(value))
	
	def getFMContinuousMode(self):
		return int(self.sendCommand('/?'))
	def setFMContinuousMode(self,value):
		assert value in [0,1]
		self.sendCommand('/%d'%value)
	
	def getPhaseLockStatus(self):
		#(lock=1 / unlock=0)
		return int(self.sendCommand('p'))
	
	def getTemperature(self):
		return float(self.sendCommand('z'))
	
	def getPLLReferenceFrequency(self):
		return float(self.sendCommand('*?'))*1e6
	def setPLLReferenceFrequency(self,value):
		assert value>=10e6 and value<=100e6
		self.sendCommand('*%.3f'%(value/1e6))
	
	def getModelType(self):
		return self.sendCommand('+')
	def getSerialNumber(self):
		return self.sendCommand('-')
	
