# This software is a work in progress. It is a console interface designed 
# to operate the BLAST-TNG ROACH2 firmware. 
#
# Copyright (C) January, 2018  Gordon, Sam <sbgordo1@asu.edu>
# Author: Gordon, Sam <sbgordo1@asu.edu>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import os, sys
import numpy as np
import socket as sock
import time
import struct
import select
import errno
import pygetdata as gd

class roachDownlink(object):
    """Object for handling Roach2 UDP downlink"""
    def __init__(self, ri, fpga, gc, regs, socket, data_rate):
        self.ri = ri
        self.fpga = fpga
        self.gc = gc
        self.regs = regs
        self.s = socket
        self.data_rate = data_rate
        self.data_len = 8192
        self.header_len = 42
        self.buf_size = self.data_len + self.header_len
        temp_dstip = self.gc[np.where(self.gc == 'udp_dest_ip')[0][0]][1]
        temp_dstip = sock.inet_aton(temp_dstip)
        self.udp_dest_ip = struct.unpack(">L", temp_dstip)[0]
        temp_srcip = self.gc[np.where(self.gc == 'udp_src_ip')[0][0]][1]
        temp_srcip = sock.inet_aton(temp_srcip)
        self.udp_src_ip = struct.unpack(">L", temp_srcip)[0]
        self.udp_src_port = int(self.gc[np.where(self.gc == 'udp_src_port')[0][0]][1])
        self.udp_dst_port = int(self.gc[np.where(self.gc == 'udp_dst_port')[0][0]][1])
        src_mac = self.gc[np.where(self.gc == 'udp_src_mac')[0][0]][1]
        self.udp_srcmac1 = int(src_mac[0:4], 16)
        self.udp_srcmac0 = int(src_mac[4:], 16)
        dest_mac = self.gc[np.where(self.gc == 'udp_dest_mac')[0][0]][1]
        self.udp_destmac1 = int(dest_mac[0:4], 16)
        self.udp_destmac0 = int(dest_mac[4:], 16)

    def configSocket(self):
        """Configure socket parameters"""
        try:
            self.s.setsockopt(sock.SOL_SOCKET, sock.SO_RCVBUF, self.buf_size)
            self.s.bind((self.gc[np.where(self.gc == 'udp_dest_device')[0][0]][1], 3))
        except sock.error, v:
            errorcode = v[0]
            if errorcode == 19:
                print "Ethernet device could not be found"
                pass
        return

    def configDownlink(self):
        """Configure GbE parameters"""
        
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_srcmac0_reg')[0][0]][1], self.udp_srcmac0)
        time.sleep(0.05)
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_srcmac1_reg')[0][0]][1], self.udp_srcmac1)
        time.sleep(0.05)
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_destmac0_reg')[0][0]][1], self.udp_destmac0)
        time.sleep(0.05)
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_destmac1_reg')[0][0]][1], self.udp_destmac1)
        time.sleep(0.05)
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_srcip_reg')[0][0]][1], self.udp_src_ip)
        time.sleep(0.05)
        
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_destip_reg')[0][0]][1], self.udp_dest_ip)
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_destport_reg')[0][0]][1], self.udp_dst_port)
        time.sleep(0.1)
        
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_srcport_reg')[0][0]][1], self.udp_src_port)
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_start_reg')[0][0]][1],0)
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_start_reg')[0][0]][1],1)
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'udp_start_reg')[0][0]][1],0)
        return

    def waitForData(self):
        """Uses select function to poll data socket
           outputs:
               packet: UDP packet, string packed"""
        timeout = 5.
        read, write, error = select.select([self.s], [], [], timeout)
        if not (read or write or error):
            print "Socket timed out"
            return
        else:
            for s in read:
                packet = self.s.recv(self.buf_size)
                if len(packet) != self.buf_size:
                    packet = []
        return packet

    def parseChanData(self, chan, data):
        """Parses packet data into I, Q, and phase
           inputs:
               int chan: Channel to parse
               float data: Array of packet data
           outputs:
               float I: I values for channel
               float Q: Q values for channel"""
        if (chan % 2) > 0:
            I = data[1024 + ((chan - 1) / 2)]
            Q = data[1536 + ((chan - 1) / 2)]
        else:
            I = data[0 + (chan/2)]
            Q = data[512 + (chan/2)]
        return I, Q, np.arctan2([Q],[I])

    def zeroPPS(self):
        """Sets the PPS counter to zero"""
        self.fpga.write_int(self.regs[np.where(self.regs == 'pps_start_reg')[0][0]][1], 0)
        time.sleep(0.1)
        self.fpga.write_int(self.regs[np.where(self.regs == 'pps_start_reg')[0][0]][1], 1)
        time.sleep(0.1)
        return

    def parsePacketData(self):
        """Parses packet data, filters reception based on source IP
           outputs:
               packet: The original data packet
               float data: Array of channel data
               header: String packed IP/ETH header
               saddr: The packet source address"""
        packet = self.waitForData()
        if not packet:
            print "Non-Roach packet received"
            return
        data = np.fromstring(packet[self.header_len:], dtype = '<i').astype('float')
        header = packet[:self.header_len]
        saddr = np.fromstring(header[26:30], dtype = "<I")
        saddr = sock.inet_ntoa(saddr) # source addr
        ### Filter on source IP ###
        if (saddr != self.gc[np.where(self.gc == 'udp_src_ip')[0][0]][1]):
            print "Non-Roach packet received"
            return
        return packet, data, header, saddr

    def testDownlink(self, time_interval):
        """Tests UDP link. Monitors data stream for time_interval and checks
           for packet count increment
           inputs:
               float time_interval: time interval to monitor, seconds
           outputs:
               returns 0 on success, -1 on failure"""
        print "Testing downlink..."
        first_idx = np.zeros(1)
        self.zeroPPS()
        Npackets = np.ceil(time_interval * self.data_rate)
        count = 0
        while count < Npackets:
            try:
                packet, data, header, saddr = self.parsePacketData()
            except TypeError:
                continue
            if not packet:
                continue
            else:
                packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
            count += 1
        if (packet_count - first_idx < 1):
            return -1
        return 0

    def streamChanPhase(self, chan, time_interval):
        """Obtains a channel data stream over time_interval
           inputs:
               int chan: Channel to monitor
               float time_interval: time interval to integrate, seconds
           outputs:
               float phases: Array of phases for channel"""
        self.zeroPPS()
        Npackets = int(np.ceil(time_interval * self.data_rate))
        phases = np.zeros(Npackets)
        Is = np.zeros(Npackets)
        Qs = np.zeros(Npackets)
        count = 0
        while count < Npackets:
            try:
                packet, data, header, saddr = self.parsePacketData()
                if not packet:
                    continue
                else:
                    I, Q, phase = self.parseChanData(chan, data)
                    phases[count] = phase
                    Is[count] = I
                    Qs[count] = Q
            except TypeError:
                continue
            if not np.size(data):
                continue
            count += 1
        return Is, Qs, phases

    def printChanInfo(self, chan, time_interval):
        """Parses channel info from packet and prints to screen for specified
           time interval
           inputs:
               int chan: Channel index
               float time_interval: Time interval to stream, seconds"""
        self.zeroPPS()
        count = 0
        previous_idx = np.zeros(1)
        Npackets = np.ceil(time_interval * self.data_rate)
        while count < Npackets:
            try:
                packet, data, header, saddr = self.parsePacketData()
                if not packet:
                    continue
            except TypeError:
                continue
            if not np.size(data):
                continue
            daddr = np.fromstring(header[30:34], dtype = "<I")
            daddr = sock.inet_ntoa(daddr) # dest addr
            smac = np.fromstring(header[6:12], dtype = "<B")
            dmac = np.fromstring(header[:6], dtype = "<B")
            src = np.fromstring(header[34:36], dtype = ">H")[0]
            dst = np.fromstring(header[36:38], dtype = ">H")[0]
            ### Parse packet data ###
            roach_checksum = (np.fromstring(packet[40:42],dtype = '>I'))
            sec_ts = (np.fromstring(packet[-16:-12],dtype = '>I')) # seconds elapsed since 'pps_start'
            fine_ts = np.round((np.fromstring(packet[-12:-8],dtype = '>I').astype('float')/256.0e6)*1.0e3,3) # milliseconds since last packet
            packet_count = (np.fromstring(packet[-8:-4],dtype = '>I')) # raw packet count since 'pps_start'
            packet_info_reg = (np.fromstring(packet[-4:],dtype = '>I'))
	    
	    if count > 0:
                if (packet_count - previous_idx != 1):
                    print "Warning: Packet index error"
                __, __, phase = self.parseChanData(chan, data)
                print
                print "Roach chan =", chan
                print "src MAC = %x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB", smac)
                print "dst MAC = %x:%x:%x:%x:%x:%x" % struct.unpack("BBBBBB", dmac)
                print "src IP : src port =", saddr,":", src
                print "dst IP : dst port  =", daddr,":", dst
                print "Roach chksum =", roach_checksum[0]
                print "PPS count =", sec_ts[0]
                print "Packet count =", packet_count[0]
                print "Chan phase (rad) =", phase[0]
		print "Packet info reg =", packet_info_reg[0] 
            count += 1
            previous_idx = packet_count
        return

    def saveSweepData(self, Navg, savepath, lo_freq, Nchan, skip_packets = 0):
        """Saves sweep data as .npy in path specified at savepath
           inputs:
               int Navg: Number of data points to average at each sweep step
               char savepath: Absolute path to save location
               float lo_freq: LO frequency at given sweep step, Hz
               int Nchan: Number of channels running
               int skip_packets: Number of packets to drop beginning of each sweep step"""
        channels = np.arange(Nchan)
        I_buffer = np.empty((Navg + skip_packets, Nchan))
        Q_buffer = np.empty((Navg + skip_packets, Nchan))
        self.zeroPPS()
        count = 0
        while count < Navg + skip_packets:
            try:
                packet, data, header, saddr = self.parsePacketData()
                if not packet:
                    print "Packet error"
                    return -1
                if not np.size(data):
                    print "Packet error"
                    return -1
                odd_chan = channels[1::2]
                even_chan = channels[0::2]
                I_odd = data[1024 + ((odd_chan - 1) / 2)]
                Q_odd = data[1536 + ((odd_chan - 1) /2)]
                I_even = data[0 + (even_chan/2)]
                Q_even = data[512 + (even_chan/2)]
                #even_phase = np.arctan2(Q_even,I_even)
                #odd_phase = np.arctan2(Q_odd,I_odd)
                if len(channels) % 2 > 0:
                    if len(I_odd) > 0:
                        I = np.hstack(zip(I_even[:len(I_odd)], I_odd))
                        Q = np.hstack(zip(Q_even[:len(Q_odd)], Q_odd))
                        I = np.hstack((I, I_even[-1]))
                        Q = np.hstack((Q, Q_even[-1]))
                    else:
                        I = I_even[0]
                        Q = Q_even[0]
                else:
                    I = np.hstack(zip(I_even, I_odd))
                    Q = np.hstack(zip(Q_even, Q_odd))
                I_buffer[count] = I
                Q_buffer[count] = Q
                count += 1
                I_file = 'I' + str(lo_freq)
                Q_file = 'Q' + str(lo_freq)
                np.save(os.path.join(savepath,I_file), np.mean(I_buffer[skip_packets:], axis = 0))
                np.save(os.path.join(savepath,Q_file), np.mean(Q_buffer[skip_packets:], axis = 0))
            except TypeError:
                print "Packet error"
                return -1
        return 0

    def saveDirfile_adcIQ(self, time_interval):
        data_path = self.gc[np.where(self.gc == 'DIRFILE_SAVEPATH')[0][0]][1] 
        sub_folder = raw_input("Insert subfolder name (e.g. single_tone): ")
        Npackets = np.ceil(time_interval * self.data_rate)
        Npackets = np.int(np.ceil(Npackets/1024.))
        save_path = os.path.join(data_path, sub_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = save_path + '/' + \
                   str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
        # make the dirfile
        d = gd.dirfile(filename,gd.CREAT|gd.RDWR|gd.UNENCODED)
        d.add_spec('IADC RAW FLOAT32 1')
        d.add_spec('QADC RAW FLOAT32 1')
        d.close()
        d = gd.dirfile(filename,gd.RDWR|gd.UNENCODED)
        i_file = open(filename + "/IADC", "ab")
        q_file = open(filename + "/QADC", "ab")
        count = 0
        while count < Npackets:
            I, Q = self.ri.adcIQ()
            for i in range(len(I)):
                i_file.write(struct.pack('<f', I[i]))
                q_file.write(struct.pack('<f', Q[i]))
                i_file.flush()
                q_file.flush()
                d.flush()
            count += 1
        i_file.close()
        q_file.close()
        d.close()
        return

    def saveDirfile_chanRange(self, time_interval, stage_coords = False):
        """Saves a dirfile containing the phases for a range of channels, streamed
           over a time interval specified by time_interval
           inputs:
               float time_interval: Time interval to integrate over, seconds
               bool stage_coords: Currently deprecated, to be used when beam mapping"""
        start_chan = input("Start chan # ? ")
        end_chan = input("End chan # ? ")
        chan_range = range(start_chan, end_chan + 1)
        data_path = self.gc[np.where(self.gc == 'DIRFILE_SAVEPATH')[0][0]][1]
        sub_folder = raw_input("Insert subfolder name (e.g. single_tone): ")
        Npackets = np.ceil(time_interval * self.data_rate)
        self.zeroPPS()
        save_path = os.path.join(data_path, sub_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = save_path + '/' + \
                   str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
        # make the dirfile
        d = gd.dirfile(filename,gd.CREAT|gd.RDWR|gd.UNENCODED)
        # add fields
        phase_fields = []
        for chan in chan_range:
            phase_fields.append('chP_' + str(chan))
            d.add_spec('chP_' + str(chan) + ' RAW FLOAT64 1')
        d.close()
        d = gd.dirfile(filename,gd.RDWR|gd.UNENCODED)
        nfo_phase = map(lambda z: filename + "/chP_" + str(z), chan_range)
        fo_phase = map(lambda z: open(z, "ab"), nfo_phase)
        fo_time = open(filename + "/time", "ab")
        fo_count = open(filename + "/packet_count", "ab")
        count = 0
        while count < Npackets:
            ts = time.time()
            try:
                packet, data, header, saddr = self.parsePacketData()
                if not packet:
                    continue
            #### Add field for stage coords ####
            except TypeError:
                continue
            packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
            idx = 0
            for chan in range(start_chan, end_chan + 1):
                __, __, phase = self.parseChanData(chan, data)
                fo_phase[idx].write(struct.pack('d', phase))
                fo_phase[idx].flush()
                idx += 1
            fo_count.write(struct.pack('L',packet_count))
            fo_count.flush()
            fo_time.write(struct.pack('d', ts))
            fo_time.flush()
            count += 1
        for idx in range(len(fo_phase)):
            fo_phase[idx].close()
        fo_time.close()
        fo_count.close()
        d.close()
        return

    def saveDirfile_chanRangeIQ(self, time_interval, stage_coords = False):
        """Saves a dirfile containing the I and Q values for a range of channels, streamed
           over a time interval specified by time_interval
           inputs:
               float time_interval: Time interval to integrate over, seconds
               bool stage_coords: Currently deprecated, to be used when beam mapping"""
        start_chan = input("Start chan # ? ")
        end_chan = input("End chan # ? ")
        chan_range = range(start_chan, end_chan + 1)
        data_path = self.gc[np.where(self.gc == 'DIRFILE_SAVEPATH')[0][0]][1] 
        sub_folder = raw_input("Insert subfolder name (e.g. single_tone): ")
        Npackets = int(np.ceil(time_interval * self.data_rate))
        self.zeroPPS()
        save_path = os.path.join(data_path, sub_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = save_path + '/' + \
                   str(int(time.time())) + '-' + time.strftime('%b-%d-%Y-%H-%M-%S') + '.dir'
        print filename
        # make the dirfile
        d = gd.dirfile(filename,gd.CREAT|gd.RDWR|gd.UNENCODED)
        # add fields
        I_fields = []
        Q_fields = []
        for chan in chan_range:
            I_fields.append('I_' + str(chan))
            Q_fields.append('Q_' + str(chan))
            d.add_spec('I_' + str(chan) + ' RAW FLOAT64 1')
            d.add_spec('Q_' + str(chan) + ' RAW FLOAT64 1')
        d.close()
        d = gd.dirfile(filename,gd.RDWR|gd.UNENCODED)
        nfo_I = map(lambda z: filename + "/I_" + str(z), chan_range)
        nfo_Q = map(lambda z: filename + "/Q_" + str(z), chan_range)
        fo_I = map(lambda z: open(z, "ab"), nfo_I)
        fo_Q = map(lambda z: open(z, "ab"), nfo_Q)
        fo_time = open(filename + "/time", "ab")
        fo_count = open(filename + "/packet_count", "ab")
        count = 0
        while count < Npackets:
            ts = time.time()
            try:
                packet, data, header, saddr = self.parsePacketData()
                if not packet:
                    continue
            #### Add field for stage coords ####
            except TypeError:
                continue
            packet_count = (np.fromstring(packet[-4:],dtype = '>I'))
            idx = 0
            for chan in range(start_chan, end_chan + 1):
                I, Q, __ = self.parseChanData(chan, data)
                fo_I[idx].write(struct.pack('d', I))
                fo_Q[idx].write(struct.pack('d', Q))
                fo_I[idx].flush()
                fo_Q[idx].flush()
                idx += 1
            fo_count.write(struct.pack('L',packet_count))
            fo_count.flush()
            fo_time.write(struct.pack('d', ts))
            fo_time.flush()
            count += 1
        for idx in range(len(fo_I)):
             fo_I[idx].close()
             fo_Q[idx].close()
        fo_time.close()
        fo_count.close()
        d.close()
        return

