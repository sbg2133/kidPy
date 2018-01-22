%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://seti.ssl.berkeley.edu/casper/                                      %
%   Copyright (C) 2006 University of California, Berkeley                     %
%                                                                             %
%   This program is free software; you can redistribute it and/or modify      %
%   it under the terms of the GNU General Public License as published by      %
%   the Free Software Foundation; either version 2 of the License, or         %
%   (at your option) any later version.                                       %
%                                                                             %
%   This program is distributed in the hope that it will be useful,           %
%   but WITHOUT ANY WARRANTY; without even the implied warranty of            %
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             %
%   GNU General Public License for more details.                              %
%                                                                             %
%   You should have received a copy of the GNU General Public License along   %
%   with this program; if not, write to the Free Software Foundation, Inc.,   %
%   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.               %
%                                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Modified by Adrian Sinclair(Arizona State University, Astronomy Inst. Lab)

function b = xps_async_gpio_in(blk_obj)

%Is the block under consideration tagged as an xps block?
if ~isa(blk_obj,'xps_block')
    error('XPS_GPIO class requires a xps_block class object');
end

% Is the block under consideration an  async_gpio_in block? 
if ~strcmp(get(blk_obj,'type'),'xps_async_gpio_in')
    error(['Wrong XPS block type: ',get(blk_obj,'type')]);
end

% Grab the block name. It will be handy when we need to grab parameters
blk_name = get(blk_obj,'simulink_name');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get the hardware platform (i.e. ROACH board, etc, and GPIO bank used by the block
xsg_obj = get(blk_obj,'xsg_obj');
hw_sys_full =  get(xsg_obj,'hw_sys');
hw_sys = strtok(hw_sys_full,':') %hw_sys_full is ROACH:SX95t (We only want "ROACH")

s.hw_sys = hw_sys
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%constructor
b = class(s,'xps_async_gpio_in',blk_obj);

% ip name
%The name of your pcore which will be instantiated when the toolflow sees the yellow block
b = set(b, 'ip_name','async_gpio_in');

% external ports
% Here we set up the external (i.e. to FPGA pin) connections required by the yellow block pcore

iostandard = 'LVCMOS15';%set io standard


%io direction is inout
bus_width = 1

ucf_fields = {};
ucf_values = {};

ucf_constraints = struct('IOSTANDARD',iostandard);

% Give a name to the external port name and buffer names. The iobname should match an entry in the hw_routes table,
% allowing the tools to map the name to an FPGA pin number.
% the extportname is the name given to the signal connected to the pins. It should be connected somewhere else in your pcore
% (see the last few lines of this script for the connection to the pcore)
% Be careful to make sure your signal names won't clash if you use multiple copies of the same yellow block in your design.
% here we use a modification the block name (which simulink mandates is unique)
extportname = [clear_name(blk_name), '_ext'];
iobname = [s.hw_sys, '.','gpio'];

%this string is passed to the external port structure 
pin_str = ['{',iobname,'{[',num2str([1:bus_width]),']}}']



%ucf_constraints = cell2struct(ucf_values, ucf_fields, length(ucf_fields));

ext_ports.io_pad =   {bus_width  'in'  extportname  pin_str  'vector=true'  struct()  ucf_constraints};
b = set(b,'ext_ports',ext_ports);
