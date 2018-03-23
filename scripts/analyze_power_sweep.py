import numpy as np
import matplotlib.pyplot as plt
from multitone_kidPy import analyze
from scipy import interpolate
from matplotlib.backends.backend_pdf import PdfPages

# I was putting this in a directory called power sweeps so that it doesn't pollute
# some other directory with a bunch of output files

# if you have already fit all the data select false i.e. used analyze on the go in power sweep
fit_scans = False

# select the level of non-liearity paramter (a) you would like to normalize power to
bif_level = 0.5

#read in the names of the scan in the power sweep
fine_names = np.load("2018-03-2209:55:36_fine_names.npy")
gain_names = np.load("2018-03-2209:55:36_gain_names.npy")
attn_levels = np.load("2018-03-2209:55:36_attn_levels.npy")


# fit each sweep individually
if fit_scans:
	for i in range(0,len(fine_names)):
		print("fitting scan "+str(i))
		analyze.fit_fine_gain("../targ/"+fine_names[i].split('/')[3],"../targ/"+gain_names[i].split('/')[3])

	
#load in the first data set
mag_fits = np.load("../targ/"+fine_names[0].split('/')[3]+"/all_fits_mag.npy")
iq_fits = np.load("../targ/"+fine_names[0].split('/')[3]+"/all_fits_iq.npy")

#intialize arrays to hold all of the fit results
all_fits_mag = np.zeros((mag_fits.shape[0],mag_fits.shape[1],len(attn_levels)))
all_fits_iq = np.zeros((iq_fits.shape[0],iq_fits.shape[1],len(attn_levels)))

# collect all of the fit results
for i in range(0,len(fine_names)):
	all_fits_mag[:,:,i] = np.load("../targ/"+fine_names[i].split('/')[3]+"/all_fits_mag.npy")
	all_fits_iq[:,:,i] = np.load("../targ/"+fine_names[i].split('/')[3]+"/all_fits_iq.npy")

# make a pdf to plot
pdf_pages = PdfPages("power_sweep_"+fine_names[0].split('/')[3]+".pdf")

# initialize arrays to hold the power level at a = bif_level
bif_levels_mag = np.zeros(all_fits_mag.shape[1])
bif_levels_iq = np.zeros(all_fits_mag.shape[1])

#llop through plot and extract the power level at a = bif_level
for i in range(0,all_fits_mag.shape[1]):
	fig = plt.figure(i)
	plt.title("Resonator index = "+str(i))
	plt.plot(-attn_levels,all_fits_mag[4,i,:],color = 'r')
	plt.plot(-attn_levels,all_fits_iq[4,i,:],color = 'g')
	plt.plot(-attn_levels,all_fits_mag[4,i,:],'o',color = 'r',label = "Mag fit")
	plt.plot(-attn_levels,all_fits_iq[4,i,:],'o',color = 'g',label = "IQ fit")

	try:
		first_bif_mag = np.where(all_fits_mag[4,i,:]>bif_level)[0][0]
	except:
		first_bif_mag = all_fits_mag.shape[2]-1
	try:
		first_bif_iq = np.where(all_fits_mag[4,i,:]>bif_level)[0][0]
	except:
		first_bif_iq = all_fits_iq.shape[2]-1
	if first_bif_mag == 0:
		first_bif_mag =1
	if first_bif_iq == 0:
		first_bif_iq = 1

	interp_mag  = interpolate.interp1d(-attn_levels[0:first_bif_mag+1],all_fits_mag[4,i,:][0:first_bif_mag+1] )
	interp_iq  = interpolate.interp1d(-attn_levels[0:first_bif_iq+1],all_fits_iq[4,i,:][0:first_bif_iq+1] )

	powers_mag = np.linspace(-attn_levels[0]+.01,-attn_levels[first_bif_mag]-0.01,1000)
	powers_iq = np.linspace(-attn_levels[0]+.01,-attn_levels[first_bif_iq]-0.01,1000)

	bifurcation_mag = powers_mag[np.argmin(np.abs(interp_mag(powers_mag)-np.ones(len(powers_mag))*bif_level))]
	bifurcation_iq = powers_iq[np.argmin(np.abs(interp_iq(powers_iq)-np.ones(len(powers_iq))*bif_level))]

	plt.plot((bifurcation_mag,bifurcation_mag),(0,bif_level),color = 'r')
	plt.plot((bifurcation_iq,bifurcation_iq),(0,bif_level),color = 'g')
	bif_levels_mag [i] = bifurcation_mag
	bif_levels_iq [i] = bifurcation_iq	
	plt.ylim(0,1)
	plt.xlim(np.min(-attn_levels)-1,np.max(-attn_levels)+1)
	plt.legend(loc = 4)



	pdf_pages.savefig(fig)
	plt.close(fig)

pdf_pages.close()


plt.figure(1,figsize = (12,16))
plt.subplot(211)
plt.plot(bif_levels_mag+attn_levels[-1],'o',color = 'g',label = "Mag fit")
plt.plot(bif_levels_iq+attn_levels[-1],'o',color = 'r',label = "IQ fit")
plt.xlabel("Resonator Index")
plt.ylabel("Power at a = 0.5")
plt.legend(loc = 4)

plt.subplot(212)
plt.title("Transfer function")
plt.plot(np.sqrt(10**((bif_levels_mag+attn_levels[-1])/10)/np.min(10**((bif_levels_mag+attn_levels[-1])/10))),'o')
plt.xlabel("Resonator Index")
plt.ylabel("Power at a = 0.5")

plt.savefig("power_sweep_results_"+fine_names[0].split('/')[3]+".pdf")

# save the output
np.savetxt("bif_levels_mag_"+fine_names[0].split('/')[3]+".csv",bif_levels_mag+attn_levels[-1])
np.savetxt("bif_levels_iq_"+fine_names[0].split('/')[3]+".csv",bif_levels_mag+attn_levels[-1])
np.save("trans_func_"+fine_names[0].split('/')[3],np.sqrt(10**((bif_levels_mag+attn_levels[-1])/10)/np.min(10**((bif_levels_mag+attn_levels[-1])/10))))

plt.show()
