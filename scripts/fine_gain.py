# this script takes a fine scan and gain scan combo
#then analyzes it if you like
analyze = False

print("Running fine scan and gain scan combo")
#fine scan
targetSweep(ri, udp, valon)
fine_name = str(np.load("last_targ_dir.npy"))
plotTargSweep(fine_name)
#gain scan
targetSweep(ri, udp, valon,span = 1.e6,lo_step = 1.e6/100.)
gain_name = str(np.load("last_targ_dir.npy"))
plotTargSweep(gain_name)


print("The fine scan is: " + fine_name)
print("The gain scan is: " + gain_name)


if analyze:
	#Analysis section uncomment if you want to analyze the data 
	#with methods from https://github.com/GlennGroupUCB/submm_python_routines.git
	# requires that repository to be in your python path


	#from kidPy import openStoredSweep
	#import numpy as np
	#import matplotlib.pyplot as plt
	from matplotlib.backends.backend_pdf import PdfPages
	from multitone_kidPy import read_multitone
	from KIDs import resonance_fitting

	# these were file names for testing
	#fine_name = "../data/targ/1521158106-Mar-15-2018-17-55-06.dir"
	#gain_name = "../data/targ/1521158129-Mar-15-2018-17-55-29.dir"

	fine = read_multitone.read_iq_sweep(fine_name)
	gain = read_multitone.read_iq_sweep(gain_name)
	outfile_dir = fine_name

	#plt.ioff() stop plots form popping up automatically

	pdf_pages = PdfPages(outfile_dir+"/"+"fit_plots.pdf")

	all_fits_mag = np.zeros((8,fine['freqs'].shape[1]))
	all_fits_iq = np.zeros((9,fine['freqs'].shape[1]))

	for i in range(0,fine['freqs'].shape[1]):
		f = np.hstack((fine['freqs'][:,i],gain['freqs'][:,i]))*10**6
		z = np.hstack((fine['I'][:,i],gain['I'][:,i]))+1.j*np.hstack((fine['Q'][:,i],gain['Q'][:,i]))

		# fit nonlinear magnitude
		x0 = resonance_fitting.guess_x0_mag_nonlinear(f,z,verbose = True)
		fit_dict_mag = resonance_fitting.fit_nonlinear_mag(f,z,x0=x0)#,bounds =bounds)
		all_fits_mag[:,i] = fit_dict_mag['fit'][0]

		fig = plt.figure(i,figsize = (12,12))

		plt.subplot(221)
		plt.plot(fine['freqs'][:,i],10*np.log10(fine['I'][:,i]**2+fine['Q'][:,i]**2),'o')
		plt.plot(gain['freqs'][:,i],10*np.log10(gain['I'][:,i]**2+gain['Q'][:,i]**2),'o')
		plt.plot(f/10**6,10*np.log10(fit_dict_mag['fit_result']),"+")
		plt.plot(f/10**6,10*np.log10(fit_dict_mag['x0_result']),"x")
		plt.xlabel("Frequency")
		plt.ylabel("Power (dB)")

		plt.subplot(223)
		plt.plot(fine['freqs'][:,i],10*np.log10(fine['I'][:,i]**2+fine['Q'][:,i]**2),'o')
		plt.plot(gain['freqs'][:,i],10*np.log10(gain['I'][:,i]**2+gain['Q'][:,i]**2),'o')
		plt.plot(f/10**6,10*np.log10(fit_dict_mag['fit_result']),"+")
		plt.plot(f/10**6,10*np.log10(fit_dict_mag['x0_result']),"x")
		plt.xlabel("Frequency")
		plt.ylabel("Power (dB)")
		plt.xlim(np.min(fine['freqs'][:,i]),np.max(fine['freqs'][:,i]))

		# fit nonlinear iq 
		x0 = resonance_fitting.guess_x0_iq_nonlinear(f,z,verbose = True)
		fit_dict_iq = resonance_fitting.fit_nonlinear_iq(f,z,x0=x0)
		all_fits_iq[:,i] = fit_dict_iq['fit'][0]

		plt.subplot(222,aspect ='equal')
		plt.plot(fine['I'][:,i],fine['Q'][:,i],'o')
		plt.plot(gain['I'][:,i],gain['Q'][:,i],'o')
		plt.plot(np.real(fit_dict_iq['fit_result']),np.imag(fit_dict_iq['fit_result']),"+")
		plt.plot(np.real(fit_dict_iq['x0_result']),np.imag(fit_dict_iq['x0_result']),"+")
		plt.xlabel("I")
		plt.ylabel("Q")


		plt.subplot(224,aspect ='equal')
		plt.plot(fine['I'][:,i],fine['Q'][:,i],'o')
		plt.plot(gain['I'][:,i],gain['Q'][:,i],'o')
		plt.plot(np.real(fit_dict_iq['fit_result']),np.imag(fit_dict_iq['fit_result']),"+")
		plt.plot(np.real(fit_dict_iq['x0_result']),np.imag(fit_dict_iq['x0_result']),"+")
		plt.plot(fine['I'][:,i],fine['Q'][:,i])
		plt.xlabel("I")
		plt.ylabel("Q")
		plt.xlim(np.min(fine['I'][:,i]),np.max(fine['I'][:,i]))
		plt.ylim(np.min(fine['Q'][:,i]),np.max(fine['Q'][:,i]))

		pdf_pages.savefig(fig)
		plt.close(fig)

	pdf_pages.close()

	pdf_pages = PdfPages(outfile_dir+"/"+"fit_results.pdf")

	fig = plt.figure(1, figsize = (12,6))
	plt.title("Center frequency")
	plt.plot(all_fits_mag[0,:]/10**6,'o',label = "Mag fit")
	plt.plot(all_fits_iq[0,:]/10**6,'o',label = "IQ fit")
	plt.xlabel("resonator index")
	plt.ylabel("Resonator Frequency (MHz)")
	plt.legend(loc = 4)
	pdf_pages.savefig(fig)
	plt.close()

	fig = plt.figure(2,figsize = (12,6))
	plt.title("Resonator Qs")
	plt.plot(all_fits_mag[1,:],'o',label = "Qr Mag",color = 'g')
	plt.plot(all_fits_iq[1,:],'*',label = "Qr IQ",color = 'g')
	plt.plot(all_fits_mag[1,:]/all_fits_mag[2,:],'o',label = "Qc Mag",color = 'b')
	plt.plot(all_fits_iq[1,:]/all_fits_mag[2,:],'*',label = "Qc IQ",color = 'b')
	plt.plot(1/(1/all_fits_mag[1,:]+1/(all_fits_mag[1,:]/all_fits_mag[2,:])),'o',label = "Qi Mag",color = 'r')
	plt.plot(1/(1/all_fits_iq[1,:]+1/(all_fits_iq[1,:]/all_fits_mag[2,:])),'*',label = "Qi IQ",color = 'r')
	plt.xlabel("Resonator index")
	plt.ylabel("Resonator Q")
	plt.yscale('log')
	plt.legend()
	pdf_pages.savefig(fig)
	plt.close()

	fig = plt.figure(3,figsize = (12,6))
	plt.title("Non linearity parameter a")
	plt.plot(all_fits_mag[4,:],'o',label = "a Mag")
	plt.plot(all_fits_iq[4,:],'o',label = "a IQ")
	plt.xlabel("Resonator index")
	plt.ylabel("Non-linearity parameter a")
	plt.legend()
	pdf_pages.savefig(fig)
	plt.close()

	pdf_pages.close()

	





