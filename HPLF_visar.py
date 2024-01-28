"""VISAR analysis tools."""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.fftpack as sc_fft
from HPLF_streaks import *
from matplotlib.colors import LogNorm
import copy
import os
import scipy.integrate

c = 299792458 # m/s

class HPLF_visar():

    def __init__(self, streak_im_ref, streak_im_shot, wavelength, dict_etalon, ROI=None):
        # loading ref and shot streak_im objects
        self.refimage = copy.deepcopy(streak_im_ref)
        self.shotimage = copy.deepcopy(streak_im_shot)
        # substract background intensity value
        self.refimage.im = self.refimage.im - float(self.refimage.background_level)
        self.shotimage.im = self.shotimage.im - float(self.shotimage.background_level)
        # wavelength of probe laser
        self.wavelength = wavelength # in m
        # etalon dictionary and VPF determination
        self.sign = dict_etalon['sign'] # set orientation of fringe motion
        if 'VPF' in dict_etalon.keys():
            self.VPF = self.sign*dict_etalon['VPF']
        else:
            self.etalon_n = dict_etalon['refr_index']
            self.etalon_dndl = dict_etalon['dn_dl']
            self.etalon_thickness = dict_etalon['thickness'] # in m
            # Compute VPF from HPLF VISARs geometry
            self.height = 4.8e-2 # m
            self.width = 22.5e-2 # m
            self.deltaWL = 0 # white light calibration error
            self.tau_0 = 2*self.etalon_thickness/c*(self.etalon_n-1)/self.etalon_n
            self.delta = - self.etalon_dndl*self.wavelength*self.etalon_n/(self.etalon_n**2 -1)
            self.gamma = np.arctan(self.height/self.width)
            self.gamma_n = np.arcsin(np.sin(self.gamma)/self.etalon_n)
            self.d = self.etalon_thickness*(1-np.tan(self.gamma_n)/np.tan(self.gamma))
            self.OPL = 2*( self.deltaWL + self.etalon_n*self.etalon_thickness/np.cos(self.gamma_n) + (self.height - self.etalon_thickness*np.tan(self.gamma_n))/np.sin(self.gamma) - self.height/np.sin(self.gamma) )
            self.tau = self.OPL/c
            self.VPF = self.sign*self.wavelength/(2*(1+self.delta)*self.tau)*1e-3 # in km/s
            print(f'Calculated VPF: {self.VPF} km/s')
        if ROI==None:
            self.ROI = [(np.min(self.shotimage.space), np.max(self.shotimage.space)), (np.min(self.shotimage.time), np.max(self.shotimage.time))]
        else:
            self.ROI = ROI        
    
    def apply_ROI(self):
        update_streak_im_ROI(self.refimage, self.ROI)
        update_streak_im_ROI(self.shotimage, self.ROI)
        # Plot
        [self.reftime_grid, self.refspace_grid] = np.meshgrid(self.refimage.time, self.refimage.space)
        [self.shottime_grid, self.shotspace_grid] = np.meshgrid(self.shotimage.time, self.shotimage.space)
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(r'Time (ns)')
        ax1.set_ylabel(r'Space ($\mu$m)')
        ax1.set_title(r'ref. ROI')
        ax1.pcolormesh(self.reftime_grid, self.refspace_grid, self.refimage.im.transpose(), cmap='binary', vmax=(np.max(self.refimage.im)-np.min(self.refimage.im))/3+np.min(self.refimage.im))
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(r'Time (ns)')
        ax2.set_ylabel(r'Space ($\mu$m)')
        ax2.set_title(r'Shot ROI')
        ax2.pcolormesh(self.shottime_grid, self.shotspace_grid, self.shotimage.im.transpose(), cmap='binary', vmax=(np.max(self.shotimage.im)-np.min(self.shotimage.im))/3+np.min(self.shotimage.im))
        fig.tight_layout()
        plt.show()

    def deghost(self, deghROI=None, dqt=0, apply=False):
        # Apply deghROI
        if deghROI is None:
            self.deghROI = [(np.min(self.shotimage.space), np.max(self.shotimage.space)), (np.min(self.shotimage.time), np.max(self.shotimage.time))]
        else:
            self.deghROI = deghROI
        idx_degh_space = np.where( (self.shotimage.space>=deghROI[0][0])&(self.shotimage.space<deghROI[0][1]) )
        idx_degh_time = np.where( (self.shotimage.time>=deghROI[1][0])&(self.shotimage.time<deghROI[1][1]) )
        self.degh_space = self.shotimage.space[ idx_degh_space ]
        self.degh_time = self.shotimage.time[ idx_degh_time ]
        self.degh_shotimage = self.shotimage.im[idx_degh_time].transpose()[idx_degh_space].transpose()
        # Compute forward 2D FFT
        self.degh_shotFT = sc_fft.fft2(self.degh_shotimage)
        degh_shotFT_plot = abs(np.roll(np.roll(self.degh_shotFT, int(len(self.degh_space)/2.)), int(len(self.degh_time)/2.), axis=0))
        self.degh_shotfreqspaceFT = sc_fft.fftfreq(len(self.degh_space), 1)
        self.degh_shotfreqtimeFT = sc_fft.fftfreq(len(self.degh_time), 1)
        [self.degh_shotfreqspaceFT_grid, self.degh_shotfreqtimeFT_grid] = np.meshgrid(self.degh_shotfreqspaceFT, self.degh_shotfreqtimeFT)
        # Filter zero frequencies in time with width dqt
        if dqt is None:
            self.degh_shotFT_mask = np.where((self.degh_shotfreqtimeFT_grid==np.nan)) # False mask
        else:
            self.degh_shotFT_mask = np.where((((self.degh_shotfreqtimeFT_grid>=-dqt/2.) & (self.degh_shotfreqtimeFT_grid<=dqt/2.)) & (self.degh_shotfreqspaceFT_grid!=0)))
        self.degh_shotFT[self.degh_shotFT_mask]=0
        degh_shotFT_filtered_plot = abs(np.roll(np.roll(self.degh_shotFT, int(len(self.degh_space)/2.)), int(len(self.degh_time)/2.), axis=0))
        # invFFT 2D
        self.degh_shotinvFT = sc_fft.ifft2(self.degh_shotFT)
        # Incoporate ghost subtracted image in shot image
        self.shotimage_deghosted = copy.deepcopy(self.shotimage.im)
        self.shotimage_deghosted[np.min(idx_degh_time):np.max(idx_degh_time)+1, np.min(idx_degh_space):np.max(idx_degh_space)+1] = abs(self.degh_shotinvFT)
        # Apply deghost to shotimage or not
        if apply==True:
            self.shotimage.im = self.shotimage_deghosted
        # Plot
        fig = plt.figure(figsize=(6, 5))
        # ax1 = fig.add_subplot(131)
        # ax1.set_xlabel(r'Time frequency (/pixel)')
        # ax1.set_ylabel(r'Space frequency (/pixel)')
        # ax1.set_title(r'2D FFT')
        # ax1.pcolormesh(np.roll(self.degh_shotfreqtimeFT_grid, int(len(self.degh_time)/2.), axis=0), np.roll(self.degh_shotfreqspaceFT_grid, int(len(self.degh_space)/2.), axis=1), degh_shotFT_plot, cmap='binary', norm=LogNorm())
        # ax1.set_ylim(-0.05, 0.05)
        # ax1.set_xlim(-0.05, 0.05)
        # ax2 = fig.add_subplot(132)
        # ax2.set_xlabel(r'Time frequency (/pixel)')
        # ax2.set_ylabel(r'Space frequency (/pixel)')
        # ax2.set_title(r'2D FFT filtered')
        # ax2.pcolormesh(np.roll(self.degh_shotfreqtimeFT_grid, int(len(self.degh_time)/2.), axis=0), np.roll(self.degh_shotfreqspaceFT_grid, int(len(self.degh_space)/2.), axis=1), degh_shotFT_filtered_plot, cmap='binary', norm=LogNorm())
        # ax2.set_ylim(-0.05, 0.05)
        # ax2.set_xlim(-0.05, 0.05)
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel(r'Time (ns)')
        ax1.set_ylabel(r'Space ($\mu$m)')
        ax1.set_title(r'Shot image without ghost')
        ax1.pcolormesh(self.shottime_grid, self.shotspace_grid, self.shotimage_deghosted.transpose(), cmap='binary', vmax=(np.max(self.shotimage_deghosted)-np.min(self.shotimage_deghosted))/3+np.min(self.shotimage_deghosted))
        fig.tight_layout()
        plt.show()
        

    def FFT_process(self, filter_bounds):
        [self.filter_freq_min, self.filter_freq_max] = filter_bounds
        # Compute forward FT
        self.refFT = sc_fft.fft2(self.refimage.im, axes=(1))
        self.shotFT = sc_fft.fft2(self.shotimage.im, axes=(1))
        # Frequencies
        self.reffreqFT = sc_fft.fftfreq(self.refimage.npx_space, 1)
        [self.reftime_grid, self.reffreq_grid] = np.meshgrid(self.refimage.time, self.reffreqFT)
        self.shotfreqFT = sc_fft.fftfreq(self.shotimage.npx_space, 1)
        [self.shottime_grid, self.shotfreq_grid] = np.meshgrid(self.shotimage.time, self.shotfreqFT)
        # Plot
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax1.set_xlabel(r'Time (ns)')
        ax1.set_ylabel(r'Frequency (/pixel)')
        ax1.set_title(r'ref. FT')
        ax1.pcolormesh(self.reftime_grid, np.roll(self.reffreq_grid, int(self.refimage.npx_space/2), axis=0), np.roll(abs(self.refFT.transpose()), int(self.refimage.npx_space/2), axis=0), cmap='binary')
        ax1.set_ylim(-0.05, 0.05)
        ax2 = fig.add_subplot(132)
        ax2.set_xlabel(r'Time (ns)')
        ax2.set_ylabel(r'Frequency (/pixel)')
        ax2.set_title(r'Shot FT')
        ax2.pcolormesh(self.shottime_grid, np.roll(self.shotfreq_grid, int(self.shotimage.npx_space/2), axis=0), np.roll(abs(self.shotFT.transpose()), int(self.shotimage.npx_space/2), axis=0), cmap='binary')
        ax2.set_ylim(-0.05, 0.05)
        ax3 = fig.add_subplot(133)
        ax3.set_xlabel(r'Frequency (/pixel)')
        ax3.set_ylabel(r'Time-averaged ref. FT norm.')
        ax3.plot(np.roll(self.reffreqFT, int(self.refimage.npx_space/2)), np.average(np.roll(abs(self.refFT.transpose()), int(self.refimage.npx_space/2), axis=0), axis=1), 'k-')
        ax3.axvline(x=self.filter_freq_min)
        ax3.axvline(x=self.filter_freq_max)
        ax3.set_xlim(0, 0.05)
        fig.tight_layout()
        plt.show()

    def Filter(self):
        # set FT to 0 in regions outside bounds
        idx_above_middle = np.where((self.shotfreq_grid<self.filter_freq_min)&(self.shotfreq_grid>-self.filter_freq_min))
        idx_above_max_pos = np.where(self.shotfreq_grid>self.filter_freq_max)
        idx_below_max_neg = np.where(self.shotfreq_grid<-self.filter_freq_max)
        # ref image
        self.refFT_filtered = np.copy(self.refFT)
        # self.refFT_filtered.transpose()[idx_above_middle]=0
        # self.refFT_filtered.transpose()[idx_above_max_pos]=0
        # self.refFT_filtered.transpose()[idx_below_max_neg]=0
        self.refFT_filtered.transpose()[ np.where(self.reffreq_grid<self.filter_freq_min) ] = 0
        self.refFT_filtered.transpose()[ np.where(self.reffreq_grid>self.filter_freq_max) ] = 0
        # shot image
        self.shotFT_filtered = np.copy(self.shotFT)
        # self.shotFT_filtered.transpose()[idx_above_middle]=0
        # self.shotFT_filtered.transpose()[idx_above_max_pos]=0
        # self.shotFT_filtered.transpose()[idx_below_max_neg]=0
        self.shotFT_filtered.transpose()[ np.where(self.shotfreq_grid<self.filter_freq_min) ] = 0
        self.shotFT_filtered.transpose()[ np.where(self.shotfreq_grid>self.filter_freq_max) ] = 0

        # invFT
        self.refinvFT = sc_fft.ifft2(self.refFT_filtered, axes=(1))
        self.shotinvFT = sc_fft.ifft2(self.shotFT_filtered, axes=(1))

        # Grid
        [self.shottime_grid, self.shotspace_grid] = np.meshgrid(self.shotimage.time, self.shotimage.space)

        # Phase
        from skimage.restoration import unwrap_phase 
        self.refwphase = -np.angle(self.refinvFT)#/(2*np.pi)
        self.shotwphase = -np.angle(self.shotinvFT)#/(2*np.pi)
        #self.refwphase = np.average( self.shotwphase[ np.where(self.shotimage.time<=-10) ], axis=0)
        self.refuphase = unwrap_phase(self.refwphase)
        self.shotuphase = unwrap_phase(self.shotwphase)
        self.phasediff = unwrap_phase((self.shotwphase - self.refwphase))/(2*np.pi) #(self.shotuphase - self.refuphase)

        # Reflectivity
        self.refintensity = abs(self.refinvFT) #- self.refimage.background_level
        self.shotintensity = abs(self.shotinvFT) #- self.shotimage.background_level
        self.reflectivity = self.shotintensity/self.refintensity

        # Plot
        fig = plt.figure(figsize=(20, 5))
        ax1 = fig.add_subplot(141)
        ax1.set_xlabel(r'Time (ns)')
        ax1.set_ylabel(r'Space ($\mu$m)')
        ax1.set_title(r'Shot image')
        ax1.pcolormesh(self.shottime_grid, self.shotspace_grid, self.shotimage.im.transpose(), cmap='binary', vmax=(np.max(self.shotimage.im)-np.min(self.shotimage.im))/3+np.min(self.shotimage.im))
        ax2 = fig.add_subplot(142)
        ax2.set_xlabel(r'Time (ns)')
        ax2.set_ylabel(r'Space ($\mu$m)')
        ax2.set_title(r'Shot filtered image')
        ax2.pcolormesh(self.shottime_grid, self.shotspace_grid, np.real(self.shotinvFT.transpose()), cmap='binary')
        ax3 = fig.add_subplot(143)
        ax3.set_xlabel(r'Time (ns)')
        ax3.set_ylabel(r'Space ($\mu$m)')
        ax3.set_title(r'Apparent reflectivity of shot filtered image')
        ax3.pcolormesh(self.shottime_grid, self.shotspace_grid, self.reflectivity.transpose(), cmap='binary', vmin=0, vmax=2)
        ax4 = fig.add_subplot(144)
        ax4.set_xlabel(r'Time (ns)')
        ax4.set_ylabel(r'Space ($\mu$m)')
        ax4.set_title(r'Phase difference map')
        col = ax4.pcolormesh(self.shottime_grid, self.shotspace_grid, self.phasediff.transpose(), cmap='binary')
        #fig.colorbar(col)
        fig.tight_layout()
        plt.show()
    
        # fig = plt.figure(figsize=(6,5))
        # ax1=fig.add_subplot(111)
        # ax1.set_xlabel('Time (ns)')
        # ax1.set_ylabel('Velocity (km/s)')
        # ax1.plot(self.shotimage.time, np.average( self.velocity.transpose()[500:700], axis=0), 'k-')
        # fig.tight_layout()
        # plt.show()

    def Velocity(self, offset=0, tjump_list=[], njump_list=[], corr_index_list=[], vROI=None):
        self.offset = offset
        self.tjump_list = tjump_list
        self.njump_list = njump_list
        self.corr_index_list = corr_index_list
        if vROI is None:
            self.vROI = [(np.min(self.shotimage.space), np.max(self.shotimage.space)), (np.min(self.shotimage.time), np.max(self.shotimage.time))]
        else:
            self.vROI = vROI
        self.idx_vel_space = np.where( (self.shotimage.space>=vROI[0][0])&(self.shotimage.space<vROI[0][1]) )
        self.idx_vel_time = np.where( (self.shotimage.time>=vROI[1][0])&(self.shotimage.time<vROI[1][1]) )
        self.vel_space = self.shotimage.space[ self.idx_vel_space ]
        self.vel_time = self.shotimage.time[ self.idx_vel_time ]
        self.vel_phasediff = self.phasediff[self.idx_vel_time].transpose()[self.idx_vel_space].transpose()
        self.velocity = (self.offset/self.VPF + self.vel_phasediff + 0)*self.VPF/1  # no jump, temp
        # Average apparent velocity over vROI
        self.velocity_av = np.average( self.velocity, axis=1 )
        self.velocity_std = np.std( self.velocity, axis=1 )
        # Apply jumps and corrective index
        if self.tjump_list!=[]:
            for ii in range(0, len(self.tjump_list)):
                self.velocity_av[ np.where(self.vel_time>=self.tjump_list[ii]) ] = np.average((self.offset/self.VPF + self.vel_phasediff[ np.where(self.vel_time>=self.tjump_list[ii]) ] + self.sign*self.njump_list[ii])*self.VPF/self.corr_index_list[ii], axis=1)

        return [self.vel_time, self.velocity_av, self.velocity_std]
    
    def Reflectivity_shock_front(self, time, shock_velocity, tstart, thickness_list=[], refr_index_list=[], mu_list=[], R0=1, RFS=None):
        self.R0 = R0 # reference reflectivity of the layer onto which the probe laser reflects before the shot
        # for the following lists, the order should be such as the layer with the free surface is the last one.
        self.R_thickness_list = np.asarray(thickness_list) # list of thicknesses of the layers that are between the initial reflective interface and the free surface (in microns)
        self.R_refr_index_list = np.asarray(refr_index_list) # list of refractive indexes of the corresponding layers
        self.R_mu_list = np.asarray(mu_list)#*1e-4 # list of linear absorption coefficients of the corresponding layers (input in cm-1 and conversion in um-1)
        self.R_shock_velocity = shock_velocity # array with the shock velocity in the region between the initial reflective interface and the free surface (in km/s)
        self.R_time = time # time array corresponding to the shock velocity values
        self.R_L0 = np.sum(self.R_thickness_list) # maximum length of the absorbing media
        if RFS == None:
            self.R_free_surface = ((1-self.R_refr_index_list[-1])/(1+self.R_refr_index_list[-1]))**2
        else:
            self.R_free_surface = RFS
        # Distance between free surface and shock front in absorbing media at each time
        self.R_Lt = np.zeros_like(self.R_time)
        R_shock_velocity_cumintegral = scipy.integrate.cumtrapz(self.R_shock_velocity, x=self.R_time, initial=0)
        for tt in range(0, len(self.R_time)):
            self.R_Lt[tt] = self.R_L0 - R_shock_velocity_cumintegral[tt] # Length of absorbing media along time array (maybe need to be flipped?)
        # mu
        self.R_mu = np.zeros_like(self.R_time)
        cumthickness = np.cumsum(self.R_thickness_list)
        for tt in range(0, len(R_shock_velocity_cumintegral)):
            if self.R_time[tt] > tstart:
                pos_x = R_shock_velocity_cumintegral[tt]
                if (pos_x - R_shock_velocity_cumintegral[tt-1])>=0: # to avoid the consequence of possible unrealistic negative velocity
                    for aa in range(0, len(thickness_list)):
                        if pos_x <= cumthickness[aa]:
                            self.R_mu[tt] = mu_list[aa]
                            break
                else:
                    self.R_mu[tt] = 0
    
        # Apply vROI to apparent reflectivity map -- requires an existing visar vROI
        self.R_app = self.reflectivity[self.idx_vel_time].transpose()[self.idx_vel_space].transpose()  # roi of Apparent reflectivity, simple ratio (Ishot-Ibkg)/(Iref-Ibkg)
        self.R_app_av = np.average( self.R_app, axis=1 ) # ROI average
        self.R_app_std = np.std( self.R_app, axis=1 )  # ROI std
        # Shock front reflectivity
        self.R_shock_front = np.zeros_like(self.R_time)
        num_factor = self.R_free_surface + (1-self.R_free_surface)*self.R0*np.exp(-2*np.trapz(self.R_mu, x=self.R_Lt))
        print(np.trapz(self.R_mu, x=self.R_Lt))
        for tt in range(0, len(self.R_time)):
            num = self.R_app_av[tt]*num_factor
            den = self.R_free_surface + (1-self.R_free_surface)*np.exp(-2*np.trapz(self.R_mu[0:tt], x=self.R_Lt[0:tt]))
            self.R_shock_front[tt] = num/den
        return [self.R_shock_front, self.R_free_surface, R_shock_velocity_cumintegral, self.R_Lt, self.R_mu, self.R_app_av, self.R_app_std]

        
        
    def Export_velocity_trace(self, outdir, prefixe, visar_nb, ext='.txt'):
        path = outdir + os.sep + prefixe + ext
        fi = open(path, 'w')
        # Neutrino-like header
        fi.write(f'#VISAR {visar_nb}\n')
        fi.write(f'#Offset shift         : {self.offset}\n')
        fi.write(f'#Sensitivity          : {self.VPF}\n')
        fi.write(f'#Sweep Time           : {self.shotimage.A1} {self.shotimage.A2} {self.shotimage.A3} {self.shotimage.A4}\n')
        fi.write(f'#Time zero & delay    : {self.shotimage.t0_px} {self.shotimage.delay}\n')
        fi.write(f'#Center zero & fov    : {self.shotimage.pos0_px} {self.shotimage.fov}\n')
        if self.tjump_list!=[]:
            str_temp = '#Jumps                :'
            for ii in range(0, len(self.tjump_list)):
                str_temp += f' {self.tjump_list[ii]} {self.njump_list[ii]} {self.corr_index_list[ii]}'
                if ii != len(self.tjump_list)-1:
                    str_temp += ';'
            fi.write(f'{str_temp}\n')
        else:
            fi.write(f'#Jumps                : \n')
        fi.write(f'# Time       Velocity       ErrVel\n')
        for ii in range(0, len(self.vel_time)):
            fi.write(f'{self.vel_time[ii]:.4E} {self.velocity_av[ii]:.4E} {self.velocity_std[ii]:.4E}\n')
        fi.close()


# Functions
def update_streak_im_ROI(streak_im_obj, ROI):
    # Modify streak object
    # Space dimension
    idx_space = np.where((streak_im_obj.space>=ROI[0][0])&(streak_im_obj.space<ROI[0][1]))
    streak_im_obj.npx_space = len(idx_space[0])
    streak_im_obj.space = streak_im_obj.space[idx_space]
    streak_im_obj.space_px = streak_im_obj.space_px[idx_space]
    # Time dimension
    idx_time = np.where((streak_im_obj.time>=ROI[1][0])&(streak_im_obj.time<ROI[1][1]))
    streak_im_obj.npx_time = len(idx_time[0])
    streak_im_obj.time = streak_im_obj.time[idx_time]
    streak_im_obj.time_px = streak_im_obj.time_px[idx_time]
    # Image
    streak_im_obj.im = streak_im_obj.im[idx_time].transpose()[idx_space].transpose()