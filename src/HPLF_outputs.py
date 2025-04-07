"""Extraction of data in HPLF run data folder."""

import numpy as np 
import matplotlib.pyplot as plt 
from gnu_parser import *
import os
from PIL import Image
from HPLF_streaks import *

class HPLF_outputs():
    def __init__(self, data_dir, I0=True, metadata=True, SOP=False):
        # datadir
        self.data_dir = data_dir
        # os.chdir(self.data_dir)
        # Check HPLF sequence type
        if 'streak_ref' in data_dir:
            self.sequence = 'streak_ref'
        elif 'laser_only' in data_dir:
            self.sequence = 'laser_only'
        elif 'shock' in data_dir:
            self.sequence = 'shock'
        elif 'xh_diode' in data_dir:
            self.sequence = 'xh_diode'
        else:
            print('Sequence not known.')
        # metadata
        self.metadata_fi = self.data_dir + os.sep + 'metadata.dat'
        # oscillos
        self.oscillo_dir = self.data_dir + os.sep +'oscillos/'
        self.tektro54_fi = self.oscillo_dir + os.sep + 'tektro54_profile.dat'
        self.tektro64_fi = self.oscillo_dir + os.sep + 'tektro64_profile.dat'
        # p100
        self.p100_dir = self.data_dir + os.sep + 'p100/'
        self.modbox_fi = self.p100_dir + os.sep + 'modbox_profile.dat'
        self.ff_amplifier_fi = self.p100_dir + os.sep + 'ff_amplifier.png'
        self.nf_amplifier_fi = self.p100_dir + os.sep + 'nf_amplifier.png'
        self.ff_intrepid_out_fi = self.p100_dir + os.sep + 'ff_out_frontend_intrepid.png'
        self.nf_intrepid_out_fi = self.p100_dir + os.sep + 'nf_out_frontend_intrepid.png'
        self.nf_intrepid_regen_fi = self.p100_dir + os.sep + 'nf_regen_intrepid.png'
        # cameras eh1
        self.cameras_eh1_dir = self.data_dir + os.sep + 'cameras/'
        self.ff_eh1_fi = self.cameras_eh1_dir + os.sep + 'bas_ff.png'
        self.nf_eh1_fi = self.cameras_eh1_dir + os.sep + 'bas_nf.png'
        # streak cameras 
        self.streak_dir = self.data_dir + os.sep + 'streak/'
        self.visar1_fi = self.streak_dir + os.sep + 'visar1.tif'
        self.visar2_fi = self.streak_dir + os.sep + 'visar2.tif'
        if SOP == True:
            self.sop_fi = self.streak_dir + os.sep + 'sop.tif'

        if self.sequence in ['laser_only', 'shock', 'streak_ref']:
            # Load streak objects
            # VISAR1
            self.VISAR1 = streak_im(self.visar1_fi) 
            # VISAR2
            self.VISAR2 = streak_im(self.visar2_fi) 
            if SOP == True:
                self.SOP = streak_im(self.sop_fi)

        if self.sequence in ['shock', 'xh_diode']:
            # Load XAS data
            if I0 != False:
                self.xas_dir = self.data_dir + os.sep + 'XAS/'
                for XAS_path in os.listdir(self.xas_dir):
                    path_fi = os.path.join(self.xas_dir, XAS_path)
                    if os.path.isfile(path_fi):
                        if 'mu.dat' in path_fi:
                            self.xas_mu_ascii_fi = path_fi
                        if 'I0.dat' in path_fi:
                            self.xas_I0_ascii_fi = path_fi
                        if 'I.dat' in path_fi:
                            self.xas_I_ascii_fi = path_fi

    def read_metadata(self):
        # definition
        temp = grep('definition:', self.metadata_fi)
        self.definition = ' '.join(temp.split()[1:])
        if self.sequence in ['laser_only', 'shock']:
            # energy in eh1
            temp = grep_nlines_after('enmeter_eh1', self.metadata_fi, 1)
            self.energy_eh1 = np.float64(temp[0].split()[1])
            # energy from intrepid
            temp = grep_nlines_after('slink1', self.metadata_fi, 1)
            self.energy_intrepid = np.float64(temp[0].split()[1])
            # phase plate
            temp = grep('phase_plate', self.metadata_fi)
            self.phase_plate = ' '.join(temp.split()[1:])

    def read_oscillos(self):
        if self.sequence in ['laser_only', 'shock']:
            # tektro54
            [self.tektro54_t, self.tektro54_profile] = np.loadtxt(self.tektro54_fi, usecols=(0,3), comments='#', unpack=True)
            # tektro64
            [self.tektro64_t, self.tektro64_profile] = np.loadtxt(self.tektro64_fi, usecols=(0,2), comments='#', unpack=True)

    def read_drive_images(self):
        if self.sequence in ['laser_only', 'shock']:
            # NF and FF in EH1
            self.nf_eh1_im = plt.imread(self.nf_eh1_fi)
            self.ff_eh1_im = plt.imread(self.ff_eh1_fi)

    def plot_drive_profile_eh1(self, delay_oscillo=300, win_oscillo=50):
        if self.sequence in ['laser_only', 'shock']:
            # Read oscillo profiles
            self.read_oscillos()
            # Read NF and FF EH1 images
            self.read_drive_images()
            # Summary figure for drive
            fig = plt.figure(figsize=(10,3))
            # Temporal profile
            ax1 = fig.add_subplot(131)
            ax1.set_title(r'Temporal profile in EH1')
            ax1.set_xlabel(r'Time (ns)')
            ax1.set_ylabel(r'Voltage (mV)')
            ax1.plot(self.tektro64_t*1e9, self.tektro64_profile, 'k-')
            ax1.set_xlim(delay_oscillo, delay_oscillo+win_oscillo)
            # Near field in EH1
            ax2 = fig.add_subplot(132)
            ax2.set_title(r'NF in EH1')
            ax2.imshow(self.nf_eh1_im)
            # Far field in EH1
            ax3 = fig.add_subplot(133)
            ax3.set_title(r'FF in EH1')
            ax3.imshow(self.ff_eh1_im)
            fig.tight_layout()
            plt.show()

    def plot_VISAR_images(self, calibrated=True, cmap='binary', fontsize=12, div_map=3):   
        if self.sequence in ['laser_only', 'shock', 'streak_ref']: 
            # # Calibrate VISAR images
            # self.VISAR1.apply_param_dict(S1_param)
            # self.VISAR2.apply_param_dict(S2_param)
            # Plot visar images
            if calibrated==True: #calibrated images
                fig = plt.figure(figsize=(8, 4))
                # fig.suptitle(f'Calibrated images for sequence {self.sequence}')
                # VISAR1
                ax1 = fig.add_subplot(121)
                # Set tick font size
                for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
                    label.set_fontsize(fontsize)
                ax1.set_title(r'VISAR1')
                ax1.set_xlabel(r'Time (ns)', fontsize=fontsize)
                ax1.set_ylabel(r'Space ($\mu$m)', fontsize=fontsize)
                grid_tV1, grid_sV1 = np.meshgrid(self.VISAR1.time, self.VISAR1.space)
                ax1.pcolormesh(grid_tV1, grid_sV1, self.VISAR1.im.transpose(), vmax=np.max(self.VISAR1.im)/div_map, cmap=cmap, rasterized=True)
                if self.VISAR1.tX!=None:
                    ax1.plot([self.VISAR1.tX, self.VISAR1.tX], [-self.VISAR1.sizeX/2,self.VISAR1.sizeX/2], 'r-')
                # VISAR2
                ax2 = fig.add_subplot(122)
                # Set tick font size
                for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
                    label.set_fontsize(fontsize)
                ax2.set_title(r'VISAR2')
                ax2.set_xlabel(r'Time (ns)', fontsize=fontsize)
                ax2.set_ylabel(r'Space ($\mu$m)', fontsize=fontsize)
                grid_tV2, grid_sV2 = np.meshgrid(self.VISAR2.time, self.VISAR2.space)
                ax2.pcolormesh(grid_tV2, grid_sV2, self.VISAR2.im.transpose(), vmax=np.max(self.VISAR2.im)/div_map, cmap=cmap, rasterized=True)
                if self.VISAR2.tX!=None:
                    ax2.plot([self.VISAR2.tX, self.VISAR2.tX], [-self.VISAR2.sizeX/2,self.VISAR2.sizeX/2], 'r-')
                fig.tight_layout()
                self.VISAR_images = fig
            else:
                fig = plt.figure(figsize=(8, 4))
                # fig.suptitle(f'Raw images for sequence {self.sequence}')
                # VISAR1
                ax1 = fig.add_subplot(121)
                ax1.set_title(r'VISAR1')
                ax1.set_xlabel(r'Time (pixels)')
                ax1.set_ylabel(r'Space (pixels)')
                grid_tV1, grid_sV1 = np.meshgrid(self.VISAR1.time_px, self.VISAR1.space_px)
                ax1.pcolormesh(grid_tV1, grid_sV1, self.VISAR1.im.transpose(), vmax=np.max(self.VISAR1.im)/div_map, cmap=cmap, rasterized=True)
                if self.VISAR1.tX!=None:
                    ax1.plot([self.VISAR1.tX, self.VISAR1.tX], [-self.VISAR1.sizeX/2,self.VISAR1.sizeX/2], 'r-')
                # VISAR2
                ax2 = fig.add_subplot(122)
                ax2.set_title(r'VISAR2')
                ax2.set_xlabel(r'Time (pixels)')
                ax2.set_ylabel(r'Space (pixels)')
                grid_tV2, grid_sV2 = np.meshgrid(self.VISAR2.time_px, self.VISAR2.space_px)
                ax2.pcolormesh(grid_tV2, grid_sV2, self.VISAR2.im.transpose(), vmax=np.max(self.VISAR2.im)/div_map, cmap=cmap, rasterized=True)
                if self.VISAR2.tX!=None:
                    ax2.plot([self.VISAR2.tX, self.VISAR2.tX], [-self.VISAR2.sizeX/2,self.VISAR2.sizeX/2], 'r-')
                fig.tight_layout()
            plt.show()

    def plot_SOP_image(self, calibrated=True, cmap='binary', fontsize=12, div_map=3):   
        if self.sequence in ['laser_only', 'shock', 'streak_ref']: 
            # # Calibrate SOP image
            # self.SOP.apply_param_dict(S3_param)
            # Plot image
            if calibrated==True: #calibrated image
                fig = plt.figure(figsize=(7, 4))
                # SOP
                ax1 = fig.add_subplot(121)
                # Set tick font size
                for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
                    label.set_fontsize(fontsize)
                ax1.set_title(r'SOP')
                ax1.set_xlabel(r'Time (ns)', fontsize=fontsize)
                ax1.set_ylabel(r'Space ($\mu$m)', fontsize=fontsize)
                grid_tSOP, grid_sSOP = np.meshgrid(self.SOP.time, self.SOP.space)
                ax1.pcolormesh(grid_tSOP, grid_sSOP, self.SOP.im.transpose(), vmax=np.max(self.SOP.im)/div_map, cmap=cmap, rasterized=True)
                if self.SOP.tX!=None:
                    ax1.plot([self.SOP.tX, self.SOP.tX], [-self.SOP.sizeX/2,self.SOP.sizeX/2], 'r-')
                # ax2 = fig.add_subplot(122)
                # ax2.set_title(r'SOP')
                # ax2.set_xlabel(r'Time (ns)', fontsize=fontsize)
                # ax2.set_ylabel(r'Space ($\mu$m)', fontsize=fontsize)
                # ax2.pcolormesh(grid_tSOP, grid_sSOP, self.SOP.im.transpose(), vmax=np.max(self.SOP.im)/div_map, cmap=cmap, rasterized=True)
                # if self.SOP.tX!=None:
                #     ax2.plot([self.SOP.tX, self.SOP.tX], [-self.SOP.sizeX/2,self.SOP.sizeX/2], 'r-')
                fig.tight_layout()
                self.SOP_image = fig
            else:
                fig = plt.figure(figsize=(8, 4))
                # SOP
                ax1 = fig.add_subplot(121)
                ax1.set_title(r'SOP')
                ax1.set_xlabel(r'Time (pixels)')
                ax1.set_ylabel(r'Space (pixels)')
                grid_tSOP, grid_sSOP = np.meshgrid(self.SOP.time_px, self.SOP.space_px)
                ax1.pcolormesh(grid_tSOP, grid_sSOP, self.SOP.im.transpose(), vmax=np.max(self.SOP.im)/div_map, cmap=cmap, rasterized=True)
                if self.SOP.tX!=None:
                    ax1.plot([self.SOP.tX, self.SOP.tX], [-self.SOP.sizeX/2,self.SOP.sizeX/2], 'r-')
                fig.tight_layout()
                self.SOP_image = fig
            plt.show()

    def read_XAS(self, nb_frame, shot_frame, calibrated=True, calib_coefs=None, start_frame=2):
        self.nb_frame = nb_frame
        self.shot_frame = shot_frame
        cols_np = np.linspace(0, int(nb_frame)+2, int(nb_frame)+3).astype(int)
        cols = tuple(cols_np)
        # mu
        data_mu = np.loadtxt(self.xas_mu_ascii_fi, usecols=cols, comments='#', unpack=True)
        self.XAS_px = data_mu[0]
        self.XAS_E = data_mu[1]
        self.XAS_data_mu = data_mu[2:]
        if calibrated == False:
            if calib_coefs is not None:
                [c, b, a] = calib_coefs
                self.XAS_E = a + b*self.XAS_px + c*self.XAS_px**2
        # I0
        self.XAS_data_I0 = np.loadtxt(self.xas_I0_ascii_fi, usecols=(2), comments='#', unpack=True)
        # I
        data_I = np.loadtxt(self.xas_I_ascii_fi, usecols=cols, comments='#', unpack=True)
        self.start_frame = start_frame
        self.XAS_data_I = data_I[2:]

    def normalize_XAS(self, Ea_range, Eb_range):
        # Normalization with simple jump
        self.XAS_data_mu_norm = np.zeros_like(self.XAS_data_mu)
        for ii in range(0, len(self.XAS_data_mu)):
            temp = self.XAS_data_mu[ii] - np.average(self.XAS_data_mu[ii][ (self.XAS_E>Ea_range[0])&(self.XAS_E<Ea_range[1]) ])
            self.XAS_data_mu_norm[ii] = temp / np.average(temp[ (self.XAS_E>Eb_range[0])&(self.XAS_E<Eb_range[1]) ])
        
    def plot_XAS(self, xlim=None, ylim1=(0, 66000), ylim2=(-0.2, 2.5)):
        if xlim is None:
            xlim = (np.min(self.XAS_E), np.max(self.XAS_E))
        fig = plt.figure(figsize=(10, 5))
        # fig.suptitle(f'Calibrated images for sequence {self.sequence}')
        # I0 and I1
        ax1 = fig.add_subplot(121)
        ax1.set_title(r'Intensity')
        ax1.set_xlabel(r'Energy (eV)')
        ax1.set_ylabel(r'Intensity (counts)')
        ax1.plot(self.XAS_E, self.XAS_data_I0, 'k-', label='I0')
        ax1.plot(self.XAS_E, np.average(self.XAS_data_I[self.start_frame:int(self.shot_frame)-1], axis=0), '-', color='darkcyan')
        ax1.fill_between(self.XAS_E, np.min(self.XAS_data_I[self.start_frame:int(self.shot_frame)-1], axis=0), np.max(self.XAS_data_I[self.start_frame:int(self.shot_frame)-1], axis=0), color='darkcyan', alpha=0.5)
        ax1.plot(self.XAS_E, self.XAS_data_I[int(self.shot_frame)], '-', color='red')
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim1)
        # mu
        ax2 = fig.add_subplot(122)
        ax2.set_title(r'Spectra')
        ax2.set_xlabel(r'Energy (eV)')
        ax2.set_ylabel(r'$\mu$')
        ax2.plot(self.XAS_E, np.average(self.XAS_data_mu_norm[self.start_frame:int(self.shot_frame)-1], axis=0), '-', color='darkcyan')
        ax2.fill_between(self.XAS_E, np.min(self.XAS_data_mu_norm[self.start_frame:int(self.shot_frame)-1], axis=0), np.max(self.XAS_data_mu_norm[self.start_frame:int(self.shot_frame)-1], axis=0), color='darkcyan', alpha=0.5)
        ax2.plot(self.XAS_E, self.XAS_data_mu_norm[int(self.shot_frame)], '-', color='red')
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim2)
        fig.tight_layout()
        self.XAS_image = fig
        plt.show()

    def export_XAS(self, pathout):
        mu_cold_min = np.min(self.XAS_data_mu_norm[2:int(self.shot_frame)-1], axis=0)
        mu_cold_max = np.max(self.XAS_data_mu_norm[2:int(self.shot_frame)-1], axis=0)
        mu_cold_av = np.average(self.XAS_data_mu_norm[2:int(self.shot_frame)-1], axis=0)
        mu_hot =  self.XAS_data_mu_norm[int(self.shot_frame)]
        fi = open(pathout, 'w')
        fi.write('# Energy (eV)\tmu_cold_av\tmu_cold_min\tmu_cold_max\tmu_hot\n')
        for ii in range(0, len(self.XAS_E)):
            fi.write(f'{self.XAS_E[ii]}\t{mu_cold_av[ii]}\t{mu_cold_min[ii]}\t{mu_cold_max[ii]}\t{mu_hot[ii]}\n')
        fi.close()
        print(f'--> {pathout} exported.')
