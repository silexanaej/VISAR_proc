"""Extraction of data in EuXFEL output folders (early version)."""

import numpy as np 
import matplotlib.pyplot as plt 
from gnu_parser import *
import os
from PIL import Image
from HPLF_streaks import *

class EuXFEL_outputs():
    def __init__(self, data_dir, im1_name, im2_name):
        # datadir
        self.data_dir = data_dir
        # os.chdir(self.data_dir)
        # Look for VISAR images
        self.visar1_fi = self.data_dir + os.sep + im1_name
        self.visar2_fi = self.data_dir + os.sep + im2_name
        self.VISAR1 = streak_im(self.visar1_fi, streak_model="Sydor")
        self.VISAR2 = streak_im(self.visar2_fi, streak_model="Sydor")

    def plot_VISAR_images(self, S1_param, S2_param, calibrated=True, cmap='binary', div=3):   
        # Calibrate VISAR images
        self.VISAR1.apply_param_dict(S1_param)
        self.VISAR2.apply_param_dict(S2_param)
        # Plot visar images
        if calibrated==True: #calibrated images
            fig = plt.figure(figsize=(10, 5))
            # fig.suptitle(f'Calibrated images for sequence {self.sequence}')
            # VISAR1
            ax1 = fig.add_subplot(121)
            ax1.set_title(r'VISAR1')
            ax1.set_xlabel(r'Time (ns)')
            ax1.set_ylabel(r'Space ($\mu$m)')
            grid_tV1, grid_sV1 = np.meshgrid(self.VISAR1.time, self.VISAR1.space)
            ax1.pcolormesh(grid_tV1, grid_sV1, self.VISAR1.im.transpose(), cmap=cmap,vmin=np.min(self.VISAR1.im), vmax=np.max(self.VISAR1.im)/div) # 
            if self.VISAR1.tX!=None:
                ax1.plot([self.VISAR1.tX, self.VISAR1.tX], [-self.VISAR1.sizeX/2,self.VISAR1.sizeX/2], 'r-')
            # VISAR2
            ax2 = fig.add_subplot(122)
            ax2.set_title(r'VISAR2')
            ax2.set_xlabel(r'Time (ns)')
            ax2.set_ylabel(r'Space ($\mu$m)')
            grid_tV2, grid_sV2 = np.meshgrid(self.VISAR2.time, self.VISAR2.space)
            ax2.pcolormesh(grid_tV2, grid_sV2, self.VISAR2.im.transpose(), cmap=cmap, vmin=np.min(self.VISAR2.im), vmax=np.max(self.VISAR2.im)/div) #
            if self.VISAR2.tX!=None:
                ax2.plot([self.VISAR2.tX, self.VISAR2.tX], [-self.VISAR2.sizeX/2,self.VISAR2.sizeX/2], 'r-')
            fig.tight_layout()
            plt.show()
        
        else:
            fig = plt.figure(figsize=(10, 5))
            # fig.suptitle(f'Calibrated images for sequence {self.sequence}')
            # VISAR1
            ax1 = fig.add_subplot(121)
            ax1.set_title(r'VISAR1')
            ax1.set_xlabel(r'Time (ns)')
            ax1.set_ylabel(r'Space ($\mu$m)')
            grid_tpxV1, grid_spxV1 = np.meshgrid(self.VISAR1.time_px, self.VISAR1.space_px)
            ax1.pcolormesh(grid_tpxV1, grid_spxV1, self.VISAR1.im.transpose(), cmap=cmap,vmin=np.min(self.VISAR1.im), vmax=np.max(self.VISAR1.im)/div) # 
            if self.VISAR1.tX!=None:
                ax1.plot([self.VISAR1.tX, self.VISAR1.tX], [-self.VISAR1.sizeX/2,self.VISAR1.sizeX/2], 'r-')
            # VISAR2
            ax2 = fig.add_subplot(122)
            ax2.set_title(r'VISAR2')
            ax2.set_xlabel(r'Time (ns)')
            ax2.set_ylabel(r'Space ($\mu$m)')
            grid_tpxV2, grid_spxV2 = np.meshgrid(self.VISAR2.time_px, self.VISAR2.space_px)
            ax2.pcolormesh(grid_tpxV2, grid_spxV2, self.VISAR2.im.transpose(), cmap=cmap, vmin=np.min(self.VISAR2.im), vmax=np.max(self.VISAR2.im)/div) #
            if self.VISAR2.tX!=None:
                ax2.plot([self.VISAR2.tX, self.VISAR2.tX], [-self.VISAR2.sizeX/2,self.VISAR2.sizeX/2], 'r-')
            fig.tight_layout()
            plt.show()      
