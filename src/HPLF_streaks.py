"""Class for streak images."""

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage import rotate
from PIL import Image
from struct import unpack

class streak_im():
    
    def __init__(self, path, streak_model="HM C7700", scaling_factor=1):
        self.path = path
        if self.path.endswith('.img'):
            [self.header,self.im,self.error_flag] = _reader_img(self.path)
        elif self.path.endswith('.tif'):
            [self.header,self.im] = _reader_tif(self.path)
        elif self.path.endswith('.tiff'):
            [self.header,self.im] = _reader_tif(self.path)
        else:
            print('Format not recognized.')

        self.streak_model = streak_model

        if self.streak_model == "Sydor":
            self.im = self.im.transpose()
            self.npx_time = self.header['Image_width']
            self.npx_space = self.header['Image_height']
        else:
            # self.streak_model = 'HM C7700'
            self.npx_time = self.header['Image_height']
            self.npx_space = self.header['Image_width']
            self.slit_size = 17.48 * 1e3 # in microns

        self.time_px = np.arange(0, self.npx_time, 1)
        self.space_px = np.arange(0, self.npx_space, 1)
        self.t0_px = 0 # for initialisation
        self.pos0_px = 0 # for initialisation
        self.background_level = 110 # intensity with closed shutter
        self.delay = 0 # in ns

    def set_time_calibration(self, HM_poly):
        """
        HM_poly: array with coefs in the following order [A1/1000, A2/1000, A3/1000, A4/1000] (in ns)
        """
        self.time_calib_HM_coefs = HM_poly
        self.A1, self.A2, self.A3, self.A4 = self.time_calib_HM_coefs
        self.time = np.cumsum(self.A1 + self.A2*self.time_px + self.A3*self.time_px**2 + self.A4*self.time_px**3)
        self.time = self.time - self.time[self.t0_px]
        
    def set_t0_px(self, t0_px):
        self.t0_px = t0_px
        
    def set_delay(self, delay):
        self.delay = delay
        self.time = self.time + self.delay
    
    def set_space_calibration(self, fov):
        """
        fov: field of view on streak slit in microns
        """
        self.fov = fov
        self.space = (self.space_px-self.pos0_px)*fov/self.npx_space
        
    def set_pos0_px(self, pos0_px):
        self.pos0_px = pos0_px
        
    def rotate_im(self, angle):
        """
        Angle in degrees.
        """
        self.im = rotate(self.im, angle, reshape=False, cval=np.average(self.im))
        try:
            self.set_time_calibration(self.time_calib_HM_coefs) # updating time calibration if any
        except AttributeError:
            pass
        try:
            self.set_space_calibration(self.space_calib) # updating time calibration if any
        except AttributeError:
            pass

    def flipspace_im(self):
        self.im = np.fliplr(self.im)
        try:
            self.set_time_calibration(self.time_calib_HM_coefs) # updating time calibration if any
        except AttributeError:
            pass
        try:
            self.set_space_calibration(self.space_calib) # updating time calibration if any
        except AttributeError:
            pass
            
    def set_tX(self, tX=None):
        self.tX = tX
        
    def set_sizeX(self, sizeX=15):
        self.sizeX = sizeX
        
    def apply_param_dict(self, param_dict):
        # Used to modify streak parameters, order must be kept
        self.set_t0_px(param_dict['t0_px'])
        self.set_pos0_px(param_dict['pos0_px'])
        if 'poly_coefs' not in param_dict.keys():
            self.set_time_calibration(get_HM_poly(param_dict['name'], param_dict['window'])) # names are 'S1' or 'S2', for HPLF only
        else:
            self.set_time_calibration(param_dict['poly_coefs'])
        self.set_tX(tX=param_dict['tX'])
        self.set_delay(param_dict['delay_to_drive'])
        self.set_space_calibration(param_dict['fov'])
        self.set_sizeX(sizeX=param_dict['sizeX'])
        self.background_level = param_dict['counts_bkg']

    def plot_calibrated_im(self):
        T, S = np.meshgrid(self.time, self.space)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(r'Space ($\mu$m)')
        ax.pcolormesh(T, S, self.im.transpose(), vmax=np.max(self.im)/3, cmap='binary')
        if self.tX!=None:
            ax.plot([self.tX, self.tX], [-self.sizeX/2,self.sizeX/2], 'r-')
        fig.tight_layout()
        plt.show()

    def plot_raw_im(self):
        Y, X = np.meshgrid(self.time_px, self.space_px)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time (pixels)')
        ax.set_ylabel(r'Space (pixels)')
        ax.pcolormesh(Y, X, self.im.transpose(), vmax=np.max(self.im)/3, cmap='binary')
        fig.tight_layout()
        plt.show()


# Readers for .img and .tif images
def _reader_img(image_path):
    """Reads an image of format ITEX with file extension *.img, from Hamamatsu devices (DPC, HPD-TA)
    """
       
    img_header = {}

    try:
        fid = open(image_path,mode='rb')

        img_header['Version'] = fid.read(2).decode('ascii')
        if img_header['Version'] != 'IM':
            print('Warning: Not a valid Hamamatsu IMG image file')
        car = unpack('h',fid.read(2))
        comment_length = car[0]

        car = unpack(30*'h',fid.read(60))
        nc,nl =  car[0],car[1]
        img_header['Image_width'] = nc
        img_header['Image_height'] = nl
        img_header['X_Offset'] = car[2]
        img_header['Y_Offset'] = car[3]
        if car[4] == 0:
            img_header['File_type'] = '8 Bit'
            typ = 'uint8'
        elif car[4] == 1:
            img_header['File_type'] = 'Compressed'
            typ = 'None'
        elif car[4] == 2:
            img_header['File_type'] = '16 Bit'
            typ = 'uint16'
        elif car[4] == 3:
            img_header['File_type'] = '32 Bit'
            typ = 'uint32'
        else :
            img_header['File_type'] = 'Unkwown'
            typ = 'None'
        img_header['Reserved'] = car[5:30]

        comment = fid.read(comment_length).decode('ascii')
        comment = comment[0]+comment[1:].replace('[','\n[') # separate each section by a newline
        comment = comment.replace('\r','\n')
        comment_list = comment.split('\n')
        while comment_list.count('') != 0: # suppress empty lines in the comment list
            index = comment_list.index('')
            comment_list.pop(index)
        img_header['Comment'] = comment_list

#       Image data
        ima = np.fromfile(fid,dtype=typ,count=nl*nc)
        ima = ima.reshape((nl,nc))

        fid.close()

        error_flag = 0

    except:
        ima = np.empty((1,1),dtype=None)
        error_flag = -1

    return [img_header,ima,error_flag]

def _reader_tif(path):
    im = Image.open(path)
    im = np.array(im)
    header = {'Image_height':len(im), 'Image_width':len(im[0])}
    return [header, im]

# HPLF streak cameras time calibration HM coefficients
def get_HM_poly(name, window):
    # time calibration coefficients for S1
    S1_poly_t = {
            '5ns':np.array([5.580e00, -7.286e-04, 3.412e-07, -8.646e-10])*1e-3,
            '10ns':np.array([9.032e00, 3.519e-03, -1.154e-05, 7.029e-09])*1e-3,
            '20ns':np.array([1.673e-02, 1.447e-05, -1.715e-08, 8.393e-12]),
            '50ns':np.array([5.595e01, -1.635e-02, 1.763e-05, -6.324e-09])*1e-3,
            '100ns':np.array([9.471e01, 5.057e-03, -1.845e-06, 1.653e-09])*1e-3
            }
    # time calibration coefficients for S2
    S2_poly_t = {
            '5ns':np.array([5.402e00, -3.773e-04, 3.667e-09, -7.660e-10])*1e-3,
            '10ns':np.array([1.100e01, 3.115e-03, -5.205e-06, 1.931e-09])*1e-3,
            '20ns':np.array([1.677e-02, 4.554e-06, 8.857e-10, -1.642e-12]),
            '50ns':np.array([5.083e01, -9.930e-03, 9.126e-06, -1.869e-09])*1e-3,
            '100ns':np.array([9.420e01, 3.780e-03, 7.927e-06, -4.947e-09])*1e-3
            }
    # time calibration coefficients for S3
    S3_poly_t = {
            '5ns':np.array([5.400e00, -2.307e-03, 1.402e-06, 0.0])*1e-3,
            '10ns':np.array([1.051e01, -3.098e-03, 1.575e-06, 0.0])*1e-3,
            '20ns':np.array([1.751e01, 1.308e-03, 4.591e-07, 0.0])*1e-3,
            '50ns':np.array([5.000e01, -6.802e-03, 5.751e-06, 0.0])*1e-3,
            '100ns':np.array([9.224e01, -3.003e-02, 4.019e-05, 0.0])*1e-3
            }
    # top level dictionary with streak names as keys
    poly_t = {'S1': S1_poly_t, 'S2':S2_poly_t, 'S3':S3_poly_t}
    return poly_t[name][window]