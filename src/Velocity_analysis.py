"""Velocity trace analysis tools."""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.fftpack as sc_fft
from HPLF_streaks import *
from HPLF_visar import *
from matplotlib.colors import LogNorm
from scipy.optimize import fsolve
from scipy import interpolate

# Some function for MC
def randomize(mu, sigma, seed):
    np.random.seed(seed)
    n = mu + sigma*np.random.randn()
    return n

def randomize_array(vec_mu, vec_sigma, seed):
    np.random.seed(seed)
    vec_n = vec_mu + vec_sigma*np.random.randn(len(vec_mu))
    return vec_n

# EOS and Hugoniot related functions
def rho_from_P_Murnaghan(P, P0, rhoP0, B0, Bp):
    # all variables are ufloat
    rho = rhoP0*( (B0+Bp*P)/(B0+Bp*P0) )**(1/Bp)
    return rho

def P_from_RH(Up, Us, rho0, P0):
    P = P0 + rho0*Up*Us
    return P

def rho_from_RH(Up, Us, rho0):
    rho = rho0*Us/(Us-Up)
    return rho

def Us_from_Up_polynomial(Up, coefs, cov, seed=None):
    """"
    Inputs:
    - Up can be a scalar or a 1D-array
    - coefs in a 1D-array of size N>=2 and coefs are arranged from the N-order coef. to the constant
    - cov is a 2D-array with shape (N, N)
    - seed is the seed for Monte Carlo error propagation, if seed is None, the average value is returned.
    Output:
    - Us as a 1D-array
    """
    if seed is None:
        poly = np.poly1d(coefs)
        return poly(Up)
    else:
        coefs_MC = np.random.multivariate_normal(coefs, cov, 1).T
        poly = np.poly1d(coefs_MC)
        return poly(Up)
    
def get_intersection(func1, func2, x_guess):
    func_intersect = lambda x : func1(x) - func2(x)
    x_intersect = fsolve(func_intersect, x_guess)[0]
    return x_intersect


# Materials database
class LiF():
    # LiF properties from Hawreliak et al., PRB 2023
    def __init__(self):
        self.name = 'LiF'
        self.rho0 = 2.64
        self.P0 = 0
        self.UsUp_coefs = np.array([1.355, 5.144])
        self.UsUp_cov = np.array([[0.004**2, 0], [0, 0.010**2]])
        self.Vp0 = 6.57
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
        return Up

class BKapton():
    # Black Kapton properties from Katagiri et al., PRB 2022
    def __init__(self):
        self.name = 'BKapton'
        self.rho0 = 1.415
        self.P0 = 0
        self.Vp0 = 2.327
        self.Up_transition_12 = 2.78
        self.UsUp_coefs_1 = np.array([1.55, 2.327])
        self.UsUp_cov_1 = np.array([[0, 0], [0, 0.06**2]])
        self.UsUp_coefs_2 = np.array([1.43, 1.79])
        self.UsUp_cov_2 = np.array([[0.02**2, 0], [0, 0.11**2]])
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        Us = np.copy(Up)
        Us[Up < self.Up_transition_12] = Us_from_Up_polynomial(Up[Up < self.Up_transition_12], self.UsUp_coefs_1, self.UsUp_cov_1)
        Us[Up >= self.Up_transition_12] = Us_from_Up_polynomial(Up[Up >= self.Up_transition_12], self.UsUp_coefs_2, self.UsUp_cov_2)
        return Us
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        # Dealing with overlapping region
        Us_temp1 = self.Us_from_Up(np.array([self.Up_transition_12-0.001]))
        Us_temp2 = self.Us_from_Up(np.array([self.Up_transition_12]))
        Us_dw_min = np.min([Us_temp1[0], Us_temp2[0]])
        Us_dw_max = np.max([Us_temp1[0], Us_temp2[0]])
        Up[Us < Us_dw_max] = (Us -  self.UsUp_coefs_1[1]) / self.UsUp_coefs_1[0]
        Up[Us >= Us_dw_max] = (Us -  self.UsUp_coefs_2[1]) / self.UsUp_coefs_2[0]
        return Up
    
class GeO2glass():
    # GeO2 glass properties from Andrea's report
    def __init__(self):
        self.name = 'GeO2glass'
        self.rho0 = 3.65
        self.P0 = 0
        self.UsUp_coefs = np.array([1.40, 1.59])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 3.75 # km/s
        self.shock_impedance = self.Vp0*self.rho0 # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
        return Up    

class Copper():
    # Copper Hugoniot from McCoy et al., PRB (2017)
    def __init__(self):
        self.name = 'Copper'
        self.rho0 = 8.93
        self.P0 = 0
        self.UsUp_coefs = np.array([1.413, 4.272])
        self.UsUp_cov = np.array([[2.315e-4, -1.116e-3], [-1.116e-3, 5.964e-3]])
        self.Vp0 = 4.67  # km/s
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
        return Up
    
class Cobalt():
    def __init__(self):
        self.name = 'Cobalt'
        self.rho0 = 8.82
        self.P0 = 0
        self.UsUp_coefs = np.array([1.28, 4.77])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 5.73  # km/s
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
            return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
            Up = np.copy(Us)
            Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
            return Up

class Nickel():
    # From Marsh 1980
    def __init__(self):
        self.name = 'Nickel'
        self.rho0 = 8.875
        self.P0 = 0
        self.UsUp_coefs = np.array([1.4, 4.59])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 5.79  # km/s
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
            return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
            Up = np.copy(Us)
            Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
            return Up
    
class Molybdenum():
    # Linear Hugoniot fitted to sesame 2981, fit valid up to Up = 4 km/s
    def __init__(self):
        self.name = 'Molybdenum'
        self.rho0 = 10.22
        self.P0 = 0
        self.UsUp_coefs = np.array([1.2656, 5.0936])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 6.3  # km/s
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
            return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
            Up = np.copy(Us)
            Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
            return Up
    
class CuZrAlglass():
    # From Marsh 1980
    def __init__(self):
        self.name = 'CuZrAlglass'
        self.rho0 = 6.91
        self.P0 = 0
        self.UsUp_coefs = np.array([0.7244, 5.0583])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 4.95  # km/s
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
            return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
            Up = np.copy(Us)
            Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
            return Up

class MgSiO3glass():
    # Polynomial fit Militzer (2013) + gas gun data
    def __init__(self):
        self.name = 'MgSiO3glass'
        self.rho0 = 2.74
        self.P0 = 0
        self.UsUp_coefs = np.array([-0.01458547,  1.59185749,  3.30201729])
        self.UsUp_cov = np.array([[ 4.87285239e-05, -3.64004816e-04, 5.62073157e-04],
               [-3.64004816e-04,  3.10767135e-03, -5.55906957e-03],
               [ 5.62073157e-04, -5.55906957e-03,  1.21636356e-02]])
        self.Vp0 = 3.5  # km/s
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        a = (np.copy(Us)*0+1)*self.UsUp_coefs[0]
        b = (np.copy(Us)*0+1)*self.UsUp_coefs[1]
        c = (np.copy(Us)*0+1)*self.UsUp_coefs[2]
        delta = b**2 - 4*a*(c-Us)
        Up = (-b + np.sqrt(delta))/(2*a)
        return Up
    
class MgSiO3enstatite():
    # Linear fit LULI + gas gun data
    def __init__(self):
        self.name = 'MgSiO3enstatite'
        self.rho0 = 3.2
        self.P0 = 0
        self.UsUp_coefs = np.array([1.37, 4.75])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 5.78  # km/s, estimated
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
        return Up
    
class FeO():
    # Linear relation from Jeanloz and Ahrens 1980
    def __init__(self):
        self.name = 'FeO'
        self.rho0 = 5.74
        self.P0 = 0
        self.UsUp_coefs = np.array([1.45, 4.27])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 6.08  # km/s, average
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
        return Up

class Iron():
    # Rough linear fit of plastic Hugoniot, valid for Up > 1 and < 4.5 km/s
    def __init__(self):
        self.name = 'Iron'
        self.rho0 = 7.874
        self.P0 = 0
        self.UsUp_coefs = np.array([1.609, 3.892])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 5.9  # km/s
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
        return Up

class Iridium():
    # Rough linear fit of iridium Hugoniot
    def __init__(self):
        self.name = 'Iridium'
        self.rho0 = 22.5
        self.P0 = 0
        self.UsUp_coefs = np.array([1.3196, 4.0645])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 3.7  # km/s
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
        return Up
    
class Fayalite():
    # Linear fit from Thomas et al. JGR 2012
    def __init__(self):
        self.name = 'Fayalite'
        self.rho0 = 4.375
        self.P0 = 0
        self.UsUp_coefs = np.array([1.58, 2.438])
        self.UsUp_cov = np.array([[0, 0], [0, 0]])
        self.Vp0 = 5.34  # km/s, estimated
        self.shock_impedance = self.Vp0*self.rho0  # at ambient only (self.Us*self.rho0 otherwise)
    def Us_from_Up(self, Up):
        return Us_from_Up_polynomial(Up, self.UsUp_coefs, self.UsUp_cov)
    def Up_from_Us(self, Us):
        Up = np.copy(Us)
        Up = (Us -  self.UsUp_coefs[1]) / self.UsUp_coefs[0]
        return Up

class impedance_matching():

    def __init__(self, mat1, mat2, measurement_type, vecUp, Umeasured, plot):
        
        if mat1.shock_impedance < mat2.shock_impedance:
            # mat1 is reshocked by mat2
            if (measurement_type == 'Up_2') or (measurement_type == 'Us_2'):
                # we want Up_1 so that mat1 reversed Hugoniot intersects mat2 Hugoniot at Up_2
                if measurement_type == 'Us_2':
                    # need to compute Up from Us before impedance matching
                    # Up_from_Umeasured = 
                    # self.IM_mat1_inf_mat2_measUp2(mat1, mat2, vecUp, Up_from_Umeasured) # done
                    print()
                else:
                    # Calculation of impedance
                    self.IM_mat1_inf_mat2_measUp2(mat1, mat2, vecUp, Umeasured) # done
            if (measurement_type == 'Up_1') or (measurement_type == 'Us_1'):
                # we want Up_2 so that mat1 reversed Hugoniot intersects mat2 Hugoniot
                if measurement_type == 'Us_1':
                    # need to compute Up from Us before impedance matching
                    print()
                else:
                    # Calculation of impedance
                    [self.P_mat2_intersect, self.Up_mat2_intersect] = self.IM_mat1_inf_mat2_measUp1(mat1, mat2, vecUp, Umeasured, x_guess=5, plot=plot)
        if mat1.shock_impedance > mat2.shock_impedance:
            # mat1 releases into mat2
            if (measurement_type == 'Up_2') or (measurement_type == 'Us_2'):
                # we want Up_1 so that mat1 release intersects mat2 Hugoniot at Up_2
                # mat1 reversed Hugoniot can be used if impedance difference not too large
                if measurement_type == 'Us_2':
                    # need to compute Up from Us before impedance matching
                    print()
                else:
                    # Calculation of impedance
                    self.IM_mat1_sup_mat2_measUp2(mat1, mat2, vecUp, Umeasured)
            if (measurement_type == 'Up_1') or (measurement_type == 'Us_1'):
                # we want Up_2 so that mat1 release or reversed Hugoniot intersects mat2 Hugoniot
                if measurement_type == 'Us_1':
                    # need to compute Up from Us before impedance matching
                    print()
                # Calculation of impedance

    def IM_mat1_inf_mat2_measUp2(self, mat1, mat2, vecUp, Up_2, x_guess=5):
        
        # Hugoniot pressures
        mat1_P = mat1.rho0*vecUp*mat1.Us_from_Up(vecUp)
        mat2_P = mat2.rho0*vecUp*mat2.Us_from_Up(vecUp)

        # mat1 reversed Hugoniot that intersects mat2 Hugoniot at Up_2
        mat1_P_r = np.flip(mat1_P)

        # Interpolation to find intersection
        P1 = interpolate.interp1d(vecUp, mat1_P, kind='cubic', fill_value='extrapolate')
        P1r = interpolate.interp1d(vecUp, mat1_P_r, kind='cubic', fill_value='extrapolate')
        P2 = interpolate.interp1d(vecUp, mat2_P, kind='cubic', fill_value='extrapolate')

        P2_measured = P2(vecUp)[np.where(vecUp==Up_2)][0]

        func_P1rP2meas_intersect = lambda x : P1r(x) - P2_measured
        Up_temp = fsolve(func_P1rP2meas_intersect, x_guess)[0]

        new_vecUp = vecUp + Up_2 - Up_temp
        P1r = interpolate.interp1d(new_vecUp, mat1_P_r, kind='cubic', fill_value='extrapolate')
        
        func_P1rP1_intersect = lambda x : P1r(x) - P1(x)
        Up_mat1_intersect = fsolve(func_P1rP1_intersect, x_guess)[0]
        P_mat1_intersect = P1r(Up_mat1_intersect)

        print(f'*** Results of impedance matching ***\n')
        print(f'Conditions measured in {mat2.name}:')
        print(f'\tUp measured in {mat2.name} = {Up_2:.3f} km/s')
        print(f'\tP in {mat2.name} = {P2_measured:.1f} GPa\n')
        print(f'Conditions measured in {mat1.name}:')
        print(f'\tUp in {mat1.name} = {Up_mat1_intersect:.3f} km/s')
        print(f'\tP in {mat1.name} = {P_mat1_intersect:.1f} GPa')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Particle velocity (km/s)')
        ax.set_ylabel('Pressure (GPa)')
        ax.plot(vecUp, mat1_P, 'k-', label=f'Hug. {mat1.name}')
        ax.plot(vecUp, mat2_P, 'r-', label=f'Hug. {mat2.name}')
        ax.plot(new_vecUp, P1r(new_vecUp), 'k--', label=f'Rev. Hug. {mat1.name}')
        ax.plot(Up_2, P2_measured, 'ro', label=f'Conditions measured in {mat2.name}')
        ax.plot(Up_mat1_intersect, P_mat1_intersect, 'ko', label=f'Conditions in {mat1.name}')
        fig.legend()
        fig.tight_layout()
        plt.show()

    def IM_mat1_sup_mat2_measUp2(self, mat1, mat2, vecUp, Up_2, x_guess=5):
        # Only reversed Hugoniot so far
        
        # Hugoniot pressures
        mat1_P = mat1.rho0*vecUp*mat1.Us_from_Up(vecUp)
        mat2_P = mat2.rho0*vecUp*mat2.Us_from_Up(vecUp)

        # mat1 reversed Hugoniot that intersects mat2 Hugoniot at Up_2
        mat1_P_r = np.flip(mat1_P)

        # Interpolation to find intersection
        P1 = interpolate.interp1d(vecUp, mat1_P, kind='cubic', fill_value='extrapolate')
        P1r = interpolate.interp1d(vecUp, mat1_P_r, kind='cubic', fill_value='extrapolate')
        P2 = interpolate.interp1d(vecUp, mat2_P, kind='cubic', fill_value='extrapolate')

        P2_measured = P2(vecUp)[np.where(vecUp==Up_2)][0]

        func_P1rP2meas_intersect = lambda x : P1r(x) - P2_measured
        Up_temp = fsolve(func_P1rP2meas_intersect, x_guess)[0]

        new_vecUp = vecUp + Up_2 - Up_temp
        P1r = interpolate.interp1d(new_vecUp, mat1_P_r, kind='cubic', fill_value='extrapolate')
        
        func_P1rP1_intersect = lambda x : P1r(x) - P1(x)
        Up_mat1_intersect = fsolve(func_P1rP1_intersect, x_guess)[0]
        P_mat1_intersect = P1r(Up_mat1_intersect)

        print(f'*** Results of impedance matching ***\n')
        print(f'Conditions measured in {mat2.name}:')
        print(f'\tUp measured in {mat2.name} = {Up_2:.3f} km/s')
        print(f'\tP in {mat2.name} = {P2_measured:.1f} GPa\n')
        print(f'Conditions measured in {mat1.name}:')
        print(f'\tUp in {mat1.name} = {Up_mat1_intersect:.3f} km/s')
        print(f'\tP in {mat1.name} = {P_mat1_intersect:.1f} GPa')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Particle velocity (km/s)')
        ax.set_ylabel('Pressure (GPa)')
        ax.plot(vecUp, mat1_P, 'k-', label=f'Hug. {mat1.name}')
        ax.plot(vecUp, mat2_P, 'r-', label=f'Hug. {mat2.name}')
        ax.plot(new_vecUp, P1r(new_vecUp), 'k--', label=f'Rev. Hug. {mat1.name}')
        ax.plot(Up_2, P2_measured, 'ro', label=f'Conditions measured in {mat2.name}')
        ax.plot(Up_mat1_intersect, P_mat1_intersect, 'ko', label=f'Conditions in {mat1.name}')
        fig.legend()
        fig.tight_layout()
        plt.show()

    def IM_mat1_inf_mat2_measUp1(self, mat1, mat2, vecUp, Up_1, x_guess=5, plot=True):
        
        # Hugoniot pressures
        mat1_P = mat1.rho0*vecUp*mat1.Us_from_Up(vecUp)
        mat2_P = mat2.rho0*vecUp*mat2.Us_from_Up(vecUp)

        # mat1 reversed Hugoniot with symmetry plane at Up_1
        mat1_P_r = np.flip(mat1_P)

        # Interpolation to find intersection
        P1 = interpolate.interp1d(vecUp, mat1_P, kind='cubic', fill_value='extrapolate')
        P1r = interpolate.interp1d(vecUp, mat1_P_r, kind='cubic', fill_value='extrapolate')
        P2 = interpolate.interp1d(vecUp, mat2_P, kind='cubic', fill_value='extrapolate')

        P1_measured = P1(vecUp)[np.where(vecUp==Up_1)][0]

        func_P1rP1meas_intersect = lambda x : P1r(x) - P1_measured
        Up_temp = fsolve(func_P1rP1meas_intersect, x_guess)[0]

        new_vecUp = vecUp + Up_1 - Up_temp
        P1r = interpolate.interp1d(new_vecUp, mat1_P_r, kind='cubic', fill_value='extrapolate')
        
        func_P1rP2_intersect = lambda x : P1r(x) - P2(x)
        Up_mat2_intersect = fsolve(func_P1rP2_intersect, x_guess)[0]
        P_mat2_intersect = P1r(Up_mat2_intersect)

        if plot == True:

            print(f'*** Results of impedance matching ***\n')
            print(f'Conditions measured in {mat1.name}:')
            print(f'\tUp measured in {mat1.name} = {Up_1:.3f} km/s')
            print(f'\tP in {mat1.name} = {P1_measured:.1f} GPa\n')
            print(f'Conditions measured in {mat2.name}:')
            print(f'\tUp in {mat2.name} = {Up_mat2_intersect:.3f} km/s')
            print(f'\tP in {mat2.name} = {P_mat2_intersect:.1f} GPa')
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('Particle velocity (km/s)')
            ax.set_ylabel('Pressure (GPa)')
            ax.plot(vecUp, mat1_P, 'k-', label=f'Hug. {mat1.name}')
            ax.plot(vecUp, mat2_P, 'r-', label=f'Hug. {mat2.name}')
            ax.plot(new_vecUp, P1r(new_vecUp), 'k--', label=f'Rev. Hug. {mat1.name}')
            ax.plot(Up_1, P1_measured, 'ko', label=f'Conditions measured in {mat1.name}')
            ax.plot(Up_mat2_intersect, P_mat2_intersect, 'ro', label=f'Conditions in {mat2.name}')
            fig.legend()
            fig.tight_layout()
            plt.show()

            return [P_mat2_intersect, Up_mat2_intersect]
        
        else:
            return [P_mat2_intersect, Up_mat2_intersect]