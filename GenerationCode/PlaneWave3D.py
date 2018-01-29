import numpy as np
import numpy.fft as F
from parameters import demo_para
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class PlaneWave3D(object):
    def __init__(self):

        #constant parameters
        self.lamda = demo_para.lamda
        self.magnification = demo_para.magnification
        self.n0 = demo_para.n0
        self.k0 = 2 * np.pi / self.lamda
        self.k = self.n0 * self.k0

        #computational grid(size of initial_focused_field)
        self.Nx = demo_para.Nx
        self.Ny = demo_para.Ny
        self.Nz = demo_para.Nz

        #physical dimensions
        self.Lx = demo_para.Lx
        self.Ly = demo_para.Ly
        self.Lz = demo_para.Lz

        # discretization steps = (lamda/NA)*(N_disk/N_detection_window)
        self.drx = demo_para.dx
        self.dry = demo_para.dy
        self.drz = demo_para.dz

        #domain coordinate vectors
        self.rx = demo_para.xx
        self.ry = demo_para.yy
        self.rz = demo_para.zz

        #2d meshgrid
        [self.Rxx, self.Ryy] = np.meshgrid(self.rx, self.ry)

        #discretization in spatial frequency domain: dkx = 2*pi/Lx
        self.dkx = demo_para.dkx
        self.dky = demo_para.dky
        self.dkz = demo_para.dkz

        #spatial frequency vectors
        self.kx = self.dkx * np.hstack((np.arange(0, self.Nx / 2), np.arange(-self.Nx / 2, 0)))
        self.ky = self.dky * np.hstack((np.arange(0, self.Ny / 2), np.arange(-self.Ny / 2, 0)))

        #phase factor for Fresnel diffrantion
        self.dphi = demo_para.dphi

        self.prop_window = demo_para.prop_window


    def backward(self, uin, L):
	# back propagate uin for L distance
        N = int(L / self.drz)
        imag_log_ur = np.zeros([self.Nx, self.Ny])
        u = uin
        for ind_n in range(N):
            u_before = u
            u = F.ifft2(F.fft2(u) * np.exp(-1j * (-self.drz) * self.dphi))
            imag_log_ur += np.imag(np.log((u + 1e-8) / (u_before + 1e-8)))

        return u, imag_log_ur

    def forward(self, x, g_in, anglex, angley, ind_data):
		# forward model for beam propagation model
        imag_log_ur = np.zeros([self.Nx, self.Ny])
        uin = g_in
        [u, delta_imag_log_ur] = self.backward(g_in, self.Lz / 2)


        imag_log_ur += delta_imag_log_ur
        gtotal = np.zeros([self.Nz, self.Nx, self.Ny], dtype=complex)

        cos_theta = 1 / np.sqrt(np.tan(anglex) ** 2 + np.tan(angley) ** 2 + 1)

        for ind_z in range(self.Nz):
            u_before = u
            u = F.ifft2(F.fft2(u) * np.exp(-1j * self.drz * self.dphi))   #diffraction step
            u = u * np.exp(1j * self.k0 * self.drz / cos_theta * x[:, :, ind_z])  #refraction step
            gtotal[ind_z, :, :] = u
            imag_log_ur += np.imag(np.log((u + 1e-8) / (u_before + 1e-8)))

        #Backpropagate to the center
        [u, delta_imag_log_ur] = self.backward(u, self.Lz / 2)
        imag_log_ur += delta_imag_log_ur
        imag_log_ur *= self.prop_window
        
        uob = u

        return uin, uob, imag_log_ur
