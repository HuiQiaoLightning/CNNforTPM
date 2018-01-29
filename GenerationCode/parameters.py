import numpy as np

class parameters(object):
	# Set the parameters for simulation and reconstruction CNN
    def __init__(self, Nx, Ny, Nz):
        self.lamda = 561e-9
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.n0 = 1.518
        self.magnification = 100.0 * 200 / 180
        self.camera_pixal_size = 8e-6
        self.NA = 1.4
        self.effective_NA = 1.33

        self.k0 = 2 * np.pi / self.lamda
        self.k = self.k0 * self.n0

        # discretizetion step along axis
        self.dx = 2 * self.camera_pixal_size / self.magnification
        self.dy = 2 * self.camera_pixal_size / self.magnification
        self.dz = 2 * self.camera_pixal_size / self.magnification

        # length of computational window along axis
        self.Lx = self.dx * self.Nx
        self.Ly = self.dy * self.Ny
        self.Lz = self.dz * self.Nz

        # computational grid along axis
        self.xx = self.dx * np.arange(-self.Nx / 2 + 1, self.Nx / 2 + 1)
        self.yy = self.dy * np.arange(-self.Ny / 2 + 1, self.Ny / 2 + 1)
        self.zz = self.dz * np.arange(1, self.Nz + 1)

        # frequency discretization step along axis
        self.dkx = 2 * np.pi / self.Lx
        self.dky = 2 * np.pi / self.Ly
        self.dkz = 2 * np.pi / self.Lz

        # spatial frequency vectors
        self.kx = self.dkx * np.hstack((np.arange(0, self.Nx / 2 + 1), np.arange(-self.Nx / 2 + 1, 0)))
        self.ky = self.dky * np.hstack((np.arange(0, self.Ny / 2 + 1), np.arange(-self.Ny / 2 + 1, 0)))

        # phase factor for Fresnel diffrantion
        [Kx, Ky] = np.meshgrid(self.kx, self.ky)
        K2 = Kx ** 2 + Ky ** 2
        temp = (self.k ** 2 - K2).astype(np.complex)
        self.dphi = np.real(K2 / (self.k + np.sqrt(temp)))

        self.spectral_filter_bandwidth = self.k0 * self.effective_NA
        self.spectral_filter = np.exp(-((Kx ** 2 + Ky ** 2) / self.spectral_filter_bandwidth ** 2) ** 40)

        [X, Y] = np.meshgrid(self.xx, self.yy)
        self.prop_window = np.exp(-((X ** 2 + Y ** 2) / (0.8 * self.Lx / 2) ** 2) ** 10)


demo_para = parameters(160, 160, 160)