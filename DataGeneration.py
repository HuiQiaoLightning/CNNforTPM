import numpy as np
from GenerationCode.PlaneWave3D import PlaneWave3D
import multiprocessing
import scipy.io as sio
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


D_bead = 5e-6      #Diameter for simulation bead
dn = 0.03          #Delta refraction index of bead
data_num = 81      #Varible for data number simulation. Default is 81
noise_scale = 0    #Varible for noise simulation. If noise_scale=0, the simulation result has no noise.


class DemoPlaneWave(object):
    def __init__(self):
        # forward model
        self.forward_model = PlaneWave3D()

        # fundamental parameters
        self.lamda = self.forward_model.lamda  # free space wavelength
        self.magnification = self.forward_model.magnification # Mnominal*Fused/Fnominal
        self.n0 = self.forward_model.n0  # background refractive index
        self.k0 = 2 * np.pi / self.lamda
        self.k = self.n0 * self.k0

        # numbers of pixels along axis
        self.Nx = self.forward_model.Nx
        self.Ny = self.forward_model.Ny
        self.Nz = self.forward_model.Nz

        # discretizetion step along axis
        self.dx = self.forward_model.drx
        self.dy = self.forward_model.dry
        self.dz = self.forward_model.drz

        # length of computational window along axis
        self.Lx = self.forward_model.Lx
        self.Ly = self.forward_model.Ly
        self.Lz = self.forward_model.Lz

        #frequency discretization step along axis
        self.dkx = self.forward_model.dkx
        self.dky = self.forward_model.dky
        self.dkz = self.forward_model.dkz

        # frequency grid along axis
        self.kx = self.forward_model.kx
        self.ky = self.forward_model.ky

        # computational grid along axis
        self.rx = self.forward_model.rx
        self.ry = self.forward_model.ry
        self.rz = self.forward_model.rz

        [self.Rxx, self.Ryy] = np.meshgrid(self.rx, self.ry)

        [X, Y, Z] = np.meshgrid(self.rx, self.ry, self.rz)

        # location of three bead
        self.x1_center = -2.5e-6
        self.y1_center = 0
        self.z1_center = 8.5e-6

        self.x2_center = -2.5e-6
        self.y2_center = 0
        self.z2_center = 14.5e-6

        self.x3_center = 2.5e-6
        self.y3_center = 0
        self.z3_center = 11.5e-6

        R1 = np.sqrt((X - self.x1_center) ** 2 + (Y - self.y1_center) ** 2 + (Z - self.z1_center) ** 2)
        R2 = np.sqrt((X - self.x2_center) ** 2 + (Y - self.y2_center) ** 2 + (Z - self.z2_center) ** 2)
        R3 = np.sqrt((X - self.x3_center) ** 2 + (Y - self.y3_center) ** 2 + (Z - self.z3_center) ** 2)
        R = np.minimum(np.minimum(R1, R2), R3)

        self.phantom = np.zeros([self.Nx, self.Ny, self.Nz])
        index_bead = np.where(R < D_bead / 2)
        self.phantom[index_bead] = 1
        self.phantom = dn * self.phantom

        # phase factor for Fresnel diffrantion
        self.dphi = self.forward_model.dphi
        self.prop_window = self.forward_model.prop_window


demo = DemoPlaneWave()


class ScanPlaneWave(object):
    def __init__(self, demo):
        self.DemoPW = demo
        self.data_num = data_num
        thetas = np.zeros([self.data_num, 2])
        thetas[:, 0] = np.linspace(-45 * np.pi / 180, 45 * np.pi / 180, self.data_num)		# Scan the sample in one dimension
        self.thetas = thetas

    def angle_scan(self, pid, start_num, end_num):
		# Calculate the input and output field of the sample from different angle using multiprocess
        gobs = np.zeros([end_num-start_num, self.DemoPW.Nx, self.DemoPW.Ny], dtype=complex) # Measurements of output complex field
        gins = np.zeros([end_num-start_num, self.DemoPW.Nx, self.DemoPW.Ny], dtype=complex) # Measurements of input complex field
        imag_log_urs = np.zeros([end_num - start_num, self.DemoPW.Nx, self.DemoPW.Ny])		# Unwrapped phase for output field

        for ind_data in range(start_num, end_num, 1):
            angx = self.thetas[ind_data, 0]
            angy = self.thetas[ind_data, 1]

            Bx = 0.25 * self.DemoPW.Lx / np.cos(angx)  # Beam scale
            By = 0.25 * self.DemoPW.Ly / np.cos(angy)

            ain = np.exp(-((self.DemoPW.Rxx / Bx) ** 2 + (self.DemoPW.Ryy / By) ** 2))  # Gauss illumination beam amplitude
            pin = self.DemoPW.k * (np.sin(angx) * self.DemoPW.Rxx + np.sin(angy) * self.DemoPW.Ryy)

            gin = ain * np.exp(1j * pin)  # Define a plane wave at the origin

            [uin, uob, imag_log_ur] = self.DemoPW.forward_model.forward(self.DemoPW.phantom, gin, angx, angy, ind_data)

            gins[ind_data - start_num, :, :] = uin
            gobs[ind_data - start_num, :, :] = uob 
            imag_log_urs[ind_data - start_num, :, :] = imag_log_ur
            print 'Process %d completed %d of %d' %(pid, ind_data + 1 - start_num, end_num - start_num)
        return gins, gobs, imag_log_urs


    def draw(self, data, start_num, filename, axis=0):
		# Draw 3D data along specified axis using multiprocess
        data_length = np.size(data, axis=axis)
        if data.dtype == complex:
            data_mag = np.abs(data)
            data_phase = np.angle(data)
            for i in range(data_length):
                ax = plt.gca()
                if axis == 2:
                    im = plt.imshow(data_mag[:, :, i], cmap='jet')
                elif axis == 1:
                    im = plt.imshow(data_mag[:, i, :], cmap='jet')
                else:
                    im = plt.imshow(data_mag[i, :, :], cmap='jet')
                plt.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.savefig('./' + filename + '/mag' + '/%d.png' % (i+start_num))
                plt.clf()
                plt.close()
                ax = plt.gca()
                if axis == 2:
                    im = plt.imshow(data_phase[:, :, i], cmap='jet')
                elif axis == 1:
                    im = plt.imshow(data_phase[:, i, :], cmap='jet')
                else:
                    im = plt.imshow(data_phase[i, :, :], cmap='jet')
                plt.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.savefig('./' + filename + '/phase' + '/%d.png' % (i+start_num))
                plt.clf()
                plt.close()
        else:
            for i in range(data_length):
                ax = plt.gca()
                if axis == 2:
                    im = plt.imshow(data[:, :, i], cmap='jet')
                elif axis == 1:
                    im = plt.imshow(data[:, i, :], cmap='jet')
                else:
                    im = plt.imshow(data[i, :, :], cmap='jet')
                plt.axis('off')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.savefig('./'+filename+'/%d.png' % (i+start_num))
                plt.clf()
                plt.close()


def propagate(demo, pid, start_num, end_num):
    return demo.angle_scan(pid, start_num, end_num)


def draw(demo, data, start_num, filename, axis=0):
    demo.draw(data, start_num, filename, axis)


if __name__ == '__main__':
	# Save generated file in different folders for noise free or noise case
    if noise_scale == 0:
        if os.path.exists('./DataGenerated/NoiseFree') == True:
            shutil.rmtree('./DataGenerated/NoiseFree')
        os.mkdir('./DataGenerated/NoiseFree')
        os.mkdir('./DataGenerated/NoiseFree/object')
        os.mkdir('./DataGenerated/NoiseFree/simul_input')
        os.mkdir('./DataGenerated/NoiseFree/simul_input/mag')
        os.mkdir('./DataGenerated/NoiseFree/simul_input/phase')
        os.mkdir('./DataGenerated/NoiseFree/simul_unwrap')

    else:
        if os.path.exists('./DataGenerated/Noise') == True:
            shutil.rmtree('./DataGenerated/Noise')
        os.mkdir('./DataGenerated/Noise')
        os.mkdir('./DataGenerated/Noise/object')
        os.mkdir('./DataGenerated/Noise/simul_input')
        os.mkdir('./DataGenerated/Noise/simul_input/mag')
        os.mkdir('./DataGenerated/Noise/simul_input/phase')
        os.mkdir('./DataGenerated/Noise/simul_unwrap')

    scan = ScanPlaneWave(demo)

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()

    data_zone = np.linspace(0, scan.data_num, cpus + 1).astype(np.int)

    output_data = np.zeros([scan.data_num, demo.Nx, demo.Ny], dtype=complex)
    input_data = np.zeros([scan.data_num, demo.Nx, demo.Ny], dtype=complex)
    unwrap = np.zeros([scan.data_num, demo.Nx, demo.Ny])

    results = []
    for i in xrange(0, cpus):
        result = pool.apply_async(propagate, args=(scan, i, data_zone[i], data_zone[i+1]))
        results.append(result)
    pool.close()
    pool.join()
    num = 0
    for result in results:
        temp_in, temp_out, temp_unwrap = result.get()
        delta = np.size(temp_out, 0)
        output_data[num: num + delta, :, :] = temp_out
        input_data[num: num + delta, :, :] = temp_in
        unwrap[num: num + delta, :, :] = temp_unwrap
        num += delta

    if noise_scale == 0:
        np.save('./DataGenerated/NoiseFree/simul_input.npy', input_data)
        np.save('./DataGenerated/NoiseFree/simul_unwrap.npy', unwrap)
        np.save('./DataGenerated/NoiseFree/simul_output.npy', output_data)
        np.save('./DataGenerated/NoiseFree/thetas.npy', scan.thetas)

        print 'Drawing input ...'
        multiprocessing.freeze_support()
        pool = multiprocessing.Pool()
        results = []
        #scan.draw(data=input_data, start_num=0, filename='DataGenerated/NoiseFree/simul_input')
        for i in xrange(0, cpus):
            result = pool.apply_async(draw, args=(scan, input_data[data_zone[i]:data_zone[i + 1], :, :], data_zone[i], 'DataGenerated/NoiseFree/simul_input'))
            results.append(result)
        print 'Drawing unwrap ...'
        #scan.draw(data=unwrap, start_num=0, filename='DataGenerated/NoiseFree/simul_unwrap')
        for i in xrange(0, cpus):
            result = pool.apply_async(draw, args=(scan, unwrap[data_zone[i]:data_zone[i + 1], :, :], data_zone[i], 'DataGenerated/NoiseFree/simul_unwrap'))
            results.append(result)
        pool.close()
        pool.join()
        print 'Finished drawing'
        print 'Drawing item ...'
        scan.draw(data=scan.DemoPW.phantom, start_num=0, filename='DataGenerated/NoiseFree/object', axis=2)
        ground_truth = {'ground_truth': scan.DemoPW.phantom}
        sio.savemat('./DataGenerated/NoiseFree/ground_truth.mat', ground_truth)

    else:
        noisy_unwrap = unwrap + np.random.normal(scale=noise_scale, size=[scan.data_num, demo.Nx, demo.Ny])
        input_snr = 10 * np.log10(np.linalg.norm(unwrap ** 2) / np.linalg.norm((noisy_unwrap - unwrap) ** 2))
        print 'Input SNR: %f dB' % input_snr
        np.save('./DataGenerated/Noise/simul_input.npy', input_data)
        np.save('./DataGenerated/Noise/simul_unwrap.npy', noisy_unwrap)
        np.save('./DataGenerated/Noise/simul_output.npy', output_data)
        np.save('./DataGenerated/Noise/thetas.npy', scan.thetas)

        print 'Drawing input ...'
        multiprocessing.freeze_support()
        pool = multiprocessing.Pool()
        results = []
        for i in xrange(0, cpus):
            result = pool.apply_async(draw, args=(
                scan, input_data[data_zone[i]:data_zone[i + 1], :, :], data_zone[i],
                'DataGenerated/Noise/simul_input'))
            results.append(result)
        print 'Drawing unwrap ...'
        for i in xrange(0, cpus):
            result = pool.apply_async(draw, args=(
                scan, noisy_unwrap[data_zone[i]:data_zone[i + 1], :, :], data_zone[i],
                'DataGenerated/Noise/simul_unwrap'))
            results.append(result)
        pool.close()
        pool.join()
        print 'Finished drawing'
        print 'Drawing item ...'
        scan.draw(data=scan.DemoPW.phantom, start_num=0, filename='DataGenerated/Noise/object', axis=2)
        ground_truth = {'ground_truth': scan.DemoPW.phantom}
        sio.savemat('./DataGenerated/Noise/ground_truth.mat', ground_truth)