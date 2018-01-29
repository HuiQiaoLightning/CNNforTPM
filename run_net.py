import numpy as np
from CNNCode.generator import myGenerator
import matplotlib.pyplot as plt
import os
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
from GenerationCode.parameters import demo_para
from DataGeneration import demo
import scipy.io as sio


train_epochs = 500		#Training epochs for training iterations
save_interval = 50		#Saving training result for every save_interval
batch_size = 20			#Training batch size

#Regularizer coefficient
reg_xy_diff_ratio = 3
reg_z_diff_ratio = 1.5

#Voxel size for reconstruction volume
Nx = demo_para.Nx
Ny = demo_para.Ny
Nz = demo_para.Nz


class train_generator(object):
    def __init__(self, row, col, layer_num, batch_size, channel=1):
        self.img_rows = row
        self.img_cols = col
        self.layer_num = layer_num
        self.channel = channel
        self.lamda = demo_para.lamda
        self.dphi = demo_para.dphi
        self.n = np.zeros([row, col, layer_num])

        self.sample = 6
        self.batch_size = batch_size
        self.phantom = demo.phantom

        self.SNRs = np.zeros([train_epochs])
        self.NRMSEs = np.zeros([train_epochs])
        self.Loss = np.zeros([train_epochs])

        self.output_train, self.input_train, self.thetas, self.input_test, self.input_test_angle, self.output_test = self.load_dataset()

        self.myGen = myGenerator(row=row, col=col, layer_num=layer_num, channel=channel, batch_size=batch_size, 
								 reg_xy_diff_ratio=reg_xy_diff_ratio, reg_z_diff_ratio=reg_z_diff_ratio)
								 
        self.generator_model = self.myGen.generator_model()
        self.generator = self.myGen.generator()

    def load_dataset(self):
        # load noise free data
        input_state = np.load('./DataGenerated/NoiseFree/simul_input.npy').astype(np.complex64)
        output = np.load('./DataGenerated/NoiseFree/simul_unwrap.npy')
        thetas = np.load('./DataGenerated/NoiseFree/thetas.npy')

        # load noise data
        """
        input_state = np.load('./DataGenerated/Noise/simul_input.npy').astype(np.complex64)
        output = np.load('./DataGenerated/Noise/simul_unwrap.npy')
        thetas = np.load('./DataGenerated/Noise/thetas.npy')
        """
        self.data_num = np.size(thetas, 0)
        thetas = np.reshape(thetas, [self.data_num, 2, 1])
        index = np.linspace(0, self.data_num-1, self.sample).astype(np.int)
        test_input = input_state[index, :, :]
        test_input_angle = thetas[index, :, :]
        test_output = output[index, :, :]
        return output, input_state, thetas, test_input, test_input_angle, test_output

    def get_n(self):
		# Get refraction index from the training weights
        n = np.zeros(shape=[Nx, Ny, Nz])
        total_layers = np.size(self.generator.layers)
        k = 0
        for i in range(total_layers):
            layer_name = self.generator.layers[i].name
            if layer_name.find('Refraction') != -1:
                temp_weights = self.generator.layers[i].get_weights()
                n[:, :, k] = temp_weights[0]
                k += 1
            if k == Nz:
                break
        self.n = n

    def get_SNR(self):
		# Calculate SNR for training weights
        self.get_n()
        groundTruth = self.phantom
        SNR = 20 * np.log10(np.linalg.norm(groundTruth) / np.linalg.norm(groundTruth - self.n))
        NRMSE = np.sum(np.square(self.n - self.phantom)) / np.sum(np.square(self.phantom))
        return SNR, NRMSE

    def train(self, train_steps=2000, save_interval=0):
		# Training function
        if os.path.exists('./CNNTrainResult/training_image') == True:
            shutil.rmtree('./CNNTrainResult/training_image')

        os.mkdir('./CNNTrainResult/training_image')
        os.mkdir('./CNNTrainResult/training_image/unwrap')
        os.mkdir('./CNNTrainResult/training_image/projection')

        for i in range(train_steps):
            index = np.random.randint(0, self.data_num, size=self.batch_size)
            input_train = self.input_train[index, :, :]
            anglex = self.thetas[index, 0, :].reshape(self.batch_size, 1, 1)
            angley = self.thetas[index, 1, :].reshape(self.batch_size, 1, 1)
            output_train = self.output_train[index, :, :]
            g_loss = self.generator_model.train_on_batch([input_train, anglex, angley], output_train)
            self.Loss[i] = g_loss
            self.SNRs[i], self.NRMSEs[i] = self.get_SNR()
            log_mesg = "[%d  G loss: %f  SNR: %f  NRMSE: %f]" % (i, g_loss, self.SNRs[i], self.NRMSEs[i])
            print(log_mesg)

            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.test_images(samples=self.input_test.shape[0], input_test=self.input_test,
                                        input_test_angle=self.input_test_angle, output_truth=self.output_test, step=(i+1))
		
		# Save the training result
        #self.generator.save_weights('generator_weight.h5')
        if os.path.exists('./CNNTrainResult/solve_image') == True:
            shutil.rmtree('./CNNTrainResult/solve_image')

        os.mkdir('./CNNTrainResult/solve_image')
        os.mkdir('./CNNTrainResult/solve_image/2d')

        self.draw_n()
        self.get_n()
        np.save('./CNNTrainResult/solve_image/delta_n.npy', self.n)

        self.draw_SNR_NRMSE()
        data_save = {'X_sol_GNN': self.n, 'SNRs': self.SNRs, 'NRMSEs': self.NRMSEs, 'Loss': self.Loss}
        sio.savemat('./CNNTrainResult/solve_image/recon_data_GNN.mat', data_save)

		
    def test_images(self, samples=4, input_test=None, input_test_angle=None, output_truth=None, step=0):
		# Show the test sample during training 
        img_col = 4
        img_row = 2 * samples / img_col
        anglex = input_test_angle[:, 0, :].reshape(samples, 1, 1)
        angley = input_test_angle[:, 1, :].reshape(samples, 1, 1)
        images = self.generator.predict([input_test, anglex, angley])
        plt.figure(figsize=(10, 10))
        for i in range(2 * samples):
            mod_number = i % img_col
            plt.subplot(img_row, img_col, i + 1)
            if mod_number == 0:
                img = output_truth[i / 2, :, :]
            elif mod_number == 1:
                img = images[i / 2, :, :]
            elif mod_number == 2:
                img = output_truth[i / 2, :, :]
            else:
                img = images[i / 2, :, :]
            ax = plt.gca()
            img = np.reshape(img, [self.img_rows, self.img_cols])
            im = plt.imshow(img, cmap='jet')
            plt.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig("CNNTrainResult/training_image/unwrap/predict_%d.png" % step)
        plt.close('all')

        self.draw_n_projection(path='./CNNTrainResult/training_image/projection/', num=step)


    def draw_n_projection(self, path, num=0):
		# Draw projection of training weights
        self.get_n()
        plt.figure(figsize=[10, 10])
        plt.title('Reconstraction Bead')
        plt.subplot(1, 3, 1)
        ax = plt.gca()
        im = plt.imshow(np.max(self.n, axis=2), cmap='jet')
        plt.axis('off')
        plt.title('x  y')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.subplot(1, 3, 2)
        ax = plt.gca()
        im = plt.imshow(np.max(self.n, axis=1), cmap='jet')
        plt.axis('off')
        plt.title('x  z')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.subplot(1, 3, 3)
        ax = plt.gca()
        im = plt.imshow(np.max(self.n, axis=0), cmap='jet')
        plt.axis('off')
        plt.title('y  z')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.savefig(path + "projection_max_%d.png" % num)
        plt.close('all')


    def draw_SNR_NRMSE(self):
		# Draw SNR and NRMSE curve
        plt.figure(figsize=[10, 10])
        plt.plot(np.arange(0, train_epochs), self.SNRs)
        plt.title('SNRs in Reconstruction')
        plt.xlabel('iter times')
        plt.ylabel('SNR(dB)')
        plt.savefig('./CNNTrainResult/solve_image/SNRs.png')
        plt.close('all')
        np.save('./CNNTrainResult/solve_image/SNRs.npy', self.SNRs)

        plt.figure(figsize=[10, 10])
        plt.plot(np.arange(0, train_epochs), self.NRMSEs)
        plt.title('NRMSEs in Reconstruction')
        plt.xlabel('iter times')
        plt.ylabel('NRMSE')
        plt.savefig('./CNNTrainResult/solve_image/NRMSEs.png')
        plt.close('all')
        np.save('./CNNTrainResult/solve_image/NRMSEs.npy', self.NRMSEs)

        plt.figure(figsize=[10, 10])
        plt.plot(np.arange(0, train_epochs), self.Loss)
        plt.title('Loss in Reconstruction')
        plt.xlabel('iter times')
        plt.ylabel('Loss')
        plt.savefig('./CNNTrainResult/solve_image/Loss.png')
        plt.close('all')
        np.save('./CNNTrainResult/solve_image/Loss.npy', self.Loss)


    def draw_n(self):
		# Draw reconstructed refraction index layer by layer
        self.get_n()
        print 'Drawing 2D ...'
        for i in range(Nz):
            im = plt.imshow(self.n[:, :, i], cmap='jet')
            ax = plt.gca()
            plt.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.savefig('./CNNTrainResult/solve_image/2d/%d.png' % i)
            plt.axis('on')
            plt.clf()
            plt.close()
        self.draw_n_projection(path='./CNNTrainResult/solve_image/')
        print 'Finished drawing!'


if __name__ == '__main__':
    my_net = train_generator(row=Nx, col=Ny, layer_num=Nz, batch_size=batch_size)
    my_net.train(train_steps=train_epochs, save_interval=save_interval)