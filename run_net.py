import numpy as np
from generator import myGenerator
import matplotlib.pyplot as plt
import os
import shutil
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from parameters import demo
import scipy.io as sio


train_epochs = 100
save_interval = 20
batch_size = 20
lr = 1e-3

reg_xy_diff_ratio = 5
reg_z_diff_ratio = 0.2
reg_res_ratio = 5

Nx = demo.Nx
Ny = demo.Ny
Nz = demo.Nz


class train_generator(object):
    def __init__(self, row, col, layer_num, batch_size, channel=1):
        self.img_rows = row
        self.img_cols = col
        self.layer_num = layer_num
        self.channel = channel
        self.lamda = demo.lamda
        self.dphi = demo.dphi
        self.n = np.zeros([row, col, layer_num])

        self.sample = 6
        self.batch_size = batch_size

        self.output_train, self.input_train, self.thetas, self.input_test, self.input_test_angle, self.output_test = self.load_dataset()

        self.myGen = myGenerator(row=row, col=col, layer_num=layer_num, channel=channel, batch_size=batch_size,
                                 reg_xy_diff_ratio=reg_xy_diff_ratio, reg_z_diff_ratio=reg_z_diff_ratio, reg_res_ratio=reg_res_ratio, lr=lr)
        self.generator_model = self.myGen.generator_model()
        self.generator = self.myGen.generator()

    def load_dataset(self):
        input_state = np.load('./input.npy').astype(np.complex64)
        output = np.load('./unwrap.npy')
        thetas = np.load('./thetas.npy')
        self.data_num = np.size(thetas, 0)
        thetas = np.reshape(thetas, [self.data_num, 2, 1])
        """
        if self.channel != 1:
            input_state = input_state.reshape(-1, self.img_rows, self.img_cols, self.channel)
            output = output.reshape(-1, self.img_rows, self.img_cols, self.channel)
        else:
            input_state = input_state.reshape(-1, self.img_rows, self.img_cols)
            output = output.reshape(-1, self.img_rows, self.img_cols)
        """
        index = np.linspace(0, self.data_num-1, self.sample).astype(np.int)
        test_input = input_state[index, :, :]
        test_input_angle = thetas[index, :, :]
        test_output = output[index, :, :]
        return output, input_state, thetas, test_input, test_input_angle, test_output

    def get_n(self):
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

    def train(self, train_steps=2000, save_interval=0):
        if os.path.exists('./train_image') == True:
            shutil.rmtree('./train_image')

        os.mkdir('./train_image')
        os.mkdir('./train_image/unwrap')
        os.mkdir('./train_image/projection')

        for i in range(train_steps):
            index = np.random.randint(0, self.data_num, size=self.batch_size)
            input_train = self.input_train[index, :, :]
            anglex = self.thetas[index, 0, :].reshape(self.batch_size, 1, 1)
            angley = self.thetas[index, 1, :].reshape(self.batch_size, 1, 1)
            output_train = self.output_train[index, :, :]
            g_loss = self.generator_model.train_on_batch([input_train, anglex, angley], output_train)
            log_mesg = "[%d G loss: %f]" % (i, g_loss)
            print(log_mesg)

            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(samples=self.input_test.shape[0], input_test=self.input_test,
                                        input_test_angle=self.input_test_angle, output_truth=self.output_test, step=(i+1))

        self.generator.save_weights('generator_weight.h5')
        if os.path.exists('./solve_image') == True:
            shutil.rmtree('./solve_image')

        os.mkdir('./solve_image')
        os.mkdir('./solve_image/2d')

        self.draw_n()
        self.get_n()
        np.save('./solve_image/delta_n.npy', self.n)

        data_save = {'n_GNN': self.n}
        sio.savemat('./solve_image/n_GNN.mat', data_save)


    def plot_images(self, samples=4, input_test=None, input_test_angle=None, output_truth=None, step=0):
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
            im = plt.imshow(img)
            plt.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig("train_image/unwrap/predict_%d.png" % step)
        plt.close('all')

        self.draw_n_projection(path='./train_image/projection/', num=step)


    def draw_n_projection(self, path, num=0):
        self.get_n()
        plt.figure(figsize=[10, 10])
        plt.title('Reconstraction Bead')
        plt.subplot(1, 3, 1)
        ax = plt.gca()
        im = plt.imshow(np.max(self.n, axis=2))
        plt.axis('off')
        plt.title('x  y')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.subplot(1, 3, 2)
        ax = plt.gca()
        im = plt.imshow(np.max(self.n, axis=1))
        plt.axis('off')
        plt.title('x  z')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.subplot(1, 3, 3)
        ax = plt.gca()
        im = plt.imshow(np.max(self.n, axis=0))
        plt.axis('off')
        plt.title('y  z')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.savefig(path + "projection_max_%d.png" % num)
        plt.close('all')


        plt.figure(figsize=[10, 10])
        plt.title('Reconstraction Bead')
        plt.subplot(1, 3, 1)
        ax = plt.gca()
        im = plt.imshow(np.mean(self.n, axis=2))
        plt.axis('off')
        plt.title('x  y')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.subplot(1, 3, 2)
        ax = plt.gca()
        im = plt.imshow(np.mean(self.n, axis=1))
        plt.axis('off')
        plt.title('x  z')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.subplot(1, 3, 3)
        ax = plt.gca()
        im = plt.imshow(np.mean(self.n, axis=0))
        plt.axis('off')
        plt.title('y  z')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.savefig(path + "projection_mean_%d.png" % num)
        plt.close('all')



    def draw_n(self):
        self.get_n()
        print 'Drawing 2D ...'
        for i in range(Nz):
            im = plt.imshow(self.n[:, :, i])
            ax = plt.gca()
            plt.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.savefig('./solve_image/2d/z_%d.png' % i)
            plt.axis('on')
            plt.clf()
            plt.close()

        for i in range(Ny):
            im = plt.imshow(self.n[:, i, :])
            ax = plt.gca()
            plt.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.savefig('./solve_image/2d/y_%d.png' % i)
            plt.axis('on')
            plt.clf()
            plt.close()

        for i in range(Nx):
            im = plt.imshow(self.n[i, :, :])
            ax = plt.gca()
            plt.axis('off')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.savefig('./solve_image/2d/x_%d.png' % i)
            plt.axis('on')
            plt.clf()
            plt.close()
        """
        print 'Drawing 3D ...'
        X = np.arange(0, Nx, 1)
        Y = np.arange(0, Ny, 1)
        X, Y = np.meshgrid(X, Y)
        for i in range(Nz):
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(X, Y, self.n[:, :, i], rstride=1, cstride=1, cmap='rainbow')
            plt.savefig('./solve_image/3d/%d.png' % i)
            plt.clf()
            plt.close()
        """
        self.draw_n_projection(path='./solve_image/')
        print 'Finished drawing!'



if __name__ == '__main__':
    my_net = train_generator(row=Nx, col=Ny, layer_num=Nz, batch_size=batch_size)
    my_net.train(train_steps=train_epochs, save_interval=save_interval)
