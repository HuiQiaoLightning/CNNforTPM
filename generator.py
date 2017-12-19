from keras import backend as K
from keras import initializers
from keras.layers import Dense, Flatten, Reshape, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, SGD
from myLayers import DiffractionLayer, RefractionLayer, FrequencyShiftLayer, ImagLogLayer, AddLayer, PropWindowLayer
import numpy as np
from parameters import demo
from keras.utils.vis_utils import plot_model
import scipy.io as sio


class myGenerator(object):
    def __init__(self, row=256, col=256, channel=1, layer_num=128, batch_size=100, loss_ratio=1,
                    reg_xy_diff_ratio=5, reg_z_diff_ratio=0.2, reg_res_ratio=5, lr=1e-3):
        self.loss_ratio = loss_ratio
        self.reg_xy_diff_ratio = reg_xy_diff_ratio
        self.reg_z_diff_ratio = reg_z_diff_ratio
        self.reg_res_ratio = reg_res_ratio
	self.lr = lr
        self.row = row
        self.col = col
        self.batch_size = batch_size
        self.Lz = demo.Lz
        self.filter = demo.spectral_filter
        self.channel = channel
        self.last_weight = None
        self.drz = demo.dz
        self.dphi = demo.dphi
        self.k0 = demo.k0
        self.layer_num = layer_num

        self.prop_window = demo.prop_window

        self.initial_n = self.get_initial_n()

        self.G = None   # generator
        self.GM = None  # adversarial model


    def my_loss(self, y_true, y_pred):
        loss = self.loss_ratio * K.sum(K.tf.abs(y_true - y_pred)) / self.batch_size
        #loss = self.loss_ratio * K.sum(K.abs(K.tf.imag(K.tf.log(y_pred / y_true))))
        return loss

    def get_initial_n(self):
        dict = sio.loadmat('/home/bbnc/pycharm/PycharmProjects/data/DataKamilov/n.mat')
        phantom = dict['n']
        phantom = phantom - 1.33
        return phantom

    def my_reg(self, weight):
        if self.last_weight == None:
            self.last_weight = weight
            z_diff = K.sum(K.zeros([self.row, self.col]))
        else:
            diff_weight = weight - self.last_weight
            self.last_weight = weight
            z_diff = K.sum(K.abs(diff_weight))

        kernel = K.constant([[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]])
        x = K.reshape(x=weight, shape=[1, self.row, self.col, 1])
        kernel = K.reshape(kernel, [3, 3, 1, 1])
        xy_diff = K.sum(K.abs(K.conv2d(x, kernel))) / self.batch_size
        diff = self.reg_xy_diff_ratio * xy_diff + self.reg_z_diff_ratio * z_diff

        res = self.reg_res_ratio * K.sum(K.abs(weight)) / self.batch_size
        return diff + res

    def generator(self):
        if self.G:
            return self.G
        # This returns a tensor
        gin = Input(shape=(self.row, self.col, ), dtype=K.tf.complex64, name='gin')
        x = gin

        anglex = Input(shape=(1, 1, ), name='anglex')
        angley = Input(shape=(1, 1, ), name='angley')

        # a layer instance is callable on a tensor, and returns a tensor
        sharedDiffractionLayer = DiffractionLayer(name='DiffractionLayer',
                                                     M=self.row, N=self.col,
                                                     drz=self.drz, dphi=self.dphi,
                                                     dtype=K.tf.complex64,
                                                     input_dtype=K.tf.complex64)

        sharedFrequencyShiftLayer = FrequencyShiftLayer(name='FrequencyShiftLayer',
                                                        M=self.row, N=self.col,
                                                        Lz=self.drz, dphi=self.dphi,
                                                        input_dtype=K.tf.complex64,
                                                        dtype=K.tf.complex64)

        sharedImagLogLayer = ImagLogLayer(name='ImagLogLayer',
                                          M=self.row, N=self.col,
                                          input_dtype=K.tf.complex64)

        sharedAddLayer = AddLayer(name='AddLayer', M=self.row, N=self.col)


        for i in range(self.layer_num / 2):
            x_before = x
            x = sharedFrequencyShiftLayer(x)
            y = sharedImagLogLayer([x, x_before])
            if i == 0:
                my_unwrap = y
            else:
                my_unwrap = sharedAddLayer([my_unwrap, y])


        for i in range(self.layer_num):
            x_before = x

            x = sharedDiffractionLayer(x)

            x = RefractionLayer(name='RefractionLayer_%d' % i,
                                output_dim=self.col,
                                drz=self.drz,
                                k0=self.k0,
                                prop_window=self.prop_window,
                                kernel_constraint='non_neg',
                                kernel_initializer=initializers.constant(np.zeros([self.row, self.col])),
                                #kernel_initializer=initializers.constant(0.007 * np.ones([self.row, self.col])),
                                #kernel_initializer=initializers.constant(self.initial_n[:, :, i]),
                                kernel_regularizer=self.my_reg
                                )([x, anglex, angley])

            y = sharedImagLogLayer([x, x_before])

            my_unwrap = sharedAddLayer([my_unwrap, y])


        for i in range(self.layer_num / 2):
            x_before = x
            x = sharedFrequencyShiftLayer(x)
            y = sharedImagLogLayer([x, x_before])
            my_unwrap = sharedAddLayer([my_unwrap, y])
        """
        my_unwrap = PropWindowLayer(name='PropWindowLayer',
                                    M=self.row, N=self.col,
                                    prop_window=self.prop_window
                                    )(my_unwrap)
        """
        # This creates a model that includes
        # the Input layer and three Dense layers
        self.G = Model(inputs=[gin, anglex, angley], outputs=my_unwrap)
        self.G.summary()
        #plot_model(self.G, to_file='./GeneratorStructure.png')
        return self.G


    def generator_model(self):
        if self.GM:
            return self.GM
        optimizer = Adam(lr=self.lr)
        #optimizer = RMSprop(lr=1e-6, decay=3e-9)
        self.GM = Sequential()
        self.GM.add(self.generator())
        self.GM.compile(loss=self.my_loss, optimizer=optimizer)
        return self.GM

