from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import numpy


class DiffractionLayer(Layer):
	# Define a layer according to the theory of BPM for diffraction
    def __init__(self, M, N, drz, dphi, **kwargs):
        self.M = M
        self.N = N
        self.drz = drz
        self.dphi = dphi
        super(DiffractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DiffractionLayer, self).build(input_shape)

    def call(self, x):
        temp = K.tf.exp(-1j * self.drz * self.dphi)
        temp = K.tf.cast(temp, K.tf.complex64)
        return K.tf.ifft2d(K.tf.fft2d(x) * temp)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M, self.N)


class RefractionLayer(Layer):
	# Define a layer according to the theory of BPM for refraction
    def __init__(self, output_dim, drz, k0, prop_window, kernel_constraint=None, kernel_initializer='glorot_uniform', kernel_regularizer=None, **kwargs):
        self.output_dim = output_dim
        self.drz = drz
        self.k0 = k0
        self.prop_window = prop_window
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(RefractionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='Refraction_weight',
                                        shape=(input_shape[0][1], self.output_dim),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True)
        super(RefractionLayer, self).build(input_shape)

    def call(self, input_list):
        x = input_list[0]
        anglex = input_list[1]
        angley = input_list[2]
        cos_theta = 1 / K.sqrt(K.tf.tan(anglex) ** 2 + K.tf.tan(angley) ** 2 + 1)
        temp = self.drz * self.k0 * self.kernel / (cos_theta)
        real_kernel = K.tf.exp(K.tf.complex(real=K.tf.zeros_like(temp), imag=temp))
        return x * real_kernel

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.output_dim)

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        }
        base_config = super(RefractionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FrequencyShiftLayer(Layer):
    def __init__(self, M, N, Lz, dphi, **kwargs):
        self.M = M
        self.N = N
        self.Lz = Lz
        self.dphi = dphi
        super(FrequencyShiftLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FrequencyShiftLayer, self).build(input_shape)

    def call(self, x):
        temp = K.tf.exp(-1j * (-self.Lz) * self.dphi)
        temp = K.tf.cast(temp, K.tf.complex64)
        return K.tf.ifft2d(K.tf.fft2d(x) * temp)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M, self.N)


class LowPassLayer(Layer):
    def __init__(self, M, N, Lz, dphi, MyFilter, **kwargs):
        self.M = M
        self.N = N
        self.Lz = Lz
        self.dphi = dphi
        self.filter = self.get_filter(MyFilter)
        super(LowPassLayer, self).__init__(**kwargs)

    def get_filter(self, MyFilter):
        myfilter = numpy.zeros_like(MyFilter)
        myfilter[0: self.M / 2, 0: self.N / 2] = MyFilter[self.M / 2: self.M, self.N / 2: self.N]
        myfilter[self.M / 2: self.M, 0: self.N / 2] = MyFilter[0: self.M / 2, self.N / 2: self.N]
        myfilter[0: self.M / 2, self.N / 2: self.N] = MyFilter[self.M / 2: self.M, 0: self.N / 2]
        myfilter[self.M / 2: self.M, self.N / 2: self.N] = MyFilter[0: self.M / 2, 0: self.N / 2]
        return myfilter

    def build(self, input_shape):
        super(LowPassLayer, self).build(input_shape)

    def call(self, input_list):
        # Low pass simulation to fit experimental data
        x = input_list[0]
        gin = input_list[1]

        angle = K.tf.atan2(K.tf.imag(gin), K.tf.real(gin))
        angle = K.tf.complex(real=K.zeros_like(angle), imag=angle)

        u = K.tf.ifft2d(K.tf.fft2d(x / K.tf.exp(angle)) * self.filter) * K.tf.exp(angle)
        return u

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M, self.N)


class ImagLogLayer(Layer):
    def __init__(self, M, N, **kwargs):
        self.M = M
        self.N = N
        super(ImagLogLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ImagLogLayer, self).build(input_shape) 

    def call(self, input_list):
        x = input_list[0]
        x_before = input_list[1]
        result = K.tf.imag(K.tf.log((x + 1e-9) / (x_before + 1e-9)))
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M, self.N)


class AddLayer(Layer):
    def __init__(self, M, N, **kwargs):
        self.M = M
        self.N = N
        super(AddLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AddLayer, self).build(input_shape)

    def call(self, input_list):
        x = input_list[0]
        y = input_list[1]
        return x + y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M, self.N)


class PropWindowLayer(Layer):
    def __init__(self, M, N, prop_window, **kwargs):
        self.M = M
        self.N = N
        self.prop_window = prop_window
        super(PropWindowLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PropWindowLayer, self).build(input_shape)

    def call(self, x):
        return self.prop_window * x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.M, self.N)