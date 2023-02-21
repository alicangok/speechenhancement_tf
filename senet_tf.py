from tensorflow.math import scalar_mul, sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Concatenate
from tensorflow import transpose, squeeze

class FusedConv2dBNReLU(Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super().__init__()
        self.conv2d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, use_bias=True)
        #self.bn = BatchNormalization(center=False,scale=False)
        self.activation = Activation('relu')

    def call(self, inputs):
        x = self.conv2d(inputs)
        #x = self.bn(x)
        x = self.activation(x)
        return x


class SENet(Model):
    def __init__(self):
        super().__init__()
        self.e1 = FusedConv2dBNReLU(filters=32, kernel_size=3, strides=1, padding='same')
        self.e2 = FusedConv2dBNReLU(filters=64, kernel_size=3, strides=1, padding='same')
        self.e3 = FusedConv2dBNReLU(filters=64, kernel_size=3, strides=1, padding='same')
        self.e4 = FusedConv2dBNReLU(filters=64, kernel_size=3, strides=1, padding='same')
        self.e5 = FusedConv2dBNReLU(filters=64, kernel_size=3, strides=1, padding='same')
        
        self.m = MaxPooling2D()
        
        self.d5 = FusedConv2dBNReLU(filters=64, kernel_size=3, strides=1, padding='same')
        self.d4 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')
        self.d3 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')
        self.d2 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')
        self.d1 = Conv2DTranspose(filters=1,  kernel_size=3, strides=2, padding='same')
        
        self.cat = Concatenate(axis=3)
    
    def call(self, x):
        e1_out = self.e1(x)
        e1_out = self.m(e1_out)
        e2_out = self.e2(e1_out)
        e2_out = self.m(e2_out)
        e3_out = self.e3(e2_out)
        e3_out = self.m(e3_out)
        e4_out = self.e4(e3_out)
        e4_out = self.m(e4_out)
        e5_out = self.e5(e4_out)
        
        d5_out = self.d5(e5_out)
        d4_out = self.d4(self.cat([d5_out, e4_out]))
        d3_out = self.d3(self.cat([d4_out, e3_out]))
        d2_out = self.d2(self.cat([d3_out, e2_out]))
        d1_out = self.d1(self.cat([d2_out, e1_out]))
        
        out = sigmoid(transpose(squeeze(d1_out), perm=[0, 2, 1]))

        return out
