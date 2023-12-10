import tensorflow as tf

class ResNet(tf.keras.Model):

    def __init__(self, variant="18"):

        super().__init__()

        resnet_sizes = {
            "18": [2, 2, 2, 2],
            "34": [3, 4, 6, 3],
            "50": [3, 4, 6, 3],
            "101": [3, 4, 23, 3],
            "152": [3, 8, 36, 3]
        }

        self.blocks_per_layer = resnet_sizes[variant]
        self.kaiming_normal = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')
        # init input shape
        self.base_model = self.__base__(variant)
    
    def call(self, inputs):
        # pass the inputs through the base model
        x = self.base_model(inputs)
        return x
    
    def __base__(self, variant):
        inputs = tf.keras.Input(shape=(None, None, 3))
        x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=self.kaiming_normal, name='conv1')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
        x = tf.keras.layers.ReLU(name='relu1')(x)
        x = tf.keras.layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)
        x = self.__make_layer__(x, 64, self.blocks_per_layer[0], name='layer1')
        x = self.__make_layer__(x, 128, self.blocks_per_layer[1], stride=2, name='layer2')
        x = self.__make_layer__(x, 256, self.blocks_per_layer[2], stride=2, name='layer3')
        x = self.__make_layer__(x, 512, self.blocks_per_layer[3], stride=2, name='layer4')
        x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
        base_model = tf.keras.Model(inputs=inputs, outputs=x, name="resnet{}".format(variant))
        return base_model

    def __conv3x3__(self, x, out_planes, stride=1, name=None):
        x = tf.keras.layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
        return tf.keras.layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=self.kaiming_normal, name=name)(x)

    def __basic_block__(self, x, planes, stride=1, downsample=None, name=None):
        identity = x

        out = self.__conv3x3__(x, planes, stride=stride, name=f'{name}.conv1')
        out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
        out = tf.keras.layers.ReLU(name=f'{name}.relu1')(out)

        out = self.__conv3x3__(out, planes, name=f'{name}.conv2')
        out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

        if downsample is not None:
            for layer in downsample:
                identity = layer(identity)

        out = tf.keras.layers.Add(name=f'{name}.add')([identity, out])
        out = tf.keras.layers.ReLU(name=f'{name}.relu2')(out)

        return out

    def __make_layer__(self, x, planes, blocks, stride=1, name=None):
        downsample = None
        inplanes = x.shape[3]
        if stride != 1 or inplanes != planes:
            downsample = [
                tf.keras.layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=self.kaiming_normal, name=f'{name}.0.downsample.0'),
                tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
            ]

        x = self.__basic_block__(x, planes, stride, downsample, name=f'{name}.0')
        for i in range(1, blocks):
            x = self.__basic_block__(x, planes, name=f'{name}.{i}')

        return x
