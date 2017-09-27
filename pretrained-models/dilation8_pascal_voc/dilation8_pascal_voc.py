from kaffe.tensorflow import Network

class (Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, padding='VALID', name='conv1_1')
             .conv(3, 3, 64, 1, 1, padding='VALID', name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, padding='VALID', name='conv2_1')
             .conv(3, 3, 128, 1, 1, padding='VALID', name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, padding='VALID', name='conv3_1')
             .conv(3, 3, 256, 1, 1, padding='VALID', name='conv3_2')
             .conv(3, 3, 256, 1, 1, padding='VALID', name='conv3_3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, padding='VALID', name='conv4_1')
             .conv(3, 3, 512, 1, 1, padding='VALID', name='conv4_2')
             .conv(3, 3, 512, 1, 1, padding='VALID', name='conv4_3')
             .conv(3, 3, 512, 1, 1, padding='VALID', name='conv5_1')
             .conv(3, 3, 512, 1, 1, padding='VALID', name='conv5_2')
             .conv(3, 3, 512, 1, 1, padding='VALID', name='conv5_3')
             .conv(7, 7, 4096, 1, 1, padding='VALID', name='fc6')
             .conv(1, 1, 4096, 1, 1, name='fc7')
             .conv(1, 1, 21, 1, 1, relu=False, name='fc-final')
             .conv(3, 3, 42, 1, 1, padding=None, name='ct_conv1_1')
             .conv(3, 3, 42, 1, 1, padding='VALID', name='ct_conv1_2')
             .conv(3, 3, 84, 1, 1, padding='VALID', name='ct_conv2_1')
             .conv(3, 3, 168, 1, 1, padding='VALID', name='ct_conv3_1')
             .conv(3, 3, 336, 1, 1, padding='VALID', name='ct_conv4_1')
             .conv(3, 3, 672, 1, 1, padding='VALID', name='ct_conv5_1')
             .conv(3, 3, 672, 1, 1, padding='VALID', name='ct_fc1')
             .conv(1, 1, 21, 1, 1, relu=False, name='ct_final')
             .softmax(name='prob'))