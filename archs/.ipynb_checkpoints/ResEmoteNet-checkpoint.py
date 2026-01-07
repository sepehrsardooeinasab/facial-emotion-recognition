import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers

class SEBlock(layers.Layer):
    def __init__(self, in_channels, reduction=16, lambda_reg=1e-2):
        super(SEBlock, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(in_channels // reduction, activation='relu', use_bias=False, kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        self.fc2 = layers.Dense(in_channels, activation='sigmoid', use_bias=False, kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        
    def call(self, x):
        b, h, w, c = x.shape
        y = self.avg_pool(x)  # Global average pooling
        y = self.fc1(y)  # First dense layer
        y = self.fc2(y)  # Second dense layer
        y = tf.reshape(y, (-1, 1, 1, c))
        return x * y  # Scale the input by the attention weights

    
class ResidualBlock(layers.Layer):
    def __init__(self, in_ch, out_ch, stride=1, lambda_reg=1e-2):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_ch, 3, strides=stride, padding='same', kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv2 = layers.Conv2D(out_ch, 3, padding='same', kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        self.bn2 = layers.BatchNormalization(momentum=0.1, epsilon=1e-05)
        
        if stride != 1:
            self.shortcut = models.Sequential([
                layers.Conv2D(out_ch, 1, strides=stride, kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg)),
                layers.BatchNormalization(momentum=0.1, epsilon=1e-05)
            ])
        else:
            self.shortcut = lambda x: x  # Identity shortcut

    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tf.nn.relu(out)
        return out
    

class ResEmoteNet(models.Model):
    def __init__(self, num_class, lambda_reg=1e-2):
        super(ResEmoteNet, self).__init__()
        self.conv1 = layers.Conv2D(64, 3, padding='same', kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        self.bn1 = layers.BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv2 = layers.Conv2D(128, 3, padding='same', kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        self.bn2 = layers.BatchNormalization(momentum=0.1, epsilon=1e-05)
        self.conv3 = layers.Conv2D(256, 3, padding='same', kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        self.bn3 = layers.BatchNormalization(momentum=0.1, epsilon=1e-05)
        
        self.se = SEBlock(256, lambda_reg=lambda_reg)
        
        self.res_block1 = ResidualBlock(256, 512, stride=2, lambda_reg=lambda_reg)
        self.res_block2 = ResidualBlock(512, 1024, stride=2, lambda_reg=lambda_reg)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2, lambda_reg=lambda_reg)
        
        self.pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(1024, activation='relu', kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        self.fc2 = layers.Dense(512, activation='relu', kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        self.fc3 = layers.Dense(256, activation='relu', kernel_initializer=initializers.HeNormal(), bias_initializer=initializers.Constant(0.1), kernel_regularizer=regularizers.l2(lambda_reg))
        self.dropout1 = layers.Dropout(0.2)
        self.dropout2 = layers.Dropout(0.5)
        self.fc4 = layers.Dense(num_class, activation='softmax', kernel_initializer=initializers.GlorotNormal(), bias_initializer=initializers.Constant(0.1))

    def call(self, x):
        x = tf.nn.relu(self.bn1(self.conv1(x)))
        x = layers.MaxPooling2D(2)(x)
        x = self.dropout1(x)
        
        x = tf.nn.relu(self.bn2(self.conv2(x)))
        x = layers.MaxPooling2D(2)(x)
        x = self.dropout1(x)
        
        x = tf.nn.relu(self.bn3(self.conv3(x)))
        x = layers.MaxPooling2D(2)(x)
        x = self.se(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.pool(x)  # Global average pooling
        x = layers.Flatten()(x)
        x = tf.nn.relu(self.fc1(x))
        x = self.dropout2(x)
        x = tf.nn.relu(self.fc2(x))
        x = self.dropout2(x)
        x = tf.nn.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)
        return x