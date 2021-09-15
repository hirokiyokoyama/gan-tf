import tensorflow as tf

def matrix_sqrt(A, iteration=12):
    """ CR iteration (Meini, 2004)
    """
    dim = tf.shape(A)[0]
    Y = tf.eye(dim) - A
    Z = (tf.eye(dim) + A) * 2.
    for _ in range(iteration):
        Y = -tf.matmul(Y, tf.linalg.solve(Z, Y))
        Z = Z + 2. * Y
    return Z * 0.25

class FID(tf.keras.metrics.Metric):
    def __init__(self, name='fid', image_size=[299,299], epsilon=1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.real_moment1 = tf.keras.metrics.MeanTensor('real_moment1')
        self.real_moment2 = tf.keras.metrics.MeanTensor('real_moment2')
        self.fake_moment1 = tf.keras.metrics.MeanTensor('fake_moment1')
        self.fake_moment2 = tf.keras.metrics.MeanTensor('fake_moment2')
        self.image_size = list(image_size)
        self.epsilon = epsilon
        shape = self.image_size + [3]
        self.inception_model = tf.keras.applications.InceptionV3(
            include_top = False,
            weights = 'imagenet',
            pooling = 'avg',
            input_shape = shape)
        self.inception_dim = int(self.inception_model.compute_output_shape([None]+shape)[-1])
        self.update_real = True
        self.update_fake = True
    
    def reset_state(self):
        if self.update_real:
            self.real_moment1.reset_state()
            self.real_moment2.reset_state()
        if self.update_fake:
            self.fake_moment1.reset_state()
            self.fake_moment2.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        n_real = tf.shape(y_true)[0]
        n_fake = tf.shape(y_pred)[0]

        # adapt to InceptionV3
        if self.update_real and not self.update_fake:
            y = y_true
        elif not self.update_real and self.update_fake:
            y = y_pred
        elif self.update_real and self.update_fake:
            y = tf.concat([y_true, y_pred], axis=0)
        else:
            return
        y = tf.image.resize(y, self.image_size)
        if y.shape[-1] == 1:
            y = tf.tile(y, [1,1,1,3])
            
        y = self.inception_model(y)
        y_true, y_pred = tf.split(y, [n_real, n_fake], axis=0)

        if self.update_real:
            y_true = y[:n_real]
            real_moment1 = tf.reduce_mean(y_true, axis=0)
            real_moment2 = tf.reduce_mean(
                y_true[:,:,tf.newaxis] * y_true[:,tf.newaxis,:], axis=0)
            self.real_moment1.update_state(real_moment1, sample_weight=n_real)
            self.real_moment2.update_state(real_moment2, sample_weight=n_real)

        if self.update_fake:
            y_pred = y[-n_fake:]
            fake_moment1 = tf.reduce_mean(y_pred, axis=0)
            fake_moment2 = tf.reduce_mean(
                y_pred[:,:,tf.newaxis] * y_pred[:,tf.newaxis,:], axis=0)
            self.fake_moment1.update_state(fake_moment1, sample_weight=n_fake)
            self.fake_moment2.update_state(fake_moment2, sample_weight=n_fake)

    def result(self):
        real_moment1 = self.real_moment1.result()
        real_moment2 = self.real_moment2.result()
        fake_moment1 = self.fake_moment1.result()
        fake_moment2 = self.fake_moment2.result()
        real_cov = real_moment2 - real_moment1[:,tf.newaxis] * real_moment1[tf.newaxis,:]
        fake_cov = fake_moment2 - fake_moment1[:,tf.newaxis] * fake_moment1[tf.newaxis,:]
        
        fid = tf.reduce_sum(tf.square(real_moment1 - fake_moment1))
        fid += tf.linalg.trace(fake_cov) + tf.linalg.trace(real_cov)
        cov = tf.matmul(fake_cov, real_cov) + tf.eye(self.inception_dim) * self.epsilon

        cov = matrix_sqrt(cov)
        fid -= 2. * tf.linalg.trace(cov)
        return fid
