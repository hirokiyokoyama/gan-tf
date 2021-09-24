import tensorflow as tf

class ConditionalGAN(tf.keras.Model):
    def __init__(self, generator, discriminator,
                 noise_fn = None,
                 frechet_distance = None,
                 gradient_penalty_weight = 1.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.frechet_distance = frechet_distance

        if noise_fn is None:
            def noise_fn(batch_size):
                return tf.random.normal([batch_size,1,1,128])
        self.noise_fn = noise_fn
        self.gradient_penalty_weight = gradient_penalty_weight

        #g_input = noise_fn(1)
        #d_input = self.generator(g_input)
        #d_output = self.discriminator(d_input)
        #self.d_output_shape = d_output.shape[1:]
        self.output_names = ('generator', 'discriminator')

    @property
    def metrics(self):
        metrics = super().metrics
        # calculating frechet distance is expensive
        # so avoid calculating it at every training step
        if self.frechet_distance is not None:
            metrics.remove(self.frechet_distance)
        return metrics

    def reset_metrics(self):
        super().reset_metrics()
        if self.frechet_distance is not None:
            self.frechet_distance.reset_state()

    def call(self, inputs):
        real, cond = inputs
        n = tf.shape(real)[0]
        noise = self.noise_fn(n)
        fake = self.generator((noise, cond))

        if self.discriminator.trainable:
            d_inputs = (tf.concat([real, fake], axis=0),
                        tf.concat([cond, cond], axis=0))
            d_outputs_g = None
            d_outputs_d = self.discriminator(d_inputs)

            d_real, d_fake = tf.split(d_outputs_d, 2, axis=0)
            tf.summary.histogram('d_step/d_real', d_real)
            tf.summary.histogram('d_step/d_fake', d_fake)

            if self.gradient_penalty_weight > 0.:
                shape = tf.concat([[n], tf.ones([tf.rank(real)-1], tf.int32)], 0)
                a = tf.random.uniform(shape)
                x = real * a + fake * (1. - a)
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    y = self.discriminator(x)
                d_grads = tape.gradient(y, x)
                gp_loss = tf.reduce_sum(tf.square(d_grads)) / tf.cast(n, tf.float32)
            else:
                gp_loss = 0.
        else:
            d_inputs = fake, cond
            d_outputs_g = self.discriminator(d_inputs)
            tf.summary.histogram('g_step/d_fake', d_outputs_g)
            d_outputs_d = None
            gp_loss = 0.
        self.add_loss(self.gradient_penalty_weight * gp_loss)

        if self.frechet_distance is not None:
            if self.discriminator.trainable:
                self.frechet_distance.update_state(real, fake)

        return d_outputs_g, d_outputs_d

    def train_step(self, data):
        real, cond = data
        n = tf.shape(real)[0]
        shape = tf.concat([[n], tf.ones([tf.rank(real)-1], tf.int32)], 0)
 
        # discriminator step (real and fake)
        y_true = tf.concat([tf.ones(shape), tf.zeros(shape)], axis=0)
        self.discriminator.trainable = True
        self.generator.trainable = False
        super().train_step(((real, cond), (None, y_true)))

        # generator step (fake only)
        y_true = tf.ones(shape)
        self.discriminator.trainable = False
        self.generator.trainable = True
        losses = super().train_step(((real, cond), (y_true, None)))

        if self.frechet_distance is not None:
            tf.cond(tf.summary.should_record_summaries(),
                    lambda: tf.summary.scalar('frechet_distance', self.frechet_distance.result()),
                    lambda: False, name=self.name)
        return losses
