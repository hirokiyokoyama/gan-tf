import tensorflow as tf

class GAN(tf.keras.Model):
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

        g_input_shape = noise_fn(1).shape
        d_input_shape = self.generator.compute_output_shape(g_input_shape)
        d_output_shape = self.discriminator.compute_output_shape(d_input_shape)
        self.d_output_shape = d_output_shape[1:]

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

    def call(self, real):
        n = tf.shape(real)[0]
        noise = self.noise_fn(n)
        fake = self.generator(noise)

        if self.discriminator.trainable:
            d_inputs = tf.concat([real, fake], axis=0)
            with tf.GradientTape() as tape:
                tape.watch(d_inputs)
                d_outputs = self.discriminator(d_inputs)
            d_grads = tape.gradient(d_outputs, d_inputs)
            gp_loss = tf.reduce_mean(tf.square(d_grads))

            d_real, d_fake = tf.split(d_outputs, 2, axis=0)
            tf.summary.histogram('d_real', d_real)
            tf.summary.histogram('d_fake', d_fake)
        else:
            d_inputs = fake
            d_outputs = self.discriminator(d_inputs)
            gp_loss = 0.
        self.add_loss(self.gradient_penalty_weight * gp_loss)

        if self.frechet_distance is not None:
            if self.discriminator.trainable:
                self.frechet_distance.update_state(real, fake)

        return d_outputs

    def train_step(self, data):
        n = tf.shape(data)[0]
        shape = tf.concat([[n], self.d_output_shape], axis=0)
 
        # discriminator step (real and fake)
        y_true = tf.concat([tf.ones(shape), tf.zeros(shape)], axis=0)
        self.discriminator.trainable = True
        self.generator.trainable = False
        d_losses = super().train_step((data, y_true))

        # generator step (fake only)
        y_true = tf.ones(shape)
        self.discriminator.trainable = False
        self.generator.trainable = True
        g_losses = super().train_step((data, y_true))

        if self.frechet_distance is not None:
            tf.cond(tf.summary.should_record_summaries(),
                    lambda: tf.summary.scalar('frechet_distance', self.frechet_distance.result()),
                    lambda: False)

        losses = {}
        losses.update({'d_'+k: v for k, v in d_losses.items()})
        losses.update({'g_'+k: v for k, v in g_losses.items()})
        return losses
