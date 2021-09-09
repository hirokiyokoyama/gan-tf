import tensorflow as tf

class ImageGenerator(tf.keras.Model):
    def __init__(self,
                 model,
                 summary_name = 'generated_image',
                 summary_kwargs = {},
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.summary_name = summary_name
        self.summary_kwargs = summary_kwargs

    def call(self, x):
        y = self.model(x)
        tf.summary.image(self.summary_name, y, **self.summary_kwargs)
        return y

class ImageDiscriminator(tf.keras.Model):
    def _default_postprocess_fn(x):
        return (x + 1.) / 2.

    def __init__(self, model, channels=3, **kwargs):
        super().__init__(**kwargs)
        self.model = tf.keras.Model(
            model.input, (model.output, {}))
        self.channels = channels
        self.reconstruction_models = {}
        self.reconstruction_params = {}

    def add_reconstruction_loss(self,
        layer_name, decoder, weight=1.0, loss='mse',
        crop = False,
        postprocess_fn = _default_postprocess_fn,
        summary_name = None,
        summary_kwargs = {}):
        self.reconstruction_models[layer_name] = decoder

        params = {}
        params['crop'] = crop
        params['weight'] = tf.cast(weight, self.model.dtype)
        params['loss'] = tf.keras.losses.get(loss)
        if summary_name is None:
            summary_name = '_'.join([
                layer_name, 'part' if crop else 'full', 'reconstruction'])
        params['summary_name'] = summary_name
        params['summary_kwargs'] = summary_kwargs
        params['postprocess'] = postprocess_fn
        self.reconstruction_params[layer_name] = params

        z = {}
        for name in self.reconstruction_models:
            z[name] = self.model.get_layer(name).output
        self.model = tf.keras.Model(
            self.model.input, (self.model.output[0], z))

    def _crop(self, x, part):
        size = tf.shape(x)[1:3]
        a = size // 2

        if part == 0:
            x_part = x[:,:a[0],:a[1]]
        elif part == 1:
            x_part = x[:,:a[0],a[1]:]
        elif part == 2:
            x_part = x[:,a[0]:,:a[1]]
        else:
            x_part = x[:,a[0]:,a[1]:]
        return x_part

    def call(self, x):
        y, h = self.model(x)

        for name, decoder in self.reconstruction_models.items():
            params = self.reconstruction_params[name]
            hidden = h[name]
            if params['crop']:
                part = tf.random.uniform([], 0, 4, dtype=tf.int32)
                x = self._crop(x, part)
                hidden = self._crop(hidden, part)
            reconst = decoder(hidden)
            tf.summary.image(
                params['summary_name'],
                params['postprocess'](reconst),
                **params['summary_kwargs'])
            x_small = tf.image.resize(x, tf.shape(reconst)[1:3])
            loss = tf.reduce_mean(params['loss'](x_small, reconst))
            self.add_loss(loss * params['weight'])
        return y
