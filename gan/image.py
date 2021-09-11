import tensorflow as tf

def _default_postprocess_fn(x):
    return (x + 1.) / 2.
    
class ImageGenerator(tf.keras.Model):
    def __init__(self,
                 model,
                 summary_name = 'generated_image',
                 summary_kwargs = {},
                 summary_postprocess_fn = _default_postprocess_fn,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.summary_name = summary_name
        self.summary_kwargs = summary_kwargs
        self.summary_postprocess = summary_postprocess_fn

    def call(self, x):
        y = self.model(x)
        tf.summary.image(
            self.summary_name,
            self.summary_postprocess(y),
            **self.summary_kwargs)
        return y

class ImageDiscriminator(tf.keras.Model):
    def _default_postprocess_fn(x):
        return (x + 1.) / 2.

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = tf.keras.Model(
            model.input, (model.output, {}))
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

        if self.trainable:
            for name, decoder in self.reconstruction_models.items():
                params = self.reconstruction_params[name]
                image = x
                hidden = h[name]
                #print(tf.shape(image)[0] % 2)
                #if tf.shape(image)[0] % 2 == 0:
                n = tf.shape(image)[0] // 2
                #image, _ = tf.split(image, 2, axis=0)
                #hidden, _ = tf.split(hidden, 2, axis=0)
                image = image[:n]
                hidden = hidden[:n]
                if params['crop']:
                    part = tf.random.uniform([], 0, 4, dtype=tf.int32)
                    image = self._crop(image, part)
                    hidden = self._crop(hidden, part)
                reconst = decoder(hidden)
                tf.summary.image(
                    params['summary_name'],
                    params['postprocess'](reconst),
                    **params['summary_kwargs'])
                image = tf.image.resize(image, tf.shape(reconst)[1:3])
                loss = tf.reduce_mean(params['loss'](image, reconst))
                self.add_loss(loss * params['weight'])
        return y
