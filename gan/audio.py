import tensorflow as tf

class MFCCGenerator(tf.keras.Model):
    def __init__(self,
                 model,
                 mfcc,
                 summary_name = 'generated_mfcc',
                 summary_kwargs = {},
                 summary_postprocess_fn = tf.identity,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.mfcc = mfcc
        self.summary_name = summary_name
        self.summary_kwargs = summary_kwargs
        self.summary_postprocess = summary_postprocess_fn

    def call(self, x):
        y = self.model(x)

        def summary():
            y_img = (y + 1.) / 2.
            y_img = y_img[...,tf.newaxis]
            s1 = tf.summary.image(
                self.summary_name+'_mfcc',
                y_img,
                **self.summary_kwargs)
            y_audio = self.summary_postprocess(y)
            y_audio = self.mfcc.decode(y_audio)[...,tf.newaxis]
            s2 = tf.summary.audio(
                self.summary_name,
                y_audio,
                self.mfcc.sample_rate,
                **self.summary_kwargs)
            return tf.logical_and(s1, s2)
        tf.cond(tf.summary.should_record_summaries(), summary, lambda: False)
        return y
