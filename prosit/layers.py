from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.engine.topology import Layer


class Attention(Layer):
    def __init__(
        self,
        context=False,
        W_regularizer=None,
        b_regularizer=None,
        u_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        u_constraint=None,
        bias=True,
        **kwargs
    ):

        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.bias = bias
        self.context = context
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            (input_shape[-1],),
            initializer=self.init,
            name="{}_W".format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        if self.bias:
            self.b = self.add_weight(
                (input_shape[1],),
                initializer="zero",
                name="{}_b".format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None
        if self.context:
            self.u = self.add_weight(
                (input_shape[-1],),
                initializer=self.init,
                name="{}_u".format(self.name),
                regularizer=self.u_regularizer,
                constraint=self.u_constraint,
            )

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        a = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)
        if self.bias:
            a += self.b
        a = K.tanh(a)
        if self.context:
            a = K.squeeze(K.dot(x, K.expand_dims(self.u)), axis=-1)
        a = K.exp(a)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = {
            "bias": self.bias,
            "context": self.context,
            "W_regularizer": regularizers.serialize(self.W_regularizer),
            "b_regularizer": regularizers.serialize(self.b_regularizer),
            "u_regularizer": regularizers.serialize(self.u_regularizer),
            "W_constraint": constraints.serialize(self.W_constraint),
            "b_constraint": constraints.serialize(self.b_constraint),
            "u_constraint": constraints.serialize(self.u_constraint),
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
