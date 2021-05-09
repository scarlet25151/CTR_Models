import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer

from .layers import PredictionLayer, reduce_sum
from utils.build_func import *

DEFAULT_GROUP_NAME = "default_group"


class FMLayer(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
    without linear term and bias.

    Input shape
      - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

    Output shape
      - 2D tensor with shape: ``(batch_size, 1)``.

    References
      - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs shape % d,\
                expect input_shape=3 " % (len(input_shape)))
        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


def FMModel(linear_feature_columns, dnn_feature_columns, fm_group=DEFAULT_GROUP_NAME, use_attention=True,
        attention_factor=8,
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_att=1e-5, afm_dropout=0, seed=1024,
        task='binary'):
    """Instantiates the Attentional Factorization Machine architecture.

      :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
      :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
      :param fm_group: list, group_name of features that will be used to do feature interactions.
      :param use_attention: bool,whether use attention or not,if set to ``False``.it is the same as **standard Factorization Machine**
      :param attention_factor: positive integer,units in attention net
      :param l2_reg_linear: float. L2 regularizer strength applied to linear part
      :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
      :param l2_reg_att: float. L2 regularizer strength applied to attention net
      :param afm_dropout: float in [0,1), Fraction of the attention net output units to dropout.
      :param seed: integer ,to use as random seed.
      :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
      :return: A Keras model instance.
      """

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                         seed, support_dense=False, support_group=True)

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    fm_logit = add_func([FMModel()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    final_logit = add_func([linear_logit, fm_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model