from collections import namedtuple

from tensorflow.python.keras.initializers import RandomNormal

DEFAULT_GROUP_NAME = "default_group"


class SparseFeature(namedtuple('SparseFeature',
                               ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype',
                                'embeddings_initializer',
                                'embedding_name',
                                'group_name', 'trainable'])):
    __slot__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embeddings_initializer=None,
                embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeature, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                                 embeddings_initializer,
                                                 embedding_name, group_name, trainable)


class VarLenSparseFeature(namedtuple('VarLenSparseFeature',
                                     ['sparsefeature', 'maxlen', 'combiner', 'length_name', 'weight_name',
                                      'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeature, maxlen, combiner="mean",
                length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeature, cls).__new__(cls, sparsefeature, maxlen, combiner, length_name, weight_name,
                                                       weight_norm)

    @property
    def name(self):
        return self.sparsefeature.name

    @property
    def vocabulary_size(self):
        return self.sparsefeature.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeature.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeature.use_hash

    @property
    def dtype(self):
        return self.sparsefeature.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeature.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeature.embedding_name

    @property
    def group_name(self):
        return self.sparsefeature.group_name

    @property
    def trainable(self):
        return self.sparsefeature.trainable

    def __hash__(self):
        return self.name.__hash__()


class DenseFeature(namedtuple('DenseFeature', ['name', 'dimension', 'dtype', 'transform_fn'])):
    __slot__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", transform_fn=None):
        return super(DenseFeature, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()
