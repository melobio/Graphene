import tensorflow.compat.v1 as tf

from . import inits

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class MultiLayer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties    
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """

    def __init__(self, edge_type=(), num_types=-1, **kwargs):
        self.edge_type = edge_type
        self.num_types = num_types
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolutionSparseMulti(MultiLayer):
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj_mats,
                 nonzero_feat, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseMulti, self).__init__(**kwargs)
        self.dropout = dropout
        self.adj_mats = adj_mats
        self.act = act
        self.issparse = True
        self.nonzero_feat = nonzero_feat
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(
                    input_dim[self.edge_type[1]], output_dim, name='weights_%d' % k)

    def _call(self, inputs):
        outputs = []
        for k in range(self.num_types):
            x = dropout_sparse(inputs, 1 - self.dropout, self.nonzero_feat[self.edge_type[1]])
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights_%d' % k])
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
            outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, dim=1)
        return outputs


class GraphConvolutionMulti(MultiLayer):
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj_mats, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionMulti, self).__init__(**kwargs)
        self.adj_mats = adj_mats
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(
                    input_dim, output_dim, name='weights_%d' % k)

    def _call(self, inputs):
        outputs = []
        for k in range(self.num_types):
            x = tf.nn.dropout(inputs, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights_%d' % k])
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
            outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, dim=1)
        return outputs


class GraphAttentionMulti(MultiLayer):
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim,adj_mats,  dropout, adj_bias=None,
                 in_drop=0.0, coef_drop=0.0, residual=False, act=tf.nn.relu, **kwargs):
        super(GraphAttentionMulti, self).__init__(**kwargs)
        self.output_dim = output_dim
        if self.output_dim % 1 != 0:
            raise Exception('output_dim have to be divided by 8')
        self.head_out_dim = int(self.output_dim / 1)
        self.input_dim = input_dim
        self.adj_mats = adj_mats
        self.adj_bias = adj_bias
        self.dropout = dropout
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        # self.nonzero_feat = nonzero_feat
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for type in range(self.num_types):
                self.vars['weights_%d' % type] = inits.weight_variable_glorot(
                    self.head_out_dim, self.head_out_dim, name='weights_%d' % type)
                for k in range(1):
                    self.vars['kernel_%d_%d' % (type, k)] = inits.weight_variable_glorot(
                        input_dim, self.head_out_dim, name='kernel_%d_%d' % (type, k))
                    self.vars['fl_%d_%d' % (type, k)] = inits.weight_variable_glorot(
                        self.head_out_dim, 1, name='fl_%d_%d' % (type, k))
                    self.vars['fr_%d_%d' % (type, k)] = inits.weight_variable_glorot(
                        self.head_out_dim, 1, name='fr_%d_%d' % (type, k))
                # 构造最后一层
                self.vars['kernel_%d_%d' % (type, -1)] = inits.weight_variable_glorot(
                    self.output_dim, output_dim, name='kernel_%d_%d' % (type, -1))
                self.vars['fl_%d_%d' % (type, -1)] = inits.weight_variable_glorot(
                    output_dim, 1, name='fl_%d_%d' % (type, -1))
                self.vars['fr_%d_%d' % (type, -1)] = inits.weight_variable_glorot(
                    output_dim, 1, name='fr_%d_%d' % (type, -1))

    def attn_head(self, x, head_n, type_n, in_drop=0.0, coef_drop=0.0, residual=False):
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                x = tf.nn.dropout(x, 1.0 - in_drop)
            x_fts = tf.matmul(x, self.vars['kernel_%d_%d' % (type_n, head_n)])

            # simplest self-attention possible
            f_1 = tf.matmul(x_fts, self.vars['fl_%d_%d' % (type_n, head_n)])
            f_2 = tf.matmul(x_fts, self.vars['fr_%d_%d' % (type_n, head_n)])
            logits = f_1 + tf.transpose(f_2, [1, 0])
            # coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
            # coefs = tf.nn.softmax(tf.sparse_add(tf.nn.leaky_relu(logits), self.adj_bias[self.edge_type[1]][0]))
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                x_fts = tf.nn.dropout(x_fts, 1.0 - in_drop)

            vals = tf.matmul(coefs, x_fts)
            gat_result = tf.nn.elu(vals)

            x = tf.matmul(gat_result, self.vars['weights_%d' % type_n])
            ret = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][type_n], x)
            return ret

    def _call(self, inputs):
        outputs = []
        for k in range(self.num_types):
            attns = []
            for n in range(1):
                attns.append(self.attn_head(inputs, head_n=n, type_n=k, in_drop=0,
                                            coef_drop=0, residual=False))
            h_1 = tf.concat(attns, axis=-1)
            outputs.append(self.act(h_1))

        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, dim=1)
        return outputs


class DEDICOMDecoder(MultiLayer):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DEDICOMDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['global_interaction'] = inits.weight_variable_glorot(
                input_dim, input_dim, name='global_interaction')
            for k in range(self.num_types):
                tmp = inits.weight_variable_glorot(
                    input_dim, 1, name='local_variation_%d' % k)
                self.vars['local_variation_%d' % k] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1 - self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1 - self.dropout)
            relation = tf.diag(self.vars['local_variation_%d' % k])
            product1 = tf.matmul(inputs_row, relation)
            product2 = tf.matmul(product1, self.vars['global_interaction'])
            product3 = tf.matmul(product2, relation)
            rec = tf.matmul(product3, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class DistMultDecoder(MultiLayer):
    """DistMult Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DistMultDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                tmp = inits.weight_variable_glorot(
                    input_dim, 1, name='relation_%d' % k)
                self.vars['relation_%d' % k] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1 - self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1 - self.dropout)
            relation = tf.diag(self.vars['relation_%d' % k])
            intermediate_product = tf.matmul(inputs_row, relation)
            rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class BilinearDecoder(MultiLayer):
    """Bilinear Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(BilinearDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['relation_%d' % k] = inits.weight_variable_glorot(
                    input_dim, input_dim, name='relation_%d' % k)

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1 - self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1 - self.dropout)
            intermediate_product = tf.matmul(inputs_row, self.vars['relation_%d' % k])
            rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs


class InnerProductDecoder(MultiLayer):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1 - self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1 - self.dropout)
            rec = tf.matmul(inputs_row, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs
