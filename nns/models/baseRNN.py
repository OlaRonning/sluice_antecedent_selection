import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import functools

from ..custom_tf.zoneout import ZoneoutWrapper

def lazy_property(func):
    attribute = '_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self,attribute):
            setattr(self,attribute,func(self))
        return getattr(self,attribute)
    return wrapper

class network(object):
    def __init__(self,data,target,istrain,embeddings,config):
        """
        
        Inputs:
            data: batch of sentence indecies (batch_size x max_len)
            config: num_hidden, num_layers, max_len, num_classes, stddev, num_embeddings

        """
        self._conf = config
        self.targets = target
        self.data = data
        self.istrain = istrain
        self.embeddings = embeddings
        self.updated_embeddings = embeddings
        self.predictions
        self.optimize
        self.summary

    def random_uniform(self):
        #should be hyp parameter
        return tf.random_uniform_initializer(-0.08,0.08)

    @lazy_property
    def length(self) :
        """ Computes length sequences  in batch
        Assumes:
            Embeddings have been looked up
        """
        length = tf.count_nonzero(self.data+1,axis=-1,dtype=tf.int32)
        return length


    @lazy_property
    def lookup(self):
        embs = tf.Variable(tf.zeros([self._conf.batch_size,self._conf.max_len,self._conf.num_embeddings],dtype=tf.float32),
                dtype=tf.float32,trainable=False)
        embs = tf.nn.embedding_lookup(self.embeddings,tf.abs(self.data))
        embs = tf.cast(embs,tf.float32)
        up_embs = tf.Variable(tf.zeros([self._conf.batch_size,self._conf.max_len,self._conf.num_embeddings],dtype=tf.float32),
                dtype=tf.float32)
        up_embs = tf.nn.embedding_lookup(self.updated_embeddings,tf.abs(self.data))
        up_embs = tf.cast(embs,tf.float32)
        #embs = tf.concat([embs,up_embs],-1)
        return embs

    @lazy_property
    def predictions(self):
        fw_cells = [ZoneoutWrapper(rnn.LSTMCell(self._conf.num_hidden),
            zoneout_prob=(self._conf.z_prob_cell,self._conf.z_prob_states),is_training=self.istrain) for _ in range(self._conf.num_layers)]
        fw_cell = rnn.MultiRNNCell(fw_cells)

        bw_cells = [ZoneoutWrapper(rnn.LSTMCell(self._conf.num_hidden),
            zoneout_prob=(self._conf.z_prob_cell,self._conf.z_prob_states),is_training=self.istrain) for _ in range(self._conf.num_layers)]
        bw_cell = rnn.MultiRNNCell(bw_cells)
        output,_ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                self.lookup,
                dtype=tf.float32,
                sequence_length=self.length
        )
        with tf.name_scope('prediction') :
            fw_output,bw_output = output
            fw_weight,fw_bias = self._weight_and_bias(self._conf.num_hidden,self._conf.num_classes,layer_name='fw_softmax_')
            bw_weight,bw_bias = self._weight_and_bias(self._conf.num_hidden,self._conf.num_classes,layer_name='bw_softmax_')
            fw_output = tf.reshape(fw_output,[-1,self._conf.num_hidden],name="forward_reshape")
            bw_output = tf.reshape(bw_output,[-1,self._conf.num_hidden],name="backward_reshape")
            predictions = ((tf.matmul(fw_output,fw_weight) + fw_bias) + (tf.matmul(bw_output,bw_weight) + bw_bias))
            predictions = tf.reshape(predictions,[-1,self._conf.max_len,self._conf.num_classes])
            return predictions


    @lazy_property
    def labels(self):
        labels = tf.cast(tf.argmax(self.predictions,axis=-1),tf.int32)
        return labels

    @lazy_property
    def cost(self):
        with tf.name_scope('cost') :
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predictions,labels=self.targets)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            losses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses)
            return cost

    @lazy_property
    def optimize(self):
        with tf.name_scope('optmize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate)
            grads = optimizer.compute_gradients(self.cost)
            applied_gradients = optimizer.apply_gradients(grads)
            return applied_gradients

    @lazy_property
    def _perplexity(self):
        perp = tf.pow(2.,-self.cost)
        return perp

    @lazy_property
    def summary(self):
        tf.summary.scalar('Cost',self.cost)
        tf.summary.scalar('Perplexity',self._perplexity)
        merged = tf.summary.merge_all()
        return merged
    
    @staticmethod
    def _weight_and_bias(in_channels,out_channels,layer_name='layer_'):
        bias = tf.constant(0,shape=[out_channels],dtype=tf.float32)
        bias = tf.Variable(bias,name=layer_name+'B')
        weight = tf.get_variable(layer_name+"W",shape=[in_channels,out_channels],initializer=tf.contrib.layers.xavier_initializer(False))
        return weight,bias
