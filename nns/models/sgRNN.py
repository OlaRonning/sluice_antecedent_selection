import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import functools
#import datasets as dset

from ..custom_tf.cascading_rnn_cell import MultiRNNCell
from ..custom_tf.zoneout import ZoneoutWrapper
from ..custom_tf.rnn import cascading_bidirectional_dynamic_rnn

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

        self.rnn

        ## init prediction
        self.ant_preds
        self.clause_preds
        self.chunk_preds
        self.com_preds

        ## init cost
        self.ant_cost
        self.clause_optimize
        self.chunk_cost
        self.com_cost

        ## init optimizers
        self.ant_optimize
        self.clause_optimize
        self.chunk_optimize
        self.com_optimize

        ## init summary
        self.summary

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
        embs = tf.Variable(tf.zeros([self._conf.batch_size,self._conf.max_len,self._conf.num_embeddings],dtype=tf.float32),dtype=tf.float32,trainable=False)
        embs = tf.nn.embedding_lookup(self.embeddings,tf.abs(self.data))
        embs = tf.cast(embs,tf.float32)
        return embs

    @lazy_property
    def rnn(self):
        with tf.name_scope('shared_layer'):
            fw_cells = [ZoneoutWrapper(rnn.LSTMCell(self._conf.num_hidden),
                zoneout_prob=(self._conf.z_prob_cell,self._conf.z_prob_states),is_training=self.istrain) 
                for _ in range(self._conf.num_layers)]
            fw_cell = MultiRNNCell(fw_cells)
            bw_cells = [ZoneoutWrapper(rnn.LSTMCell(self._conf.num_hidden),
                zoneout_prob=(self._conf.z_prob_cell,self._conf.z_prob_states),is_training=self.istrain) 
                for _ in range(self._conf.num_layers)]
            bw_cell = MultiRNNCell(bw_cells)

            output,_ = cascading_bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    self.lookup,
                    dtype=tf.float32,
                    sequence_length=self.length
            )

            return output

    @lazy_property
    def ant_preds(self):
        with tf.name_scope('ant_preds') :
            predictions = self.full_layer(self.rnn[2],self._conf.num_ant_classes,'ant_pred')
            return predictions

    @lazy_property
    def clause_preds(self):
        with tf.name_scope('clause_preds') :
            predictions = self.full_layer(self.rnn[1],self._conf.num_clause_classes,'clause_pred')
            return predictions

    @lazy_property
    def chunk_preds(self):
        with tf.name_scope('chunk_preds') :
            predictions = self.full_layer(self.rnn[1],self._conf.num_chunk_classes,'chunk_pred')
            return predictions

    @lazy_property
    def com_preds(self):
        with tf.name_scope('com_preds') :
            predictions = self.full_layer(self.rnn[0],self._conf.num_com_classes,'com_pred')
            return predictions


    @lazy_property
    def ant_labels(self):
        with tf.name_scope('ant_labels') :
            labels = tf.cast(tf.argmax(self.ant_preds,axis=-1),tf.int32)
            return labels

    @lazy_property
    def clause_labels(self):
        with tf.name_scope('clause_labels') :
            labels = tf.cast(tf.argmax(self.clause_preds,axis=-1),tf.int32)
            return labels

    @lazy_property
    def chunk_labels(self):
        with tf.name_scope('chunk_labels') :
            labels = tf.cast(tf.argmax(self.chunk_preds,axis=-1),tf.int32)
            return labels

    @lazy_property
    def ner_labels(self):
        with tf.name_scope('ner_labels') :
            labels = tf.cast(tf.argmax(self.ner_preds,axis=-1),tf.int32)
            return labels

    @lazy_property
    def com_labels(self):
        with tf.name_scope('com_labels') :
            labels = tf.cast(tf.argmax(self.com_preds,axis=-1),tf.int32)
            return labels

    @lazy_property
    def ant_cost(self):
        with tf.name_scope('ant_cost') :
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.ant_preds,labels=self.targets)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            osses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses)
            return cost

    @lazy_property
    def clause_cost(self):
        with tf.name_scope('clause_cost') :
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.clause_preds,labels=self.targets)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            losses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses)
            return cost

    @lazy_property
    def chunk_cost(self):
        with tf.name_scope('chunk_cost') :
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.chunk_preds,labels=self.targets)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            losses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses/self._conf.batch_size)
            cost = tf.reduce_mean(losses)
            return cost

    @lazy_property
    def ner_cost(self):
        with tf.name_scope('ner_cost') :
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.ner_preds,labels=self.targets)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            losses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses/self._conf.batch_size)
            cost = tf.reduce_mean(losses)
            return cost
        
    @lazy_property
    def com_cost(self):
        with tf.name_scope('com_cost') :
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.com_preds,labels=self.targets)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            losses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses)
            return cost

    @lazy_property
    def ant_optimize(self):
        with tf.name_scope('ant_optimize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate).minimize(self.ant_cost)
            return optimizer

    @lazy_property
    def clause_optimize(self):
        with tf.name_scope('clause_optimize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate).minimize(self.clause_cost)
            return optimizer
    
    @lazy_property
    def chunk_optimize(self):
        with tf.name_scope('chunk_optimize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate).minimize(self.chunk_cost)
            return optimizer

    @lazy_property
    def ner_optimize(self):
        with tf.name_scope('ner_optimize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate).minimize(self.ner_cost)
            return optimizer

    @lazy_property
    def com_optimize(self):
        with tf.name_scope('com_optimize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate).minimize(self.com_cost)
            return optimizer

    @property
    def _perplexity(self):
        perp = tf.pow(2.,-self.ant_cost)
        return perp

    @lazy_property
    def summary(self):
        tf.summary.scalar('Ant cost',self.ant_cost)
        tf.summary.scalar('Ant perplexity',self._perplexity)
        merged = tf.summary.merge_all()
        return merged

    def full_layer(self,input_,out_channels,name):
        with tf.name_scope(name) :
            fw_input,bw_input = input_
            fw_weight,fw_bias = self._weight_and_bias(self._conf.num_hidden,out_channels,layer_name='fw'+name+'_')
            bw_weight,bw_bias = self._weight_and_bias(self._conf.num_hidden,out_channels,layer_name='bw'+name+'_')
            fw_input = tf.reshape(fw_input,[-1,self._conf.num_hidden],name="forward_reshape")
            bw_input = tf.reshape(bw_input,[-1,self._conf.num_hidden],name="backward_reshape")
            output = ((tf.matmul(fw_input,fw_weight) + fw_bias) + (tf.matmul(bw_input,bw_weight) + bw_bias))
            output = tf.reshape(output,[-1,self._conf.max_len,out_channels])
            return output

    @staticmethod
    def _weight_and_bias(in_channels,out_channels,layer_name='layer_'):
        bias = tf.constant(0,shape=[out_channels],dtype=tf.float32)
        bias = tf.Variable(bias,name=layer_name+'B')
        weight = tf.get_variable(layer_name+"W",shape=[in_channels,out_channels],initializer=tf.truncated_normal_initializer(False))
        return weight,bias
