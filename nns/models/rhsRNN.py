import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import functools

from ..custom_tf.zoneout import ZoneoutWrapper
from ..custom_tf.rnn import bidirectional_input_dynamic_rnn

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

        ## POS
        self.pos_rnn
        self.pos_embs
        self.pos_probs
        self.pos_cost
        self.pos_optimize

        ## CHUNK
        self.chunk_rnn
        self.chunk_embs
        self.chunk_probs
        self.chunk_cost
        self.chunk_optimize

        ## CCG
        self.ccg_rnn
        self.ccg_embs
        self.ccg_probs
        self.ccg_cost
        self.ccg_optimize

        ## ANT
        self.ant_rnn
        self.ant_embs
        self.ant_probs
        self.ant_cost
        self.ant_optimize

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
    def pos_rnn(self):
        with tf.name_scope('pos_rnn') :
            rnn_out = self.rnn(self.lookup,self.lookup,'pos_rnn')
            return rnn_out

    @lazy_property
    def pos_embs(self):
        with tf.name_scope('pos_embs') :
            output,fw_embs,bw_embs = self.full_layer(self.pos_rnn,self._conf.num_pos_classes,'pos_embs')
            return output,fw_embs,bw_embs

    @lazy_property
    def pos_probs(self):
        with tf.name_scope('pos_probs'):
            output,_,_ = self.pos_embs
            probs = self.softmax(output,self._conf.num_pos_classes,'pos_probs')
            return probs

    @lazy_property
    def pos_cost(self):
        with tf.name_scope('pos_cost') :
            targets = tf.one_hot(self.targets,self._conf.num_pos_classes,axis=-1)
            losses = targets * tf.log(self.pos_probs)
            losses = - tf.reduce_sum(losses,-1)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            losses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses)
            return cost

    @lazy_property
    def chunk_rnn(self):
        with tf.name_scope('chunk_rnn') :
            fw_rnn_out,bw_rnn_out = self.pos_rnn
            _,fw_embs,bw_embs = self.pos_embs
            probs = self.pos_probs
            probs = tf.reshape(probs,[-1,self._conf.num_pos_classes])
            fw_pos_embs = tf.matmul(probs,fw_embs)
            bw_pos_embs = tf.matmul(probs,bw_embs)
            fw_pos_embs = tf.reshape(fw_pos_embs,[-1,self._conf.max_len,self._conf.num_hidden])
            bw_pos_embs = tf.reshape(bw_pos_embs,[-1,self._conf.max_len,self._conf.num_hidden])
            fw_input = tf.concat([self.lookup,fw_rnn_out,fw_pos_embs],-1)
            bw_input = tf.concat([self.lookup,bw_rnn_out,bw_pos_embs],-1)
            rnn_out = self.rnn(fw_input,bw_input,'chunk_rnn')
            return rnn_out

    @lazy_property
    def chunk_embs(self):
        with tf.name_scope('chunk_embs') :
            output,fw_embs,bw_embs = self.full_layer(self.chunk_rnn,self._conf.num_chunk_classes,'chunk_pred')
            return output,fw_embs,bw_embs

    @lazy_property
    def chunk_probs(self):
        with tf.name_scope('chunk_probs'):
            output,_,_ = self.chunk_embs
            probs = self.softmax(output,self._conf.num_chunk_classes,'chunk_probs')
            return probs

    @lazy_property
    def chunk_cost(self):
        with tf.name_scope('chunk_cost') :
            targets = tf.one_hot(self.targets,self._conf.num_chunk_classes,axis=-1)
            losses = targets * tf.log(self.chunk_probs)
            losses = - tf.reduce_sum(losses,-1)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            losses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses)
            return cost

    @lazy_property
    def ccg_rnn(self):
        with tf.name_scope('ccg_rnn') :
            _,fw_embs,bw_embs = self.pos_embs
            probs = self.pos_probs
            probs = tf.reshape(probs,[-1,self._conf.num_pos_classes])
            fw_pos_embs = tf.matmul(probs,fw_embs)
            bw_pos_embs = tf.matmul(probs,bw_embs)
            fw_pos_embs = tf.reshape(fw_pos_embs,[-1,self._conf.max_len,self._conf.num_hidden])
            bw_pos_embs = tf.reshape(bw_pos_embs,[-1,self._conf.max_len,self._conf.num_hidden])

            fw_chunk_rnn_out,bw_chunk_rnn_out = self.chunk_rnn
            _,fw_embs,bw_embs = self.chunk_embs
            probs = self.chunk_probs
            probs = tf.reshape(probs,[-1,self._conf.num_chunk_classes])
            fw_chunk_embs = tf.matmul(probs,fw_embs)
            bw_chunk_embs = tf.matmul(probs,bw_embs)
            fw_chunk_embs = tf.reshape(fw_chunk_embs,[-1,self._conf.max_len,self._conf.num_hidden])
            bw_chunk_embs = tf.reshape(bw_chunk_embs,[-1,self._conf.max_len,self._conf.num_hidden])
            fw_input = tf.concat([self.lookup,fw_chunk_rnn_out,fw_pos_embs,fw_chunk_embs],-1)
            bw_input = tf.concat([self.lookup,bw_chunk_rnn_out,bw_pos_embs,bw_chunk_embs],-1)
            rnn_out = self.rnn(fw_input,bw_input,'ccg_rnn')
            return rnn_out

    @lazy_property
    def ccg_embs(self):
        with tf.name_scope('ccg_embs') :
            output,fw_embs,bw_embs = self.full_layer(self.ccg_rnn,self._conf.num_ccg_classes,'ccg_pred')
            return output,fw_embs,bw_embs

    @lazy_property
    def ccg_probs(self):
        with tf.name_scope('ccg_probs'):
            output,_,_ = self.ccg_embs
            probs = self.softmax(output,self._conf.num_ccg_classes,'ccg_probs')
            return probs

    @lazy_property
    def ccg_cost(self):
        with tf.name_scope('ccg_cost') :
            targets = tf.one_hot(self.targets,self._conf.num_ccg_classes,axis=-1)
            losses = targets * tf.log(self.ccg_probs)
            losses = - tf.reduce_sum(losses,-1)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            losses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses)
            return cost


    @lazy_property
    def ant_rnn(self):
        with tf.name_scope('ant_rnn') :

            _,fw_embs,bw_embs = self.pos_embs
            probs = self.pos_probs
            probs = tf.reshape(probs,[-1,self._conf.num_pos_classes])
            fw_pos_embs = tf.matmul(probs,fw_embs)
            bw_pos_embs = tf.matmul(probs,bw_embs)
            fw_pos_embs = tf.reshape(fw_pos_embs,[-1,self._conf.max_len,self._conf.num_hidden])
            bw_pos_embs = tf.reshape(bw_pos_embs,[-1,self._conf.max_len,self._conf.num_hidden])

            _,fw_embs,bw_embs = self.chunk_embs
            probs = self.chunk_probs
            probs = tf.reshape(probs,[-1,self._conf.num_chunk_classes])
            fw_chunk_embs = tf.matmul(probs,fw_embs)
            bw_chunk_embs = tf.matmul(probs,bw_embs)
            fw_chunk_embs = tf.reshape(fw_chunk_embs,[-1,self._conf.max_len,self._conf.num_hidden])
            bw_chunk_embs = tf.reshape(bw_chunk_embs,[-1,self._conf.max_len,self._conf.num_hidden])

            fw_ccg_rnn_out,bw_ccg_rnn_out = self.ccg_rnn
            _,fw_embs,bw_embs = self.ccg_embs
            probs = self.ccg_probs
            probs = tf.reshape(probs,[-1,self._conf.num_ccg_classes])
            fw_ccg_embs = tf.matmul(probs,fw_embs)
            bw_ccg_embs = tf.matmul(probs,bw_embs)
            fw_ccg_embs = tf.reshape(fw_ccg_embs,[-1,self._conf.max_len,self._conf.num_hidden])
            bw_ccg_embs = tf.reshape(bw_ccg_embs,[-1,self._conf.max_len,self._conf.num_hidden])

            fw_input = tf.concat([self.lookup,fw_ccg_rnn_out,fw_pos_embs,fw_chunk_embs,fw_ccg_embs],-1)
            bw_input = tf.concat([self.lookup,bw_ccg_rnn_out,bw_pos_embs,bw_chunk_embs,bw_ccg_embs],-1)
            rnn_out = self.rnn(fw_input,bw_input,'ant_rnn')
            return rnn_out

    @lazy_property
    def ant_embs(self):
        with tf.name_scope('ant_embs') :
            output,fw_embs,bw_embs = self.full_layer(self.ant_rnn,self._conf.num_ant_classes,'ant_pred')
            return output,fw_embs,bw_embs

    @lazy_property
    def ant_probs(self):
        with tf.name_scope('ant_probs'):
            output,_,_ = self.ant_embs
            probs = self.softmax(output,self._conf.num_ant_classes,'ant_probs')
            return probs

    @lazy_property
    def ant_cost(self):
        with tf.name_scope('ant_cost') :
            targets = tf.one_hot(self.targets,self._conf.num_ant_classes,axis=-1)
            losses = targets * tf.log(self.ant_probs)
            losses = - tf.reduce_sum(losses,-1)
            mask = tf.sequence_mask(self.length,self._conf.max_len)
            losses = tf.boolean_mask(losses,mask)
            cost = tf.reduce_mean(losses)
            return cost

    @lazy_property
    def pos_labels(self):
        with tf.name_scope('pos_labels') :
            labels = tf.cast(tf.argmax(self.pos_probs,axis=-1),tf.int32)
            return labels

    @lazy_property
    def chunk_labels(self):
        with tf.name_scope('chunk_labels') :
            labels = tf.cast(tf.argmax(self.chunk_probs,axis=-1),tf.int32)
            return labels

    @lazy_property
    def ant_labels(self):
        with tf.name_scope('ant_labels') :
            labels = tf.cast(tf.argmax(self.ant_probs,axis=-1),tf.int32)
            return labels

    @lazy_property
    def pos_optimize(self):
        with tf.name_scope('pos_optimize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate).minimize(self.pos_cost)
            return optimizer

    @lazy_property
    def chunk_optimize(self):
        with tf.name_scope('chunk_optimize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate).minimize(self.chunk_cost)
            return optimizer

    @lazy_property
    def ccg_optimize(self):
        with tf.name_scope('ccg_optimize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate).minimize(self.ccg_cost)
            return optimizer

    @lazy_property
    def ant_optimize(self):
        with tf.name_scope('ant_optimize'):
            optimizer = tf.train.AdamOptimizer(self._conf.learn_rate).minimize(self.ant_cost)
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

    def rnn(self,fw_input,bw_input,name):
        with tf.name_scope(name):
            fw_cell = ZoneoutWrapper(rnn.LSTMCell(self._conf.num_hidden),
                zoneout_prob=(self._conf.z_prob_cell,self._conf.z_prob_states),is_training=self.istrain) 
            bw_cell = ZoneoutWrapper(rnn.LSTMCell(self._conf.num_hidden),
                zoneout_prob=(self._conf.z_prob_cell,self._conf.z_prob_states),is_training=self.istrain) 
            output,_ = bidirectional_input_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    fw_input,
                    bw_input,
                    dtype=tf.float32,
                    sequence_length=self.length,
                    scope=name
            )
            return output

    def full_layer(self,input_,out_channels,name):
        with tf.name_scope(name) :
            fw_input,bw_input = input_
            fw_weight,fw_bias = self._weight_and_bias(self._conf.num_hidden,out_channels,layer_name='fw'+name+'_')
            bw_weight,bw_bias = self._weight_and_bias(self._conf.num_hidden,out_channels,layer_name='bw'+name+'_')
            fw_input = tf.reshape(fw_input,[-1,self._conf.num_hidden],name="forward_reshape")
            bw_input = tf.reshape(bw_input,[-1,self._conf.num_hidden],name="backward_reshape")
            output = ((tf.matmul(fw_input,fw_weight) + fw_bias) + (tf.matmul(bw_input,bw_weight) + bw_bias))
            output = tf.reshape(output,[-1,self._conf.max_len,out_channels])
            return output,tf.transpose(fw_weight),tf.transpose(bw_weight)

    def softmax(self,input_,outer_dim,name):
        with tf.name_scope(name):
            input_ = tf.reshape(input_,[-1,outer_dim])
            output = tf.nn.softmax(input_ + tf.reduce_max(input_,-1,keep_dims=True))
            output = tf.reshape(output,[-1,self._conf.max_len,outer_dim])
            return output

    @staticmethod
    def _weight_and_bias(in_channels,out_channels,layer_name='layer_'):
        bias = tf.constant(0,shape=[out_channels],dtype=tf.float32)
        bias = tf.Variable(bias,name=layer_name+'B')
        weight = tf.get_variable(layer_name+"W",shape=[in_channels,out_channels],initializer=tf.truncated_normal_initializer(False))
        return weight,bias
