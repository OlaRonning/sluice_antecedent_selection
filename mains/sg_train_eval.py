from nns.models import sgRNN
from nns.run_network import train_sg_network as trainer
from nns.run_network import validation
from load.utils import ant_seq
from load.utils import conll
import metrics.label_writer as label_writer
import paths as paths
import config as cf
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    tf.reset_default_graph()
    conf = cf.Config()
    conf.load(paths.CONFIG_SG_BEFORE)
    dispatch = ant_seq(before=-1, prefix=True)
    test_dispatch = ant_seq(test=True,before=-1,prefix=True,max_len=dispatch.max_len)
    conll(dispatch,dataset='chunk')
    conll(dispatch,dataset='com')
    conf.max_len = dispatch.max_len

    length = dispatch.max_len
    data = tf.placeholder(tf.int32,[None,length])
    targets = tf.placeholder(tf.int32,[None,length])
    istrain = tf.placeholder(tf.bool,())

    embeddings = np.load(paths.EMBEDDINGS)
    model = sgRNN.network(data,targets,istrain,embeddings,conf)
    log = paths.TF_LOG_DIR

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        trainer(sess,saver,model,conf,log,dispatch,data,targets,istrain,store_peak=True)
        saver.restore(sess,log+'/entity_peak.ckpt')
        acts,preds,_ = validation(sess,test_dispatch,model,conf.batch_size,data,targets,istrain)
    label_writer.write_conll_style_preds(acts,preds,log+'/predictions.txt')
    conf.save(log+'/conf.txt')
