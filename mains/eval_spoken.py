# EXTERNAL IMPORTS
import tensorflow as tf
import random
import numpy as np
from os.path import isdir
from argparse import ArgumentParser
from enum import Enum

# PROJECT IMPORTS
## MODEL
from nns.models import baseRNN
from nns.models import rhsRNN
from nns.models import sgRNN

##TRAINERS
from nns.run_network import train_base_network
from nns.run_network import train_sg_network
from nns.run_network import train_rhs_network

## DATA
from load.utils import ant_seq
from load.utils import conll
from load.utils import opensub

## METRICS
import metrics.label_writer as label_writer
import metrics.f1_score as f1_score
from nns.run_network import validation

## UTILS
import paths as paths
import config as cf

class Model(Enum):
    sg = 'sg'
    rhs = 'rhs'
    base = 'base'

    def __str__(self):
        return self.value

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-l',required=True, dest='log', help='path to pretrained model')
    parser.add_argument('-m', required=True, type=Model, dest='model', choices=list(Model), 
            help='model to evaluate on spoken sluices')
    args = parser.parse_args()
    if not isdir(args.log):
        print('{} not a dir'.format(args.log))
        exit(1)

    conf = cf.Config()
    if args.model == Model.base:
        conf.load(paths.CONFIG_BEFORE)
        rnn = baseRNN
        trainer = train_base_network
    elif args.model == Model.sg:
        conf.load(paths.CONFIG_SG_BEFORE)
        rnn = sgRNN
        trainer = train_sg_network
    else:
        conf.load(paths.CONFIG_RHS_BEFORE)
        rnn = rhsRNN
        trainer = train_rhs_network


    tf.reset_default_graph()
    dispatch = ant_seq(before=-1,prefix=True)
    emb_dispatch = opensub(embedded=True,prefix=False,max_len=dispatch.max_len)
    root_dispatch = opensub(embedded=False,prefix=True,max_len=dispatch.max_len)
    conf.max_len = dispatch.max_len

    length = dispatch.max_len
    data = tf.placeholder(tf.int32,[None,length])
    targets = tf.placeholder(tf.int32,[None,length])
    istrain = tf.placeholder(tf.bool,())

    embeddings = np.load(paths.EMBEDDINGS)
    model = rnn.network(data,targets,istrain,embeddings,conf)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,args.log+'/entity_peak.ckpt')
        print('Embedded')
        acts,preds,_ = validation(sess,emb_dispatch,model,conf.batch_size,data,targets,istrain)
        label_writer.write_conll_style_preds(acts,preds,args.log+'/emb_predictions.txt')

        print('Root')
        acts,preds,_ = validation(sess,root_dispatch,model,conf.batch_size,data,targets,istrain)
        label_writer.write_conll_style_preds(acts,preds,args.log+'/root_predictions.txt')
