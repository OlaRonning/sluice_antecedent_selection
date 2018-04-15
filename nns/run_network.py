import tensorflow as tf
from load.utils import ant_seq
from load.split import split
import config as cf
import paths as paths
from metrics import f1_score as f1_score


def train_base_network(sess,saver,model,conf,log,dispatcher,data,targets,istrain,report=True,store_peak=False):
    if report:
        writer = tf.summary.FileWriter(log,sess.graph)
    peak_entity = 0
    peak_token = 0
    for epoch in range(conf.num_epochs):
        for i in range(dispatcher.main_task.train_len//conf.batch_size):
            batch = dispatcher.main_task.sample(conf.batch_size)
            sess.run(model.optimize,{data:batch.data,targets:batch.target,istrain:True})
        if report:
            summary = sess.run(model.summary,{data:batch.data,targets:batch.target,istrain:False})
            writer.add_summary(summary,epoch)
            _,_,scores = validation(sess,dispatcher,model,conf.batch_size,data,targets,istrain)
            if store_peak:
                if scores.entity >= peak_entity:
                    saver.save(sess,log+'/entity_peak.ckpt')
                    peak_entity = scores.entity
                if scores.token >= peak_token:
                    saver.save(sess,log+'/token_peak.ckpt')
                    peak_token = scores.token
    return


def train_sg_network(sess,saver,model,conf,log,dispatcher,data,targets,istrain,report=True,store_peak=False):
    if report:
        writer = tf.summary.FileWriter(log,sess.graph)
    peak_entity = 0
    peak_token = 0
    for epoch in range(conf.num_epochs):
        for i in range(dispatcher.main_task.train_len//conf.batch_size):
            for aux_key in dispatcher.aux_keys:
                batch = dispatcher[aux_key].sample(conf.batch_size)
                if aux_key == 'chunk':
                    sess.run(model.chunk_optimize,{data:batch.data,targets:batch.target,istrain:True})
                if aux_key == 'pos':
                    sess.run(model.pos_optimize,{data:batch.data,targets:batch.target,istrain:True})
                if aux_key == 'clause':
                    sess.run(model.clause_optimize,{data:batch.data,targets:batch.target,istrain:True})
                if aux_key == 'ner':
                    sess.run(model.ner_optimize,{data:batch.data,targets:batch.target,istrain:True})
            main_batch = dispatcher.main_task.sample(conf.batch_size)
            sess.run(model.ant_optimize,{data:main_batch.data,targets:main_batch.target,istrain:True})

        if report:
            summary = sess.run(model.summary,{data:main_batch.data,targets:main_batch.target,istrain:False})
            writer.add_summary(summary,epoch)
            _,_,scores = validation(sess,dispatcher,model,conf.batch_size,data,targets,istrain)
            if store_peak:
                if scores.entity >= peak_entity:
                    saver.save(sess,log+'/entity_peak.ckpt')
                    peak_entity = scores.entity
                if scores.token >= peak_token:
                    saver.save(sess,log+'/token_peak.ckpt')
                    peak_token = scores.token
    return

def train_rhs_network(sess,saver,model,conf,log,dispatcher,data,targets,istrain,report=True,store_peak=False):
    if report:
        writer = tf.summary.FileWriter(log,sess.graph)
    peak_entity = 0
    peak_token = 0
    ordering = ['pos','com','chunk','ccg']
    ordering = [aux for aux in ordering if aux in dispatcher.aux_keys]
    for epoch in range(conf.num_epochs):
        for aux in ordering:
            for _ in range(dispatcher[aux].train_len//conf.batch_size):
                batch = dispatcher[aux].sample(conf.batch_size)
                if aux == 'pos':
                    sess.run(model.pos_optimize,{data:batch.data,targets:batch.target,istrain:True})
                if aux == 'com':
                    sess.run(model.com_optimize,{data:batch.data,targets:batch.target,istrain:True})
                if aux == 'chunk':
                    sess.run(model.chunk_optimize,{data:batch.data,targets:batch.target,istrain:True})
                if aux == 'ccg':
                    sess.run(model.ccg_optimize,{data:batch.data,targets:batch.target,istrain:True})
        for _ in range(dispatcher.main_task.train_len//conf.batch_size):
            main_batch = dispatcher.main_task.sample(conf.batch_size)
            sess.run(model.ant_optimize,{data:main_batch.data,targets:main_batch.target,istrain:True})
        if report:
            summary = sess.run(model.summary,{data:main_batch.data,targets:main_batch.target,istrain:False})
            writer.add_summary(summary,epoch)
            _,_,scores = validation(sess,dispatcher,model,conf.batch_size,data,targets,istrain)
            if store_peak:
                if scores.entity >= peak_entity:
                    saver.save(sess,log+'/entity_peak.ckpt')
                    peak_entity = scores.entity
                if scores.token >= peak_token:
                    saver.save(sess,log+'/token_peak.ckpt')
                    peak_token = scores.token
    return


def validation(sess,dispatcher,model,batch_size,data,targets,istrain):
    def to_cls(l):
        classes = ['0','B-ANT','I-ANT']
        return list(map(lambda lbl:classes[lbl],l))

    act = []
    pred = []
    conf1 = cf.Config(null_cls='0',begin='B')
    try:
        model_labels = model.ant_labels
    except:
        model_labels = model.labels
    for batch in dispatcher.main_task.batch_iter(1):
        predicted_labels,ground_truth,lengths = sess.run(
                [model_labels,model.targets,model.length],
                {data:batch.data,targets:batch.target,istrain:False})
        lengths = lengths.tolist()
        predicted_labels = list(map(lambda p,l: to_cls(p[:l]),predicted_labels.tolist(),lengths))
        ground_truth = list(map(lambda p,l: to_cls(p[:l]),ground_truth.tolist(),lengths))
        pred.extend(predicted_labels)
        act.extend(ground_truth)
    scores = f1_score.F1_Score(act,pred,conf1)
    print("Entity score: {0.entity:3f} and Token score: {0.token:3f}".format(scores))
    return act,pred,scores
