import os
import paths as paths


def write_crf_style_preds(targets,predictions,path):
    classes = ['0','B-ANT','I-ANT']
    if os.path.isfile(path):
        write_file = open(path,'a')
        print('',file=write_file)
    else:
        write_file = open(path,'w')
    itr = zip(targets,predictions)
    try:
        t,p = next(itr)
    except StopIteration:
        return 
    while(True):
        try:
            for gr,pr in zip(t,p):
               print('{0}\t{1}'.format(classes[gr],classes[pr]),file=write_file)
            t,p = next(itr)
            print('',file=write_file)
        except StopIteration:
            break
    write_file.close()

def write_conll_style_preds(targets,predictions,path):
    if os.path.isfile(path):
        write_file = open(path,'a')
        print('',file=write_file)
    else:
        write_file = open(path,'w')
    itr = zip(targets,predictions)
    try:
        t,p = next(itr)
    except StopIteration:
        return 
    while(True):
        try:
            for gr,pr in zip(t,p):
               print('{0}\t{1}'.format(gr,pr),file=write_file)
            t,p = next(itr)
            print('',file=write_file)
        except StopIteration:
            break
    write_file.close()

def read_crf_style_preds(path):
    """read_crf_style_preds

    Params:
         path

    Returns:
    """
    with open(path,'r') as read_file:
        entire_file = read_file.read()
    segments = entire_file.strip().split('\n\n')
    labels_segments = list(map(lambda s: s.split('\n'),segments))
    data = [[s.split('\t')  for s in seg]for seg in labels_segments]
    act = [[s[0]  for s in seg]for seg in data]
    pred = [[s[-1]  for s in seg]for seg in data]

    assert(len(act) == len(pred))
    return act,pred
