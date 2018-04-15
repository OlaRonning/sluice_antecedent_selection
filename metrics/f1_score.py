import functools
import operator as opt

def lazy_property(func):
    """lazy_property

    Params:
         func

    Returns:
    """
    attr = '_' + func.__name__

    @property
    @functools.wraps(func)
    def wrapper(self):
        """wrapper"""
        if not hasattr(self,attr):
            setattr(self,attr,func(self))
        return(getattr(self,attr))
    return wrapper

class F1_Score(object):
    def __init__(self,targets, predictions, config):
        """__init__

        Params:
                 targets
                 predictions
                 config

        Returns:
        """
        self.targets = targets
        self.predictions = predictions
        self.conf = config

    @lazy_property
    def micro(self):
        """micro"""
        return
    @lazy_property
    def macro(self):
        """macro"""
        return

    @property
    def entity(self):
        """entity"""
        def ctp(act,pred):
            """ctp

            Params:
                         act
                         pred

            Returns:
            """
            count = {}
            seq = [self.conf.null_cls,0]
            for act,pred in zip(act,pred):
                if act[:1] == self.conf.begin :
                    if seq[1] in count:
                        count[seq[0]] += seq[1]
                    else:
                        count[seq[0]] = seq[1]
                    seq = [act,1]
                seq[1] = seq[1] and act == pred
                if act == self.conf.null_cls:
                    if seq[0] in count:
                        count[seq[0]] += seq[1]
                    else:
                        count[seq[0]] = seq[1]
                    seq[1] = 0
            if seq[0] in count:
                count[seq[0]] += seq[1]
            else:
                count[seq[0]] = seq[1]
            return sum(count.values())

        self.tp = list(map(ctp,self.targets,self.predictions))
        # does not handle multiple sequence classes 

        cfp = functools.partial(self._tally,cf=lambda a,p: p[:1] == self.conf.begin)
        self.fp = list(map(cfp,self.targets,self.predictions))
        self.fp = list(map(opt.sub,self.fp,self.tp))

        cp = functools.partial(self._tally,cf=lambda a,p: a[:1] == self.conf.begin)
        self.p = list(map(cp,self.targets,self.predictions))
        return self.f1_score

    @property
    def token(self):
        """token"""
        """ token level label agnostic f1 score
            fx. 0   0   B-ANT I-ANT
                0 B-ANT I-ANT I-ANT
            tp: 2 fp: 0 p:3 pric = 1, recall = 2/3
            f1 = 0.8
        """
        def collapse_clss(clss):
            """collapse_clss"""
            return list(map(lambda cls: cls != self.conf.null_cls,clss))
        def id_fp(a,b):
            """id_fp

            Params:
                         a
                         b

            Returns:
            """
            """ identifies false postives
                Note: not implies has same thruth table as fp
                act pred | fp
                 0    0     0
                 0    1     1
                 1    0     0
                 1    1     0
            """
            return bool(not a and b)

        acts = list(map(collapse_clss,self.targets))
        preds = list(map(collapse_clss,self.predictions))

        ctp = functools.partial(self._tally,cf=lambda a,p: a and p)
        self.tp = list(map(ctp,acts,preds))

        cfp = functools.partial(self._tally,cf=id_fp)
        self.fp = list(map(cfp,acts,preds))

        cp = functools.partial(self._tally,cf=lambda a,p: a)
        self.p  = list(map(cp,acts,preds))
        return self.f1_score
        
    def _tally(self,act,pred,cf):
        """_tally

        Params:
                 act
                 pred
                 cf

        Returns:
        """
        """ Tallies according to counting function
            Arguments:
                cf: count function
                    true when elements should be counted.
                act: ground truth labels
                pred: predicted labels
            Returns:
                count of true cf returns
        """
        tally = sum(map(cf,act,pred))
        return tally

    @property
    def precision(self):
        """precision"""
        def prec(tp,fp):
            """prec

            Params:
                         tp
                         fp

            Returns:
            """
            if not tp and not fp:
                return 0
            return tp/(tp+fp)
        return list(map(prec,self.tp,self.fp))

    @property
    def recall(self):
        """recall"""
        def recll(tp,p):
            """recll

            Params:
                         tp
                         p

            Returns:
            """
            if not tp and not p:
                return 0
            return tp/p
        return list(map(recll,self.tp,self.p))

    
    @property
    def f1_score(self):
        """f1_score"""
        def f1(prec,recll):
            """f1

            Params:
                         prec
                         recll

            Returns:
            """
            if not prec and not recll:
                return 0
            return 2*(prec*recll)/(prec+recll)
        f1_scores = list(map(f1,self.precision,self.recall))
        return sum(f1_scores)/float(len(list(f1_scores)))
