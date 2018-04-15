import itertools
import paths

def chain(*args):
    if len(args) == 1:
        chained_list = itertools.chain.from_iterable(*args)
    else:
        chained_list = itertools.chain(*args)
    return list(chained_list)

def reader(data_path=paths.OPENSUB_EMBEDDED_TEST,l2i=None):
    def ordered_set(l):
        seen = set()
        seen_add = seen.add
        return [x for x in l if not (x in seen or seen_add(x))]

    def label2int(lbl_dict):
        def lbl2int(lbl):
            try:
                idx = lbl_dict[lbl]
                return idx
            except KeyError:
                return 0
        return lbl2int

    if data_path is None:
        return None,None,None
    with open(data_path,'r') as raw_data:
        sents = []
        labels = []
        sent = []
        label = []
        for word_label in raw_data:
            word_label = word_label.strip()
            if word_label == '':
                labels.append(label)
                sents.append(sent)
                sent = []
                label = []
                continue
            word,lbl = word_label.split('\t')
            sent.append(word)
            label.append(lbl)

        if l2i is None:
            l2i = label2int({lbl: i for i,lbl in enumerate(ordered_set(chain(labels)))})
        labels = list(map(lambda l: list(map(l2i,l)),labels))
        return sents,labels,l2i
