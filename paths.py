import os
import datetime

########## DIRECTORIES ###########
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR,'data')
CONFIG_DIR = os.path.join(BASE_DIR,'configs')
AUX_DIR = os.path.join(DATA_DIR,'aux')
ESC_DIR = os.path.join(DATA_DIR,'esc')
OPENSUB_DIR = os.path.join(DATA_DIR,'opensub')
GLOVE_DIR = os.path.join(DATA_DIR,'glove')
TF_BOARD_DIR = os.path.join(BASE_DIR,'tensorboard')

##### NEWSWIRE ######
ESC_TRAIN = os.path.join(ESC_DIR,'train.jsons')
ESC_VAL = os.path.join(ESC_DIR,'val.jsons')
ESC_TEST = os.path.join(ESC_DIR,'test.jsons')

###### DIALOGUE ########
OPENSUB_EMBEDDED_TEST = os.path.join(OPENSUB_DIR,'embedded_test.jsons')
OPENSUB_ROOT_TEST = os.path.join(OPENSUB_DIR,'root_test.jsons')

###### AUXILIARY #######
CHUNK_TRAIN = os.path.join(AUX_DIR,'eng_chunking_train.conll')
CHUNK_DEV = os.path.join(AUX_DIR,'eng_chunking_dev.conll')
CHUNK_TEST = os.path.join(AUX_DIR,'eng_chunking_test.conll')
COM_TRAIN = os.path.join(AUX_DIR,'eng_com_train.conll')
COM_DEV = os.path.join(AUX_DIR,'eng_com_dev.conll')
COM_TEST = os.path.join(AUX_DIR,'eng_com_test.conll')
NER_TRAIN = os.path.join(AUX_DIR,'eng_ner_train.conll')
NER_DEV = os.path.join(AUX_DIR,'eng_ner_dev.conll')
NER_TEST = os.path.join(AUX_DIR,'eng_ner_test.conll')
POS_TRAIN = os.path.join(AUX_DIR,'eng_pos_train.conll')
POS_TEST = os.path.join(AUX_DIR,'eng_pos_test.conll')
CLAUSE_ALL = os.path.join(AUX_DIR,'clause_all.conll')
CCG_TRAIN = os.path.join(AUX_DIR,'eng_ccg_train.conll')
CCG_DEV = os.path.join(AUX_DIR,'eng_ccg_dev.conll')
CCG_TEST = os.path.join(AUX_DIR,'eng_ccg_test.conll')

######## GLOVE WORD EMBEDDINGS #########
EMBEDDINGS = os.path.join(GLOVE_DIR,'embeddings.npy')
VOCAB = os.path.join(GLOVE_DIR,'ids.npy')

######## CONFIG FILES ########
CONFIG_BEFORE = os.path.join(CONFIG_DIR,'single-task.json')
CONFIG_SG_BEFORE = os.path.join(CONFIG_DIR,'sg.json')
CONFIG_RHS_BEFORE = os.path.join(CONFIG_DIR,'rs.json')

######## UTILITY #########
TF_LOG_DIR = os.path.join(TF_BOARD_DIR,datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
