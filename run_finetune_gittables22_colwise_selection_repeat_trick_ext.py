import os
import subprocess
import time
import pickle
from multiprocessing import Process
from multiprocessing import Semaphore

'''run finetuning and evaluation on original datasets'''
# task = 'turl-re'
task = 'gt-semtab22-dbpedia-all0'
# task = 'turl'
ml = 128  # 32
bs = 16 # 16
n_epochs = 80
# n_epochs = 10
base_model = 'bert-base-uncased'
# base_model = 'distilbert-base-uncased'
cl_tag = "wikitables/simclr/bert_None_10_32_256_5e-05_sample_row4,sample_row4_tfidf_entity_column_0.05_0_last.pt"
ckpt_path = "/data/zhihao/TU/Watchog/model/"
dropout_prob = 0.1
from_scratch = True
# from_scratch = True # True means using Huggingface's pre-trained language model's checkpoint
eval_test = True
colpair = False
gpus = '0'

max_num_col = 2
comment = "max-unlabeled@{}".format(max_num_col)


small_tag = 'semi1'
ml = 64  # 32
gpus = '0'
pool = 'v0'
rand = False
use_token_type_ids = False
ctype = "v1.2"
target_num_col = 4
repeat = 1
gate_version = 'v0.2'
external_table_embedding = True
ema_decay = 0.99
# grad_clip = 5.0
    # parser.add_argument("--grad_clip", type=float, default=None)
    # parser.add_argument("--ema_decay", type=float, default=None)
for max_num_col in [8]:
    for tau in [0.1]:
        for gate_version in ['v0.2']:
            comment = "Repeat@{}-external@{}-ema_decay@{}-pool@{}-context@{}-max_num_col@{}-target_num_col@{}-tau@{}-gate@{}".format(repeat, external_table_embedding, ema_decay, pool, ctype, max_num_col, target_num_col, tau, gate_version)
            for task in ['gt-semtab22-dbpedia-all0']:
                cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_colwise_selection_repeat.py --wandb True  \
                            --shortcut_name {} --task {} --repeat {} --external_table_embedding {} --ema_decay {} --max_length {} --max_num_col {} --context_encoding_type {} --pool_version {} --batch_size {} --use_token_type_ids {} --epoch {} \
                            --tau {} --target_num_col {}  --gate_version {} \
                            --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
                    gpus, base_model, task, repeat, external_table_embedding, ema_decay, ml, max_num_col, ctype, pool, bs, use_token_type_ids, n_epochs, 
                    tau, target_num_col, gate_version, 
                    dropout_prob, ckpt_path, cl_tag, small_tag, comment,
                    '--colpair' if colpair else '',
                    '--from_scratch' if from_scratch else '',        
                    '--eval_test' if eval_test else ''
                )   
                # os.system('{} & '.format(cmd))
                subprocess.run(cmd, shell=True, check=True)

