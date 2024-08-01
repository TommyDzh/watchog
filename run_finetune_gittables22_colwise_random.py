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
n_epochs = 50
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
small_tag = 'semi'
max_num_col = 2
comment = "max-unlabeled@{}".format(max_num_col)

small_tag = 'semi1'
ml = 64  # 32
gpus = '0'
pool = 'v0'
rand = True

for max_num_col in [8]:
    comment = "rand_max_num_col@{}".format(max_num_col)
    for task in ['gt-semtab22-dbpedia-all0']:
        cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_colwise.py --wandb True --random_sample True \
                    --shortcut_name {} --task {} --max_length {} --max_num_col {} --batch_size {} --epoch {} \
                    --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
            gpus, base_model, task, ml, max_num_col, bs, n_epochs, dropout_prob,
            ckpt_path, cl_tag, small_tag, comment,
            '--colpair' if colpair else '',
            '--from_scratch' if from_scratch else '',        
            '--eval_test' if eval_test else ''
        )   
        # os.system('{} & '.format(cmd))
        subprocess.run(cmd, shell=True, check=True)


    