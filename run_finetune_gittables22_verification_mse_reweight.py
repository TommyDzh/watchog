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
bs = 64 # 16
n_epochs = 200
# n_epochs = 10
base_model = 'bert-base-uncased'
# base_model = 'distilbert-base-uncased'
cl_tag = "wikitables/simclr/bert_None_10_32_256_5e-05_sample_row4,sample_row4_tfidf_entity_column_0.05_0_last.pt"
ckpt_path = "/data/zhihao/TU/Watchog/model/"

from_scratch = True
# from_scratch = True # True means using Huggingface's pre-trained language model's checkpoint
eval_test = True
colpair = False

small_tag = 'semi1'
warmup_ratio = 0.0


max_unlabeled = 8
gpus = '0'
pos_ratio = 0.5
comment = "AttnMask-max-unlabeled@{}".format(max_unlabeled)
dropout_prob = 0.0
norm = None # "batch_norm"
for pos_ratio in [0.2, 0.8]:
    for lr in [5e-5, 1e-4, 5e-4, 1e-3]:
        for task in ['gt-semtab22-dbpedia-all0']:
            comment = "MSE-Reweight-lr@{}-warmup@{}-dp@{}-norm@{}-pos@{}".format(lr, warmup_ratio, dropout_prob, norm, pos_ratio)
            cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_verifier_binary.py --wandb True \
                        --shortcut_name {} --loss MSE --warmup_ratio {} --norm {} --reweight True --task {} --pos_ratio {} --use_attention_mask True --max_length {} --lr {} --max_unlabeled {} --batch_size {} --epoch {} \
                        --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
                gpus, base_model, warmup_ratio, norm, task,  pos_ratio, ml, lr, max_unlabeled, bs, n_epochs, dropout_prob,
                ckpt_path, cl_tag, small_tag, comment,
                '--colpair' if colpair else '',
                '--from_scratch' if from_scratch else '',        
                '--eval_test' if eval_test else ''
            )   
            # os.system('{} & '.format(cmd))
            subprocess.run(cmd, shell=True, check=True)


        
