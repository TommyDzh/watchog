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


max_unlabeled = 4
gpus = '1'
pos_ratio = 0.5
dropout_prob = 0.0
norm = "batch_norm"
context = None
veri_module = "ffn"
version = 5
test_version = None
# num_layers = 2
for pos_ratio in [0.2]:
    for lr in [ 1e-3, 1e-4, 5e-5]:
        for task in ['gt-semtab22-dbpedia-all0']: # 'gt-semtab22-dbpedia-all0'
            comment = "GT-mode@{}-context@{}-data@{}-lr@{}-warmup@{}-dp@{}-norm@{}-pos@{}".format(veri_module, context, version, lr, warmup_ratio, dropout_prob, norm,  pos_ratio)
            cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_verifier_binary_gt.py --wandb True \
                        --shortcut_name {}  --warmup_ratio {} --norm {}  --data_version {} --test_version {} --veri_module {} --context {} --reweight True --task {} --pos_ratio {} --use_attention_mask True --max_length {} --lr {} --max_unlabeled {} --batch_size {} --epoch {} \
                        --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
                gpus, base_model, warmup_ratio, norm, version,  test_version, veri_module, context, task,  pos_ratio, ml, lr, max_unlabeled, bs, n_epochs, dropout_prob,
                ckpt_path, cl_tag, small_tag, comment,
                '--colpair' if colpair else '',
                '--from_scratch' if from_scratch else '',        
                '--eval_test' if eval_test else ''
            )   
            # os.system('{} & '.format(cmd))
            subprocess.run(cmd, shell=True, check=True)


        
