import os
import subprocess
import time
import pickle
from multiprocessing import Process
from multiprocessing import Semaphore

'''run finetuning and evaluation on original datasets'''
# task = 'turl-re'
task = 'sato0'
# task = 'turl'
ml = 32  # 32
bs = 16 # 16
n_epochs = 30
# n_epochs = 10
base_model = 'bert-base-uncased'
# base_model = 'distilbert-base-uncased'
cl_tag = "wikitables/simclr/bert_100000_10_32_256_5e-05_sample_row4,sample_row4_tfidf_entity_column_0.05_0_last.pt"
ckpt_path = "/data/zhihao/TU/Watchog/model/"
dropout_prob = 0.1
from_scratch = False
# from_scratch = True # True means using Huggingface's pre-trained language model's checkpoint
eval_test = True
colpair = False
gpus = '0'
small_tag = ''
comment = "None"

# cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft.py \
#             --shortcut_name {} --task {} --max_length {} --batch_size {} --epoch {} \
#             --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
#     gpus, base_model, task, ml, bs, n_epochs, dropout_prob,
#     ckpt_path, cl_tag, small_tag, comment,
#     '--colpair' if colpair else '',
#     '--from_scratch' if from_scratch else '',        
#     '--eval_test' if eval_test else ''
# )


# os.system('{} & '.format(cmd))
for ml, gpus in zip([32], ['0']):
    cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft.py \
                --shortcut_name {} --task {} --max_length {} --batch_size {} --epoch {} \
                --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
        gpus, base_model, task, ml, bs, n_epochs, dropout_prob,
        ckpt_path, cl_tag, small_tag, comment,
        '--colpair' if colpair else '',
        '--from_scratch' if True else '',        
        '--eval_test' if eval_test else ''
    )   
    os.system('{} & '.format(cmd))