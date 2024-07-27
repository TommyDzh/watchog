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
ml = 32  # 32
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
small_tag = 'semi'
max_unlabeled = 2
comment = "max-unlabeled@{}".format(max_unlabeled)

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
# for small_tag, gpus in zip(['blank', 'comma'], ['0', '2']):
# for task in [ 'gt-semtab22-dbpedia-all1','gt-semtab22-dbpedia-all0', 'gt-semtab22-dbpedia-all2', 'gt-semtab22-dbpedia-all3', 'gt-semtab22-dbpedia-all4']:
#     cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft.py \
#                 --shortcut_name {} --task {} --max_length {} --batch_size {} --epoch {} \
#                 --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
#         gpus, base_model, task, ml, bs, n_epochs, dropout_prob,
#         ckpt_path, cl_tag, small_tag, comment,
#         '--colpair' if colpair else '',
#         '--from_scratch' if True else '',        
#         '--eval_test' if eval_test else ''
#     )   
#     # os.system('{} & '.format(cmd))
#     subprocess.run(cmd, shell=True, check=True)
# max_unlabeled = 2
# pool = 'v1'
# comment = f"pool@{pool}-max-unlabeled@{max_unlabeled}"
# for task in ['sato0']:
#     cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft.py \
#                 --shortcut_name {} --task {} --max_length {} --max_unlabeled {} --pool_version {} --batch_size {} --epoch {} \
#                 --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
#         gpus, base_model, task, ml, max_unlabeled, pool, bs, n_epochs, dropout_prob,
#         ckpt_path, cl_tag, small_tag, comment,
#         '--colpair' if colpair else '',
#         '--from_scratch' if from_scratch else '',        
#         '--eval_test' if eval_test else ''
#     )   
#     # os.system('{} & '.format(cmd))
#     subprocess.run(cmd, shell=True, check=True)


    
# max_unlabeled = 8
# gpus = '1'
# pool = 'v0'
# rand = True
# comment = f"rand_pool@{pool}-max-unlabeled@{max_unlabeled}"
# for task in [ 'gt-semtab22-dbpedia-all0', 'gt-semtab22-dbpedia-all1']:
#     cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft.py \
#                 --shortcut_name {} --task {} --max_length {} --max_unlabeled {} --pool_version {} --random_sample {} --batch_size {} --epoch {} \
#                 --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
#         gpus, base_model, task, ml, max_unlabeled, pool, rand, bs, n_epochs, dropout_prob,
#         ckpt_path, cl_tag, small_tag, comment,
#         '--colpair' if colpair else '',
#         '--from_scratch' if from_scratch else '',        
#         '--eval_test' if eval_test else ''
#     )   
#     # os.system('{} & '.format(cmd))
#     subprocess.run(cmd, shell=True, check=True)


max_unlabeled = 2
nf = 5
gpus = '0'
pool = 'v0'
rand = False
comment = f"filter@{nf}_pool@{pool}"
for task in [ 'sato0']:
    cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_filter.py --wandb True \
                --shortcut_name {} --task {} --max_length {} --max_unlabeled {} --pool_version {} --random_sample {} --batch_size {} --num_filter {} --epoch {} \
                --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
        gpus, base_model, task, ml, max_unlabeled, pool, rand, bs, nf, n_epochs, dropout_prob,
        ckpt_path, cl_tag, small_tag, comment,
        '--colpair' if colpair else '',
        '--from_scratch' if from_scratch else '',        
        '--eval_test' if eval_test else ''
    )   
    # os.system('{} & '.format(cmd))
    subprocess.run(cmd, shell=True, check=True)
    
    
max_unlabeled = 2
nf = 10
gpus = '0'
pool = 'v0'
rand = False
comment = f"filter@{nf}_pool@{pool}"
for task in [ 'sato0']:
    cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_filter.py --wandb True \
                --shortcut_name {} --task {} --max_length {} --max_unlabeled {} --pool_version {} --random_sample {} --batch_size {} --num_filter {} --epoch {} \
                --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
        gpus, base_model, task, ml, max_unlabeled, pool, rand, bs, nf, n_epochs, dropout_prob,
        ckpt_path, cl_tag, small_tag, comment,
        '--colpair' if colpair else '',
        '--from_scratch' if from_scratch else '',        
        '--eval_test' if eval_test else ''
    )   
    # os.system('{} & '.format(cmd))
    subprocess.run(cmd, shell=True, check=True)


max_unlabeled = 2
nf = 15
gpus = '0'
pool = 'v0'
rand = False
comment = f"filter@{nf}_pool@{pool}"
for task in [ 'sato0']:
    cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_filter.py --wandb True \
                --shortcut_name {} --task {} --max_length {} --max_unlabeled {} --pool_version {} --random_sample {} --batch_size {} --num_filter {} --epoch {} \
                --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
        gpus, base_model, task, ml, max_unlabeled, pool, rand, bs, nf, n_epochs, dropout_prob,
        ckpt_path, cl_tag, small_tag, comment,
        '--colpair' if colpair else '',
        '--from_scratch' if from_scratch else '',        
        '--eval_test' if eval_test else ''
    )   
    # os.system('{} & '.format(cmd))
    subprocess.run(cmd, shell=True, check=True)
    
    
max_unlabeled = 2
nf = 20
gpus = '0'
pool = 'v0'
rand = False
comment = f"filter@{nf}_pool@{pool}"
for task in [ 'sato0']:
    cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_filter.py --wandb True \
                --shortcut_name {} --task {} --max_length {} --max_unlabeled {} --pool_version {} --random_sample {} --batch_size {} --num_filter {} --epoch {} \
                --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
        gpus, base_model, task, ml, max_unlabeled, pool, rand, bs, nf, n_epochs, dropout_prob,
        ckpt_path, cl_tag, small_tag, comment,
        '--colpair' if colpair else '',
        '--from_scratch' if from_scratch else '',        
        '--eval_test' if eval_test else ''
    )   
    # os.system('{} & '.format(cmd))
    subprocess.run(cmd, shell=True, check=True)