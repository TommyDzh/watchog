import argparse
import json
import math
import os
import random
from time import time
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from collections import defaultdict

# import pytrec_eval
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from accelerate import Accelerator


torch.backends.cuda.matmul.allow_tf32 = True

from watchog.loss import listMLE

from watchog.dataset import (
    # collate_fn,
    TURLColTypeTablewiseDataset,
    TURLRelExtTablewiseDataset,
    SatoCVTablewiseDataset,
    ColPoplTablewiseDataset
)

from watchog.dataset import SOTABTablewiseIterateDataset, GittablesTablewiseIterateDataset, VerificationCompareDataset
from watchog.model import Verifier, BertMultiPairPooler, BertForMultiOutputClassification, BertForMultiOutputClassificationColPopl
from watchog.model import SupCLforTable, UnsupCLforTable, lm_mp
from watchog.utils import load_checkpoint, f1_score_multilabel, collate_fn_iter, veri_collate_fn, get_col_pred, ColPoplEvaluator
from watchog.utils import task_num_class_dict
from accelerate import DistributedDataParallelKwargs
import wandb

import itertools
from copy import deepcopy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def is_sublist(A, B):
    it = iter(B)
    return all(x in it for x in A)
def get_permutation(x):
    new = []
    x = x.tolist()
    if len(x) == 1:
        x = x[0]
    for k in x:
        if k not in new:
            new.append(k)
    return new





if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--model", type=str, default="Watchog")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--norm", type=str, default="batch_norm") 
    parser.add_argument("--max_list_length", type=int, default=10)
    parser.add_argument("--pos_ratio", type=float, default=0.5) 
    parser.add_argument("--reweight", type=bool, default=False) 
    parser.add_argument("--veri_module", type=str, default="ffn") 
    parser.add_argument("--context", type=str, default=None) 
    
    parser.add_argument("--use_attention_mask", type=bool, default=True)
    parser.add_argument("--unlabeled_train_only", type=bool, default=True)
    parser.add_argument("--pool_version", type=str, default="v0")
    parser.add_argument("--random_sample", type=bool, default=False)
    parser.add_argument("--comment", type=str, default="debug", help="to distinguish the runs")
    parser.add_argument(
        "--shortcut_name",
        default="bert-base-uncased",
        type=str,
        help="Huggingface model shortcut name ",
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_unlabeled",
        default=0,
        type=int,
    )   

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--epoch",
        default=1,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--random_seed",
        default=4649,
        type=int,
        help="Random seed",
    )
    
    parser.add_argument(
        "--train_n_seed_cols",
        default=-1,
        type=int,
        help="number of seeding columns in training",
    )

    parser.add_argument(
        "--num_classes",
        default=78,
        type=int,
        help="Number of classes",
    )
    parser.add_argument("--multi_gpu",
                        action="store_true",
                        default=False,
                        help="Use multiple GPU")
    parser.add_argument("--fp16",
                        action="store_true",
                        default=False,
                        help="Use FP16")
    parser.add_argument("--warmup",
                        type=float,
                        default=0.,
                        help="Warmup ratio")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate") # 1e-5, 2e-5
    parser.add_argument("--task",
                        type=str,
                        default='gt-semtab22-dbpedia-all0',
                        choices=[
                            "sato0", "sato1", "sato2", "sato3", "sato4",
                            "msato0", "msato1", "msato2", "msato3", "msato4",
                            "gt-dbpedia0", "gt-dbpedia1", "gt-dbpedia2", "gt-dbpedia3", "gt-dbpedia4",
                            "gt-dbpedia-all0", "gt-dbpedia-all1", "gt-dbpedia-all2", "gt-dbpedia-all3", "gt-dbpedia-all4",
                            "gt-schema-all0", "gt-schema-all1", "gt-schema-all2", "gt-schema-all3", "gt-schema-all4",
                            "gt-semtab22-dbpedia", "gt-semtab22-dbpedia0", "gt-semtab22-dbpedia1", "gt-semtab22-dbpedia2", "gt-semtab22-dbpedia3", "gt-semtab22-dbpedia4",
                            "gt-semtab22-dbpedia-all", "gt-semtab22-dbpedia-all0", "gt-semtab22-dbpedia-all1", "gt-semtab22-dbpedia-all2", "gt-semtab22-dbpedia-all3", "gt-semtab22-dbpedia-all4",
                            "gt-semtab22-schema-class-all", "gt-semtab22-schema-property-all",
                            "turl", "turl-re", "col-popl-1", "col-popl-2", "col-popl-3", "row-popl",
                            "col-popl-turl-0", "col-popl-turl-1", "col-popl-turl-2",
                            "col-popl-turl-mdonly-0", "col-popl-turl-mdonly-1", "col-popl-turl-mdonly-2", "SOTAB"
                        ],
                        help="Task names}")
    parser.add_argument("--colpair",
                        action="store_true",
                        help="Use column pair embedding")
    parser.add_argument("--metadata",
                        action="store_true",
                        help="Use column header metadata")
    parser.add_argument("--from_scratch",
                        action="store_true",
                        help="Training from scratch")
    parser.add_argument("--cl_tag",
                        type=str,
                        default="wikitables/simclr/bert_100000_10_32_256_5e-05_sample_row4,sample_row4_tfidf_entity_column_0.05_0_last.pt",
                        help="path to the pre-trained file")
    parser.add_argument("--dropout_prob",
                        type=float,
                        default=0.5)
    parser.add_argument("--eval_test",
                        action="store_true",
                        help="evaluate on testset and do not save the model file")
    parser.add_argument("--eval_interval",
                        type=int,
                        default=1)
    parser.add_argument("--small_tag",
                        type=str,
                        default="comma",
                        help="e.g., by_table_t5_v1")
    parser.add_argument("--data_path",
                        type=str,
                        default="/data/zhihao/TU/")
    parser.add_argument("--pretrained_ckpt_path",
                        type=str,
                        default="/data/zhihao/TU/Watchog/model/")    

    args = parser.parse_args()
    task = args.task
    if args.small_tag != "":
        args.eval_test = True
    
    args.num_classes = task_num_class_dict[task]
    if args.colpair:
        assert "turl-re" == task, "colpair can be only used for Relation Extraction"
    if args.metadata:
        assert "turl-re" == task or "turl" == task, "metadata can be only used for TURL datasets"
    if "col-popl":
        # metrics = {
        #     "accuracy": CategoricalAccuracy(tie_break=True),
        # }
        if args.train_n_seed_cols != -1:
            if "col-popl" in task:
                assert args.train_n_seed_cols == int(task[-1]),  "# of seed columns must match"

    print("args={}".format(json.dumps(vars(args))))

    max_length = args.max_length
    batch_size = args.batch_size
    num_train_epochs = args.epoch

    shortcut_name = args.shortcut_name

    if args.colpair and args.metadata:
        taskname = "{}-colpair-metadata".format(task)
    elif args.colpair:
        taskname = "{}-colpair".format(task)
    elif args.metadata:
        taskname = "{}-metadata".format(task)
    elif args.train_n_seed_cols == -1 and 'popl' in task:
        taskname = "{}-mix".format(task)
    else:
        taskname = "".join(task)


    if args.from_scratch:
        if "gt" in task:
            tag_name = "{}/{}-{}-{}-pool{}-unlabeled{}-rand{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname,  "{}-fromscratch".format(shortcut_name), args.small_tag, args.comment,args.pool_version, args.max_unlabeled, args.random_sample,
                batch_size, max_length, num_train_epochs, args.dropout_prob, 
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        else:
            tag_name = "{}/{}-{}-{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname,  "{}-fromscratch".format(shortcut_name), args.small_tag, args.comment,
                batch_size, max_length, num_train_epochs, args.dropout_prob, 
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        
    else:
        if "gt" in task:
            tag_name = "{}/{}_{}-{}-pool{}-unlabeled{}-rand{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname, args.cl_tag.replace('/', '-'),  shortcut_name, args.small_tag, args.comment, args.pool_version, args.max_unlabeled, args.random_sample,
                batch_size, max_length, num_train_epochs, args.dropout_prob,
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        else:
            tag_name = "{}/{}_{}-{}-{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname, args.cl_tag.replace('/', '-'),  shortcut_name, args.small_tag, args.comment,
                batch_size, max_length, num_train_epochs, args.dropout_prob,
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')

    # if args.eval_test:
    #     if args.small_tag != '':
    #         tag_name = tag_name.replace('outputs', 'small_outputs')
    #         tag_name += '-' + args.small_tag
    print(tag_name)
    file_path = os.path.join(args.data_path, "Watchog", "outputs", tag_name)

    dirpath = os.path.dirname(file_path)
    if not os.path.exists(dirpath):
        print("{} not exists. Created".format(dirpath))
        os.makedirs(dirpath)
    
    if args.fp16:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
      
        
    # accelerator = Accelerator(mixed_precision="no" if not args.fp16 else "fp16")   
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="no" if not args.fp16 else "fp16", kwargs_handlers=[ddp_kwargs])

    device = accelerator.device
    ckpt_path = os.path.join(args.pretrained_ckpt_path, args.cl_tag)
    # ckpt_path = '/efs/checkpoints/{}.pt'.format(args.cl_tag)
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_hp = ckpt['hp']
    print(ckpt_hp)
 
    setattr(ckpt_hp, 'batch_size', args.batch_size)
    setattr(ckpt_hp, 'hidden_dropout_prob', args.dropout_prob)
    setattr(ckpt_hp, 'shortcut_name', args.shortcut_name)
    setattr(ckpt_hp, 'num_labels', args.num_classes)
    
    
    
    tokenizer = BertTokenizer.from_pretrained(shortcut_name)
    if task == "turl-re" and args.colpair:
        model = BertForMultiOutputClassification(ckpt_hp, device=device, lm=ckpt['hp'].lm, col_pair='Pair')
    elif "col-popl" in task:
        model = BertForMultiOutputClassificationColPopl(ckpt_hp, device=device, lm=ckpt['hp'].lm, n_seed_cols=int(task[i][-1]), cls_for_md="md" in task)
    else:
        model = BertForMultiOutputClassification(ckpt_hp, device=device, 
                                                 lm=ckpt['hp'].lm, 
                                                 version=args.pool_version,
                                                 use_attention_mask=args.use_attention_mask)
        if args.task == 'gt-semtab22-dbpedia-all0':
            best_state_dict = torch.load("/data/zhihao/TU/Watchog/outputs/gt-semtab22-dbpedia-all0/bert-base-uncased-fromscratch-semi1-Repeat@5-AttnMask-UnlabelValid-max-unlabeled@8-poolv0-unlabeled8-randFalse-bs16-ml128-ne50-do0.1_best_f1_macro.pt", map_location=device)
        elif args.task == 'SOTAB':
            best_state_dict = torch.load("/data/yongkang/TU/Watchog/outputs/SOTAB/bert-base-uncased-fromscratch-comma-bs16-ml128-ne50-do0.5_fully_deduplicated_best_f1_micro.pt", map_location=device)
        else:
            raise ValueError("best_state_dict not found")
        model.load_state_dict(best_state_dict, strict=False)

    # if not args.from_scratch:
    #     pre_model, trainset = load_checkpoint(ckpt)
    #     model.bert = pre_model.bert
    #     tokenizer = trainset.tokenizer
    #     del pre_model
    if task == "turl-re" and args.colpair and ckpt['hp'].lm != 'distilbert':
        config = BertConfig.from_pretrained(lm_mp[ckpt['hp'].lm])
        model.bert.pooler = BertMultiPairPooler(config).to(device)
        print("Use column-pair pooling")
        # print(type(model.bert.pooler), model.bert.pooler.hidden_size)

    
        
    with accelerator.main_process_first():
        if task in [
                "sato0", "sato1", "sato2", "sato3", "sato4", "msato0",
                "msato1", "msato2", "msato3", "msato4"
        ]:
            cv = int(task[-1])

            if task[0] == "m":
                multicol_only = True
            else:
                multicol_only = False

            dataset_cls = SatoCVTablewiseDataset
            train_dataset = dataset_cls(cv=cv,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=multicol_only,
                                        train_ratio=1.0,
                                        device=device,
                                        small_tag=args.small_tag,
                                        base_dirpath=os.path.join(args.data_path, "doduo", "data"), 
                                        )
            valid_dataset = dataset_cls(cv=cv,
                                        split="valid",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=multicol_only,
                                        train_ratio=1.0,
                                        device=device,
                                        small_tag=args.small_tag,
                                        base_dirpath=os.path.join(args.data_path, "doduo", "data"))

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                            sampler=train_sampler,
                                            batch_size=batch_size,
                                        #   collate_fn=collate_fn)
                                        collate_fn=padder)
            valid_dataloader = DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                        #   collate_fn=collate_fn)
                                        collate_fn=padder)
            test_dataset = dataset_cls(cv=cv,
                                        split="test",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=multicol_only,
                                        device=device,
                                        base_dirpath=os.path.join(args.data_path, "doduo", "data"))
            test_dataloader = DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            collate_fn=padder)   
            
        elif "gt-dbpedia" in task or "gt-schema" in task or "gt-semtab" in task:
            if 'dbpedia' in task:
                if 'semtab22' in task:
                    src = 'dbpedia_property' # gt-dbpedia-semtab22-all0
                else:
                    src = 'dbpedia'
            else:
                if 'semtab22' in task:
                    if 'schema-property' in task:
                        src = 'schema_property'
                    else:
                        src = 'schema_class'
                else:
                    src = 'schema'
            if task[-1] in "01234":
                cv = int(task[-1])
                veri_dataset = VerificationCompareDataset(data_path=f"/data/zhihao/TU/Watchog/verification/{args.task}_veri_data.pth", pos_ratio=args.pos_ratio if not args.reweight else None, context=args.context)
                # veri_dataset = VerificationBinaryDataset(pos_ratio=args.pos_ratio)
                veri_padder = veri_collate_fn(0)
                veri_dataloader = DataLoader(
                    veri_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=veri_padder
                )
                
                
                
                test_dataset_iter = GittablesTablewiseIterateDataset(cv=cv,
                            split="test", src=src,
                            tokenizer=tokenizer,
                            max_length=max_length,
                            gt_only='all' not in task,
                            device=device,
                            base_dirpath=os.path.join(args.data_path, "GitTables/semtab_gittables/2022"),
                            small_tag="semi1")
                padder = collate_fn_iter(tokenizer.pad_token_id)
                test_dataloader_iter = DataLoader(test_dataset_iter,
                                                batch_size=1,
                                            #   collate_fn=collate_fn)
                                            collate_fn=padder) 
                test_dataset = torch.load(f"/data/zhihao/TU/Watchog/verification/{args.task}_test_data.pth")
                test_embs = test_dataset['embs']
                test_logits = test_dataset['logits']
                test_embs_target = test_dataset['target_embs']
            else:
                raise ValueError("cv must be in 01234")
        elif "SOTAB" in task:
            src = None


            veri_dataset = VerificationCompareDataset(data_path=f"/data/zhihao/TU/Watchog/verification/{args.task}_veri_data.pth", pos_ratio=args.pos_ratio if not args.reweight else None, context=args.context)
            # veri_dataset = VerificationBinaryDataset(pos_ratio=args.pos_ratio)
            veri_padder = veri_collate_fn(0)
            veri_dataloader = DataLoader(
                veri_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=veri_padder
            )
            
            
            
            test_dataset_iter = SOTABTablewiseIterateDataset(# cv=cv,
                                        split="test",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        # multicol_only=multicol_only,
                                        device=device,
                                        gt_only=False,
                                        base_dirpath=os.path.join(args.data_path, "SOTAB")
                                        )
            padder = collate_fn_iter(tokenizer.pad_token_id)
            test_dataloader_iter = DataLoader(test_dataset_iter,
                                            batch_size=1,
                                        #   collate_fn=collate_fn)
                                        collate_fn=padder) 
            test_dataset = torch.load(f"/data/zhihao/TU/Watchog/verification/{args.task}_test_data.pth")
            test_embs = test_dataset['embs']
            test_logits = test_dataset['logits']
            test_embs_target = test_dataset['target_embs']
       
        elif "popl" in task:
            if "col-popl" in task:
                padder = collate_fn(trainset.tokenizer.pad_token_id, data_only=False)
            
                if args.train_n_seed_cols == -1:
                    train_filepath = args.data_path + "/col_popl_turl/train_col_popl_seedmix.jsonl"    
                else:
                    train_filepath = args.data_path + "/col_popl_turl/train_col_popl_seed{}.jsonl".format(args.train_n_seed_cols)
                test_n_seed_cols = int(task[-1])
                valid_filepath = args.data_path + "/col_popl_turl/dev_col_popl_seed{}.jsonl".format(test_n_seed_cols)
                test_filepath = args.data_path + "/col_popl_turl/test_col_popl_seed{}.jsonl".format(test_n_seed_cols)
                dataset_cls = ColPoplTablewiseDataset
            else:
                raise ValueError("not col polulation.")

            train_dataset = dataset_cls(filepath=train_filepath,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        train_ratio=train_ratio,
                                        use_content="mdonly" not in task,
                                        use_metadata="md" in task,
                                        device=device)
            valid_dataset = dataset_cls(filepath=valid_filepath,
                                        split="dev",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        use_content="mdonly" not in task,
                                        use_metadata="md" in task,
                                        device=device)

            # Can be the same
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=batch_size,
                                          collate_fn=padder)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          collate_fn=padder)
            test_dataset = dataset_cls(filepath=test_filepath,
                                       split="test",
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       multicol_only=False,
                                       use_content="mdonly" not in task,
                                       use_metadata="md" in task,
                                       device=device)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         collate_fn=padder)
            valid_evaluator = ColPoplEvaluator(valid_dataset, task_type="col-popl-turl")
            test_evaluator = ColPoplEvaluator(test_dataset, task_type="col-popl-turl")    
            
        elif "turl" in task:
            if task in ["turl"]:
                if args.small_tag == "":
                    filepath = "data/doduo/table_col_type_serialized{}.pkl".format(
                        '_with_metadata' if args.metadata else ''
                    )
                else:
                    filepath = "data/turl_small/table_col_type_serialized{}_{}.pkl".format(
                        '_with_metadata' if args.metadata else '', 
                        args.small_tag
                    )
                
                dataset_cls = TURLColTypeTablewiseDataset
            elif task in ["turl-re"]:
                if args.small_tag == "":
                    filepath = "data/doduo/table_rel_extraction_serialized{}.pkl".format(
                        '_with_metadata' if args.metadata else ''
                    )
                else:
                    filepath = "data/turl-re_small/table_rel_extraction_serialized{}_{}.pkl".format(
                        '_with_metadata' if args.metadata else '',
                        args.small_tag
                    )
                dataset_cls = TURLRelExtTablewiseDataset
            else:
                raise ValueError("turl tasks must be turl or turl-re.")

            train_dataset = dataset_cls(filepath=filepath,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        train_ratio=1.0,
                                        device=device)
            valid_dataset = dataset_cls(filepath=filepath,
                                        split="dev",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        device=device)

            # Can be the same
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                            sampler=train_sampler,
                                            batch_size=batch_size,
                                            collate_fn=padder)
            valid_dataloader = DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            collate_fn=padder)
            test_dataset = dataset_cls(filepath=filepath,
                                        split="test",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        device=device)
            test_dataloader = DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            collate_fn=padder)
        else:
            raise ValueError("task name must be either sato or turl.")

    if accelerator.is_local_main_process and args.wandb:
        wandb.init(config=args,
            project="TableUnderstandingVerification",
            name=f"{args.model} {args.comment} {args.small_tag}_DS@{args.task}_scratch@{args.from_scratch}_maxlen@{args.max_length}_bs@{args.batch_size}",
            group="TU",
            )
        wandb.log({
                f"tag_name": tag_name,
            }, commit=True)
    
    verifier = Verifier(module=args.veri_module, dropout=args.dropout_prob, norm=args.norm).to(device)
    t_total = len(veri_dataloader) * num_train_epochs
    optimizer = AdamW(verifier.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(t_total*args.warmup_ratio),
                                                num_training_steps=t_total,
                                                min_lr=args.lr/100)

    # if args.reweight:
    #     pos_weight = torch.tensor([(1-args.pos_ratio)/args.pos_ratio]).to(device)
    # else:
    #     pos_weight = None
    # if args.loss == "BCE":
    #     loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # elif args.loss == "MSE":
    #     pos_weight = (1-args.pos_ratio)/args.pos_ratio
    #     loss_fn = torch.nn.MSELoss(reduction='none')
    # else:
    #     raise NotImplementedError(f"loss function {args.loss} not implemented")
    set_seed(args.random_seed)
    
    model, verifier, optimizer, veri_dataloader, scheduler = accelerator.prepare(
        model, verifier, optimizer, veri_dataloader, scheduler
    )

    model = model.to(device)
    verifier = verifier.to(device)
    # model = model.cuda()
    # Best validation score could be zero
    best_vl_micro_f1 = -1
    best_vl_macro_f1 = -1
    best_vl_loss = 1e10
    best_vl_micro_f1s_epoch = -1
    best_vl_macro_f1s_epoch = -1
    best_vl_loss_epoch = -1
    loss_info_list = []
    eval_dict = defaultdict(dict)
    time_epochs = []
    # =============================Training Loop=============================
    for epoch in range(num_train_epochs):
        t1 = time()
        print("Epoch", epoch, "starts")
        model.eval()
        verifier.train()
        tr_loss = 0.
        device = accelerator.device
        num_samples = 0
        for batch_idx, batch in enumerate(veri_dataloader):
            embs = batch["embs"].to(device)
            # logits = batch["logits"].to(device)
            
            # cls_indexes = batch["cls_indexes"].to(device)
            # input_data = batch["data"].T.to(device)
            # logits, embs = model(input_data, cls_indexes=cls_indexes, get_enc=True)
    
            scores = verifier(embs)
            num_samples += len(scores)
            pos_scores, neg_scores = scores[:, 0], scores[:, 1]
            loss = torch.clamp(pos_scores - neg_scores + args.margin, min=0.0).mean()

            accelerator.backward(loss)
            # loss.backward()
            # print(f"batch {batch_idx}", loss.cpu().detach().item(), num_samples)
            tr_loss += loss.cpu().detach().item()
            optimizer.step()
            current_lr = scheduler.get_lr()[-1]
            scheduler.step()
            optimizer.zero_grad()
            
        tr_loss /= num_samples
        t2 = time()
        time_epoch_train = t2-t1
        

        # ======================= Validation =======================
        if (epoch+1) % args.eval_interval == 0 or epoch > num_train_epochs//2:
            model.eval()
            verifier.eval()
            with accelerator.main_process_first():
                device = accelerator.device
                labels_test = []
                logits_test = []

                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_dataloader_iter):
                        embs = test_embs[batch_idx][0].reshape(-1).to(device)
                        logits = test_logits[batch_idx][0].reshape(-1).to(device)
                        # scores_init = verifier(embs)
                        max_score = -float("inf")
                        embs_target = test_embs_target[batch_idx][0].reshape(-1).to(device)
                        
                        for i in range(len(test_embs[batch_idx])):
                            logits_temp = test_logits[batch_idx][i].reshape(-1).to(device)
                            embs_temp = test_embs[batch_idx][i].reshape(-1).to(device)
                            if args.context is not None or args.context != "None":
                                if args.context == "init":
                                    embs_temp = torch.cat([embs, embs_temp], dim=-1)
                                elif args.context == "target":
                                    embs_temp = torch.cat([embs_target, embs_temp], dim=-1)
                            embs_temp = embs_temp.reshape(1,-1)
                            scores_temp = verifier(embs_temp).item()
                            predict_temp = logits_temp.argmax().item()
                            # if len(x) == 1 and 0 in x:
                            #     predict_target = predict_temp
                            #     msp_target = msp_temp
                            # # print(x, msp_temp, predict_temp)
                            # if 0 not in x and msp_temp > debias_threshold and (predict_temp != predict_target):
                            #     debias_classes.append(predict_temp)
                            #     continue
                            if scores_temp > max_score:
                                max_score = scores_temp
                                logits = logits_temp.clone()
                        labels_test.append(batch["label"].cpu())
                        logits_test.append(logits.detach().cpu())
                    labels_test = torch.cat(labels_test, dim=0)
                    logits_test = torch.stack(logits_test, dim=0)
                    preds_test = torch.argmax(logits_test, dim=1)




                from sklearn.metrics import confusion_matrix, f1_score
                ts_pred_list = logits_test.argmax(
                                            1).cpu().detach().numpy().tolist()
                ts_micro_f1 = f1_score(labels_test.reshape(-1).numpy().tolist(),
                                    ts_pred_list,
                                    average="micro")
                ts_macro_f1 = f1_score(labels_test.reshape(-1).numpy().tolist(),
                                    ts_pred_list,
                                    average="macro")
                
                t3 = time()
                time_epoch_test = t3-t2
                print(
                    "Epoch {} ({}): tr_loss={:.7f} lr={:.7f} ({:.2f} sec.)"
                    .format(epoch, task, tr_loss, current_lr, time_epoch_train),
                    "ts_macro_f1={:.4f} ts_micro_f1={:.4f} ({:.2f} sec.)"
                    .format(ts_macro_f1, ts_micro_f1, time_epoch_test))
                if accelerator.is_local_main_process and args.wandb:
                    wandb.log({
                            f"train/loss": tr_loss,
                            f"train/time": time_epoch_train,
                            f"train/learning_rate": current_lr,
                            f"test/micro_f1": ts_micro_f1,
                            f"test/macro_f1": ts_macro_f1,
                            F"test/time": time_epoch_test
                        }, step=epoch+1, commit=True)             
                if ts_macro_f1 > best_vl_macro_f1:
                    best_vl_macro_f1 = ts_macro_f1
                    best_vl_macro_f1s_epoch = epoch
                    best_state_dict_macro = deepcopy(verifier.state_dict())
                    torch.save(best_state_dict_macro, "{}_verifier_binary_best_f1_macro.pt".format(file_path))
                if ts_micro_f1 > best_vl_micro_f1:
                    best_vl_micro_f1 = ts_micro_f1
                    best_vl_micro_f1s_epoch = epoch
                    best_state_dict_micro = deepcopy(verifier.state_dict())
                    torch.save(best_state_dict_micro, "{}_verifier_binary_best_f1_micro.pt".format(file_path))
        else:
            if accelerator.is_local_main_process and args.wandb:
                wandb.log({
                        f"train/loss": tr_loss,
                        f"train/time": time_epoch_train,
                        f"train/learning_rate": current_lr,
                    }, step=epoch+1, commit=True)  
    if accelerator.is_local_main_process and args.wandb:
        wandb.finish()
    # with accelerator.main_process_first():
    #     if args.eval_test:
    #         if "popl" in task:
    #             loss_info_df = pd.DataFrame(loss_info_list,
    #                                         columns=[
    #                                             "tr_loss", "vl_loss",
    #                                             "vl_map", "vl_rpr", "vl_ndcg_10", "vl_ndcg_20", 
    #                                             "ts_map", "ts_rpr", "ts_ndcg_10", "ts_ndcg_20",
    #                                             "best_vl_map_epoch"
    #                                         ])
    #         else:
    #             loss_info_df = pd.DataFrame(loss_info_list,
    #                                         columns=[
    #                                             "tr_loss", "tr_f1_macro_f1",
    #                                             "tr_f1_micro_f1", "vl_loss",
    #                                             "vl_f1_macro_f1", "vl_f1_micro_f1",
    #                                             "ts_macro_f1", "ts_micro_f1",
    #                                             "best_vl_macro_f1_epoch", "best_vl_micro_f1_epoch"
    #                                         ])
    #         loss_info_df.to_csv("{}_loss_info.csv".format(tag_name))


