from argparse import Namespace
import torch
import random
import pandas as pd
import numpy as np
import os
import pickle
import json
import re
import transformers
from torch.utils import data
from torch.nn.utils import rnn
from transformers import AutoTokenizer
from .augment import augment
from typing import List
from functools import reduce
import operator
from .preprocessor import computeTfIdf, tfidfRowSample, preprocess, load_jsonl
from itertools import chain
import copy
from sklearn.preprocessing import LabelEncoder

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}

def collate_fn(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    batch = {"data": data, "label": label}
    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch



class TableDataset(data.Dataset):
    """Table dataset for unsupervised contrastive learning"""

    def __init__(self,
                 path,
                 augment_op,
                 max_len=256,
                 size=None,
                 lm='bert',
                 single_column=False,
                 sample_meth='wordProb',
                 table_order='column'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_len = max_len
        self.path = path

        # assuming tables are in csv format
        try:
            self.tables = [fn for fn in os.listdir(path) if '.csv' in fn]
        except:
            self.tables = []

        # only keep the first n tables
        if size is not None:
            self.tables = self.tables[:size]
        self.size = size

        self.table_cache = {}
        self.table_col_md = {}

        # augmentation operators
        self.augment_op = augment_op

        # logging counter
        self.log_cnt = 0

        # sampling method
        self.sample_meth = sample_meth

        # single-column mode
        self.single_column = single_column

        # row or column order for preprocessing
        self.table_order = table_order

        # tokenizer cache
        self.tokenizer_cache = {}

    @staticmethod
    def from_hp(path: str, hp: Namespace):
        """Construct a TableDataset from hyperparameters

        Args:
            path (str): the path to the table directory
            hp (Namespace): the hyperparameters

        Returns:
            TableDataset: the constructed dataset
        """
        return TableDataset(path,
                         augment_op=hp.augment_op,
                         lm=hp.lm,
                         max_len=hp.max_len,
                         size=hp.size,
                         single_column=hp.single_column,
                         sample_meth=hp.sample_meth,
                         table_order=hp.table_order)

    def load_from_wikitables(self, path):
        """Load the tables from the Wikitables dataset
        Args:
            path (str): the path to the table directory
        """
        inf = open(path, 'r')
        lines = inf.readlines(10000000)
        self.tables = []
        self.table_cache = {}
        while lines:
            for l in lines:
                t = json.loads(l)
                t_dict = {}
                columns = t['processed_tableHeaders']
                duplicated = {}
                for i, col in enumerate(columns):
                    if col not in t_dict:
                        t_dict[col] = []
                    else:
                        duplicated[i] = True
                for row in t['tableData']:
                    for i in range(len(row)):
                        if i not in duplicated:
                            t_dict[columns[i]].append(row[i]['text'])
                df = pd.DataFrame(t_dict)
                self.tables.append(t['_id'])
                self.table_cache[len(self.tables)-1] = df
                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
            lines = inf.readlines(1000000)
        print('Number of tables: ', len(self.tables))
    
    
    def load_gittables_train_test(self):
        """Since the semtab benchmarks are also from the gittables corpus, 
        we skip tables used in the benchmark for the contrastive learning phase
        """
        base_dirpaths = ['./data/gittables_semtab22/tables/', './data/gittables/tables/']
        self.train_test_table_cache = {}
        train_test_table_hash = {}
        for base_dirpath in base_dirpaths:
            dir_list = os.listdir(base_dirpath)
            for f in dir_list:
                if f.startswith('GitTables') and f.endswith('.csv'):
                    table_file_path = os.path.join(base_dirpath, f)
                    df = pd.read_csv(table_file_path)
                    self.train_test_table_cache[len(self.train_test_table_cache)] = df
                    new_df = df.iloc[:16,]
                    hc = '|$|'.join(['|r|'.join(map(lambda x: str(x).strip(), new_df[col].tolist())) for col in new_df.columns])
                    train_test_table_hash[hc] = len(self.train_test_table_cache)-1
        return train_test_table_hash


    def load_from_gittables(self, path, num_blocks=32, max_cols=32):
        """Load the tables from the Gittables dataset
        Args:
            path (str): the path to the table directory
        """
        dup_cnt = 0
        total_cnt = 0
        num_cols = []
        self.tables = []
        self.table_cache = {}
        self.train_test_table_hash = self.load_gittables_train_test()
        dup_table_cnt = 0
        for bid in range(num_blocks):
            tlist = pickle.load(open(path.format(bid), 'rb'))
            for t in tlist['table_cache']:
                # print(t.keys())
                columns = t['table'].columns
                df = t['table']
                hc_df = t['table'].iloc[:16,]
                hc = '|$|'.join(['|r|'.join(map(lambda x: str(x).strip(), hc_df[col].tolist())) for col in hc_df.columns])
                if hc in self.train_test_table_hash:
                    dup_table_cnt += 1
                    continue
                duplicated = {}
                t_dict = {}
                cols = [re.sub(r'.*([1-9][0-9]{3})', '<year>', col.strip().lower()) for col in columns]
                rename_map = {}
                non_dup_cols = []
                for i, col in enumerate(cols):
                    rename_map[df.columns[i]] = col
                    if col not in t_dict:
                        t_dict[col] = []
                        non_dup_cols.append(i)
                    else:
                        duplicated[i] = True
                        dup_cnt += 1
                    total_cnt += 1
                df = df.rename(columns=rename_map)
                df = df.iloc[:, non_dup_cols]
                df = df.iloc[:, :min(len(non_dup_cols), max_cols)]
                self.tables.append(len(self.table_cache))
                self.table_cache[len(self.tables)-1] = df
                num_cols.append(len(list(df.columns)))
                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
                
        print('Number of tables: ', len(self.tables))
        
    def _read_table(self, table_id):
        """Read a table"""
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn, lineterminator='\n')
            self.table_cache[table_id] = table

        return table


    def _tokenize(self, table: pd.DataFrame) -> List[int]:
        """Tokenize a DataFrame table

        Args:
            table (DataFrame): the input table

        Returns:
            List of int: list of token ID's with special tokens inserted
            Dictionary: a map from column names to the position of corresponding special tokens
        """
        res = []
        max_tokens = self.max_len * 2 // len(table.columns) # max tokens per column
        budget = max(1, self.max_len // len(table.columns) - 1)
        tfidfDict = computeTfIdf(table) if "tfidf" in self.sample_meth else None # from preprocessor.py

        # a map from column names to special token indices
        column_mp = {}

        # column-ordered preprocessing
        if self.table_order == 'column':
            if 'row' in self.sample_meth: 
                table = tfidfRowSample(table, tfidfDict, max_tokens) # TODO
            for column in table.columns:
                tokens = preprocess(table[column], tfidfDict, max_tokens, self.sample_meth) # from preprocessor.py
                col_text = self.tokenizer.cls_token + " " + \
                        ' '.join(tokens[:max_tokens]) + " "

                column_mp[column] = len(res)
                res += self.tokenizer.encode(text=col_text,
                                        max_length=budget,
                                        add_special_tokens=False,
                                        truncation=True)
        else:
            # row-ordered preprocessing
            reached_max_len = False
            for rid in range(len(table)):
                row = table.iloc[rid:rid+1]
                for column in table.columns:
                    tokens = preprocess(row[column], tfidfDict, max_tokens, self.sample_meth) # from preprocessor.py
                    if rid == 0:
                        column_mp[column] = len(res)
                        col_text = self.tokenizer.cls_token + " " + \
                                ' '.join(tokens[:max_tokens]) + " "
                    else:
                        col_text = self.tokenizer.pad_token + " " + \
                                ' '.join(tokens[:max_tokens]) + " "

                    tokenized = self.tokenizer.encode(text=col_text,
                                        max_length=budget,
                                        add_special_tokens=False,
                                        truncation=True)

                    if len(tokenized) + len(res) <= self.max_len:
                        res += tokenized
                    else:
                        reached_max_len = True
                        break

                if reached_max_len:
                    break

        self.log_cnt += 1
        
        return res, column_mp


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.tables)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the first view
            List of int: token ID's of the second view
            List of tuple: column special token [CLS] indices of each view
        """
        table_ori = self._read_table(idx)

        # single-column mode: only keep one random column
        if self.single_column:
            col = random.choice(table_ori.columns)
            table_ori = table_ori[[col]]

        # apply the augmentation operator
        if ',' in self.augment_op:
            op1, op2 = self.augment_op.split(',')
            table_tmp = table_ori
            table_ori = augment(table_tmp, op1)
            table_aug = augment(table_tmp, op2)
        elif self.augment_op in ['dropout', '', 'None', None]:
            table_aug = table_ori 
        else:
            table_aug = augment(table_ori, self.augment_op)

        # convert table into string
        x_ori, mp_ori = self._tokenize(table_ori)
        x_aug, mp_aug = self._tokenize(table_aug)

        # make sure that x_ori and x_aug has the same number of cls tokens
        # x_ori_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_ori])
        # x_aug_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_aug])
        # assert x_ori_cnt == x_aug_cnt

        # insertsect the two mappings
        cls_indices = []
        for col in mp_ori:
            if col in mp_aug:
                cls_indices.append((mp_ori[col], mp_aug[col])) 

        return x_ori, x_aug, cls_indices


    def pad(self, batch):
        """Merge a list of dataset items into a training batch

        Args:
            batch (list of tuple): a list of sequences

        Returns:
            LongTensor: x_ori of shape (batch_size, seq_len)
            LongTensor: x_aug of shape (batch_size, seq_len)
            tuple of List: the cls indices
        """
        x_ori, x_aug, cls_indices = zip(*batch)
        max_len_ori = max([len(x) for x in x_ori])
        max_len_aug = max([len(x) for x in x_aug])
        maxlen = max(max_len_ori, max_len_aug)
        x_ori_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_ori]
        x_aug_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_aug]

        # decompose the column alignment
        cls_ori = []
        cls_aug = []
        for item in cls_indices:
            cls_ori.append([])
            cls_aug.append([])

            for idx1, idx2 in item:
                cls_ori[-1].append(idx1)
                cls_aug[-1].append(idx2)
        
        return torch.LongTensor(x_ori_new), torch.LongTensor(x_aug_new), (cls_ori, cls_aug)

class SupCLTableDataset(TableDataset):
    """Table dataset for contrastive learning with supervision signal"""
    def __init__(self,
                 path,
                 augment_op,
                 max_len=256,
                 size=None,
                 lm='bert',
                 single_column=False,
                 sample_meth='wordProb',
                 table_order='column'):
        super().__init__(path, augment_op, max_len, size, lm, single_column, sample_meth, table_order)
        
    @staticmethod
    def from_hp(path: str, hp: Namespace):
        """Construct a SupCLTableDataset from hyperparameters

        Args:
            path (str): the path to the table directory
            hp (Namespace): the hyperparameters

        Returns:
            TableDataset: the constructed dataset
        """
        return SupCLTableDataset(path,
                         augment_op=hp.augment_op,
                         lm=hp.lm,
                         max_len=hp.max_len,
                         size=hp.size,
                         single_column=hp.single_column,
                         sample_meth=hp.sample_meth,
                         table_order=hp.table_order)

    def load_from_turl(self, train_path, num_classes=255):
        """Load the tables from the TURL semantic type detection dataset
        Args:
            train_path (str): the path to the training data
        """
        turl_table_cache = pickle.load(open(train_path, 'rb'))
        self.tables = []
        self.table_cache = {}
        self.col_label_cache = {}
        self.header_cache = {}

        for idx in turl_table_cache:
            tid = turl_table_cache[idx]['_id']
            self.tables.append(tid)
            self.table_cache[len(self.tables)-1] = turl_table_cache[idx]['raw_table_content']
            self.header_cache[len(self.tables)-1] = turl_table_cache[idx]['headers']
            # self.col_label_cache[len(self.tables)-1] = [turl_table_cache[idx]['label_dict'][j] for j in sorted(sato_table_cache[idx]['label_dict'].keys())]
            labels = []
            for j in turl_table_cache[idx]['label_dict']:
                oh_enc = [0 for _ in range(255)]
                for l in turl_table_cache[idx]['label_dict'][j]:
                    oh_enc[l] = 1
                labels.append(oh_enc)
            self.col_label_cache[len(self.tables)-1] = list(labels)
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
        
        print('Number of tables: ', len(self.tables))
        
    def load_from_wikitables_headerprocessed(self, path, hf_rank_path, header_top_k=-1, entity_col_only=False):
        
        """Load the tables from the Wikitables dataset, following the pre-processing in the TURL paper
        Args:
            path (str): the path to the training data
            hf_rank_path (str): the path to pre-computed ranking of headers by frequency
            header_top_k (int): use the top-k headers; -1 means no filtering
        """
        hf_rank = json.load(open(hf_rank_path, 'r'))
        self.h2i = {}
        for h in hf_rank:
            self.h2i[h] = len(self.h2i)
        inf = open(path, 'rb')
        lines = inf.readlines(1000000)
        self.tables = []
        self.table_cache = {}
        self.table_col_md = {}
        self.col_label_cache = {}
        self.header_cache = {}
        total_cnt = 0
        dup_cnt = 0
        while lines:
            for l in lines:
                t = json.loads(l)
                t_dict = {}
                columns = t['processed_tableHeaders']
                duplicated = {}
                t_dict = {}
                top_k_flag = False
                top_k_flag_per_col = {}
                is_entity_col = {}
                for i, col in enumerate(columns):
                    if entity_col_only:
                        if i not in t['entityColumn']:
                            continue
                    col = re.sub(r'.*([1-9][0-9]{3})', '<year>', col.strip().lower())
                    if header_top_k == -1 or hf_rank[col] < header_top_k:
                        top_k_flag_per_col[col] = True
                        top_k_flag = True
                    if col not in t_dict:
                        t_dict[col] = []
                        is_entity_col[col] = i in t['entityColumn']
                    else:
                        duplicated[i] = True
                        dup_cnt += 1
                    total_cnt += 1
                
                for row in t['tableData']:
                    for i in range(len(row)):
                        if i not in duplicated:
                            new_col = re.sub(r'.*([1-9][0-9]{3})', '<year>', columns[i].strip().lower())
                            t_dict[new_col].append(row[i]['text'])
                if not top_k_flag:
                    continue
                df = pd.DataFrame(t_dict)
                self.tables.append(t['_id'])
                self.table_cache[len(self.tables)-1] = df
                self.table_col_md[len(self.tables)-1] = {
                    'is_top_k_header': top_k_flag_per_col,
                    'is_entity_col': is_entity_col
                }
                cols = list(df.columns)
                self.header_cache[len(self.tables)-1] = cols
                self.col_label_cache[len(self.tables)-1] = {_:self.h2i[_] for _ in cols} # labels are column headers

                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
            lines = inf.readlines(1000000)
            
        print('Number of tables: ', len(self.tables))
    
    def load_from_sato(self, base_path, basename, cv, all_path):
        """Load the tables from the SATO semantic type detection dataset
        Args:
            train_path (str): the path to the training data
        """
        
        df_list = []
        for i in range(5):
            if i == cv:
                continue
            filepath = os.path.join(base_path, basename.format(i))
            df_list.append(pd.read_csv(filepath))
        df = pd.concat(df_list, axis=0)
        train_tables = [g[0] for _, g in enumerate(df.groupby(["table_id"]))]
        sato_table_cache = pickle.load(open(all_path, 'rb'))
        self.tables = []
        self.table_cache = {}
        self.col_label_cache = {}
        self.header_cache = {}

        num_tables = len(train_tables)
        valid_index = int(num_tables * 0.8)
        train_tables = train_tables[:valid_index]
        for idx in sato_table_cache:
            tid = sato_table_cache[idx]['_id']
            if tid in train_tables:
                self.tables.append(tid)
                self.table_cache[len(self.tables)-1] = sato_table_cache[idx]['raw_table_content']
                self.header_cache[len(self.tables)-1] = sato_table_cache[idx]['headers']
                self.col_label_cache[len(self.tables)-1] = [sato_table_cache[idx]['label_dict'][j] for j in sorted(sato_table_cache[idx]['label_dict'].keys())]
                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
    
    def load_gittables_train_test(self):
        base_dirpaths = ['./data/gittables_semtab22/tables/', './data/gittables/tables/']
        self.train_test_table_cache = {}
        train_test_table_hash = {}
        for base_dirpath in base_dirpaths:
            dir_list = os.listdir(base_dirpath)
            for f in dir_list:
                if f.startswith('GitTables') and f.endswith('.csv'):
                    table_file_path = os.path.join(base_dirpath, f)
                    df = pd.read_csv(table_file_path)
                    self.train_test_table_cache[len(self.train_test_table_cache)] = df
                    new_df = df.iloc[:16,]
                    hc = '|$|'.join(['|r|'.join(map(lambda x: str(x).strip(), new_df[col].tolist())) for col in new_df.columns])
                    train_test_table_hash[hc] = len(self.train_test_table_cache)-1
        return train_test_table_hash

    def load_from_gittables(self, path, num_blocks=32, hf_rank_path='./data/gittables/gittables.processed.header.freq.json', max_cols=32):
        dup_cnt = 0
        total_cnt = 0
        num_cols = []
        hf_rank = json.load(open(hf_rank_path, 'r'))
        h2i = {}
        for h in hf_rank:
            h2i[h] = len(h2i)
        self.tables = []
        self.table_cache = {}
        self.header_cache = {}
        self.col_label_cache = {}
        
        self.train_test_table_hash = self.load_gittables_train_test()
        dup_table_cnt = 0
        for bid in range(num_blocks):
            tlist = pickle.load(open(path.format(bid), 'rb'))
            for t in tlist['table_cache']:
                # print(t.keys())
                columns = t['table'].columns
                df = t['table']
                hc_df = t['table'].iloc[:16,]
                hc = '|$|'.join(['|r|'.join(map(lambda x: str(x).strip(), hc_df[col].tolist())) for col in hc_df.columns])
                if hc in self.train_test_table_hash:
                    dup_table_cnt += 1
                    continue
                duplicated = {}
                t_dict = {}
                cols = [re.sub(r'.*([1-9][0-9]{3})', '<year>', col.strip().lower()) for col in columns]
                rename_map = {}
                non_dup_cols = []
                for i, col in enumerate(cols):
                    rename_map[df.columns[i]] = col
                    if col not in t_dict:
                        t_dict[col] = []
                        non_dup_cols.append(i)
                    else:
                        duplicated[i] = True
                        dup_cnt += 1
                    total_cnt += 1
                df = df.rename(columns=rename_map)
                df = df.iloc[:, non_dup_cols]
                df = df.iloc[:, :min(len(non_dup_cols), max_cols)]
                self.tables.append(len(self.table_cache))
                self.table_cache[len(self.tables)-1] = df
                self.header_cache[len(self.tables)-1] = list(df.columns)
                self.col_label_cache[len(self.tables)-1] = {_:h2i[_] for _ in list(df.columns)}
                num_cols.append(len(list(df.columns)))
                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
                
        print('Number of tables: ', len(self.tables))
        print(dup_cnt, total_cnt, dup_table_cnt)
        # num_cols_df = pd.DataFrame(num_cols, columns=['num_cols'])
        # print(num_cols_df.describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]))
  
    def _read_table(self, table_id):
        """Read a table"""
        # print(table_id)
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
            col_label = self.col_label_cache[table_id]
            headers = self.header_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn, lineterminator='\n')
            self.table_cache[table_id] = table

        return table, col_label, headers

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the first view
            List of int: token ID's of the second view
            List of tuple: column special token [CLS] indices of each view
            List of lists: labels (i.e., column headers) of each column of the first view
            List of lists: labels (i.e., column headers) of each column of the second view
        """
        table_ori, col_label, headers = self._read_table(idx)

        # single-column mode: only keep one random column
        if self.single_column:
            col = random.choice(table_ori.columns)
            table_ori = table_ori[[col]]

        # apply the augmentation operator
        if ',' in self.augment_op:
            op1, op2 = self.augment_op.split(',')
            table_tmp = table_ori
            table_ori = augment(table_tmp, op1)
            table_aug = augment(table_tmp, op2)
        elif self.augment_op in ['dropout', '', 'None', None]:
            table_aug = table_ori
        else:
            table_aug = augment(table_ori, self.augment_op)

        # convert table into string
        x_ori, mp_ori = self._tokenize(table_ori)
        x_aug, mp_aug = self._tokenize(table_aug)

        # insertsect the two mappings
        cls_indices = []
        ori_col_label = []
        for col in mp_ori:
            if col in mp_aug:
                cls_indices.append((mp_ori[col], mp_aug[col]))
                ori_col_label.append(col_label[col])
        aug_col_label = []
        for col in mp_aug:
            aug_col_label.append(col_label[col])
        return x_ori, x_aug, cls_indices, ori_col_label, aug_col_label
    
    def pad(self, batch):
        """Merge a list of dataset items into a training batch

        Args:
            batch (list of tuple): a list of sequences

        Returns:
            LongTensor: x_ori of shape (batch_size, seq_len)
            LongTensor: x_aug of shape (batch_size, seq_len)
            tuple of List: the cls indices
            LongTensor: y_ori of shape (# of columns)
            LongTensor: y_aug of shape (# of columns)
        """
        x_ori, x_aug, cls_indices, ori_col_label, aug_col_label = zip(*batch)
        max_len_ori = max([len(x) for x in x_ori])
        max_len_aug = max([len(x) for x in x_aug])
        maxlen = max(max_len_ori, max_len_aug)
        x_ori_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_ori]
        x_aug_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_aug]

        # decompose the column alignment
        cls_ori = []
        cls_aug = []
        for item in cls_indices:
            cls_ori.append([])
            cls_aug.append([])

            for idx1, idx2 in item:
                cls_ori[-1].append(idx1)
                cls_aug[-1].append(idx2)
        ori_col_label = list(chain.from_iterable(ori_col_label))
        aug_col_label = list(chain.from_iterable(aug_col_label))
            
        return torch.LongTensor(x_ori_new), torch.LongTensor(x_aug_new), (cls_ori, cls_aug), torch.LongTensor(ori_col_label), torch.LongTensor(aug_col_label)


class SatoCVTablewiseDataset(data.Dataset):
    """Table dataset for finetuning and evaluating semantic type detection on the SATO dataset"""

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "./data/doduo",
            small_tag: str = "",
            ):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"
        if small_tag != '':
            if split in ["train", "valid"]:
                base_dirpath = "./data/doduo_small"
                if split not in small_tag:
                    small_tag += '_' + split
        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"

        if split in ["train", "valid"]:
            if small_tag == "":
                df_list = []
                for i in range(5):
                    if i == cv:
                        continue
                    filepath = os.path.join(base_dirpath, basename.format(i))
                    df_list.append(pd.read_csv(filepath))
                df = pd.concat(df_list, axis=0)
            else:
                filepath = os.path.join(base_dirpath, basename.format(str(cv) + '_' + small_tag))
                df = pd.read_csv(filepath)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)
        self.df = df
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)
        
        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if small_tag == "":
                if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                    break
                if split == "valid" and i < valid_index:
                    continue
            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(list(group_df["class_id"].values)) - 1)
            else:
                cur_maxlen = max(1, max_length // len(list(group_df["class_id"].values)) - 1)
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}


class SatoCVTablewiseDatasetWithDA(data.Dataset):
    """Table dataset for finetuning semantic type detection on the SATO dataset under semi-supervised setting"""
    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "/efs/task_datasets/doduo/",
            all_tables_filename: str = "sato{}.processed.pkl",
            small_tag: str = "",
            augment_op: str = "",
            num_aug: int = 1):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "unlabeled"], "split must be train or unlabeled"
        if small_tag != '':
            base_dirpath_small = "./data/doduo_small"
        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"
        
        self.augment_op = augment_op
        self.num_aug = num_aug
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        df_train = pd.read_csv(os.path.join(base_dirpath_small, basename.format(str(cv) + '_' + small_tag + '_train')))
        if split == 'train':
            raw_table_basename = "sato_cv_{}_train.processed.pkl" 
            self.table_dict = pickle.load(open(os.path.join(base_dirpath_small, raw_table_basename.format(str(cv) + '_' + small_tag)), 'rb'))
            self.tables = list(self.table_dict.values())
        elif split == 'unlabeled':
            df_valid = pd.read_csv(os.path.join(base_dirpath_small, basename.format(str(cv) + '_' + small_tag + '_valid')))
            labeled_table_id_dict = {}
            for i, tid in enumerate(df_train['table_id']):
                labeled_table_id_dict[tid] = True
            for i, tid in enumerate(df_valid['table_id']):
                labeled_table_id_dict[tid] = True    
            raw_tables = pickle.load(open(base_dirpath + all_tables_filename.format(cv), 'rb'))
            self.table_dict = {k: v for k, v in raw_tables.items() if k not in labeled_table_id_dict}
            self.tables = list(self.table_dict.values())
        print(len(self.table_dict))
       
    def __len__(self):
        return len(self.table_dict)
    
    def tokenize_one_table(self, table):
        num_cols = len(table['table_content'])
        if self.max_length <= 128:
            cur_maxlen = min(self.max_length, 512 // num_cols - 1)
        else:
            cur_maxlen = max(1, self.max_length // num_cols - 1)
        df = table['raw_table_content']
        token_ids_list = [self.tokenizer.encode(self.tokenizer.cls_token + " " + " ".join([str(_) for _ in df[x].dropna().tolist()]), 
                                                add_special_tokens=False, max_length=cur_maxlen, truncation=True) for x in df.columns]
        
        token_ids = torch.LongTensor(reduce(operator.add,
                                            token_ids_list)).to(self.device)
        cls_index_list = [0] + np.cumsum(
            np.array([len(x) for x in token_ids_list])).tolist()[:-1]
        for cls_index in cls_index_list:
            assert token_ids[
                cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
        cls_indexes = torch.LongTensor(cls_index_list).to(self.device)
        labels = [table['label_dict'][_] for _ in range(num_cols)]
        class_ids = torch.LongTensor(labels).to(self.device)
        return token_ids, cls_indexes, class_ids

    def __getitem__(self, idx):
        table_ori = self.tables[idx]

        # apply the augmentation operator
        op_list = self.augment_op.split(',')
        if len(op_list) == 1 and self.num_aug > 1:
            op_list = [self.augment_op for _ in range(self.num_aug)]
            
        table_aug = []
        for op in op_list:
            table_tmp = copy.deepcopy(table_ori)
            if op in ['dropout', '', 'None', None]:
                table_aug.append(table_tmp)
            else:
                table_tmp['raw_table_content'] = augment(table_tmp['raw_table_content'], op)
                table_aug.append(table_tmp)
        # convert table into string        
        x_ori, cls_indexes_ori, class_ids_ori = self.tokenize_one_table(table_ori)
        x_aug = []
        cls_indexes_aug = []
        class_ids_aug = []
        for table in table_aug:
            x, cls_indexes, class_ids = self.tokenize_one_table(table)
            x_aug.append(x)
            cls_indexes_aug.append(cls_indexes)
            class_ids_aug.append(class_ids)
        return {"ori": x_ori, "cls_indexes": cls_indexes_ori, "label_ori": class_ids_ori, "aug": x_aug, "cls_indexes_aug": cls_indexes_aug, "label_aug": class_ids_aug}

class TURLColTypeTablewiseDataset(data.Dataset):
    """Table dataset for finetuning and evaluating semantic type detection on the TURL dataset"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None,
                 small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)
        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break
    
            if split == "train" and len(group_df) > max_colnum:
                continue
            label_ids = group_df["label_ids"].tolist()
            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(label_ids) -1)
            else:
                cur_maxlen = max(1, max_length // len(label_ids) - 1)
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(label_ids).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        print('# Tables:', len(data_list))
        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }


class TURLColTypeTablewiseDatasetWithDA(data.Dataset):
    """Table dataset for finetuning semantic type detection on the TURL dataset under semi-supervised setting"""

    def __init__(
            self,
            filepath: str,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            all_tables_filename: str = "table_col_type_serialized.alltrain.processed.pkl",
            base_dirpath: str = "/efs/task_datasets/TURL/",
            augment_op: str = "",
            num_aug: int = 1,
            num_classes: int = 255):
        
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "unlabeled"], "split must be train or unlabeled"
        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

      
        self.augment_op = augment_op
        self.num_aug = num_aug
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.num_classes = num_classes
        self.split = split
        if split == 'train':
            with open(filepath.replace('.pkl', '.processed.pkl'), "rb") as fin_raw:
                self.table_dict = pickle.load(fin_raw)
                self.tables = list(self.table_dict[split].values())
              
        elif split == 'unlabeled':
            labeled_table_id_dict = {}
            for s in ['train', 'dev']:
                for i, tid in enumerate(df_dict[s]['table_id']):
                    labeled_table_id_dict[tid] = True
            raw_tables = pickle.load(open(base_dirpath + all_tables_filename, 'rb'))
            self.table_dict = {k: v for k, v in raw_tables.items() if k not in labeled_table_id_dict}
            self.tables = list(self.table_dict.values())
        self.tables = self.tables
        print(len(self.tables))
        
        
        
    def __len__(self):
        return len(self.tables)
    
    def tokenize_one_table(self, table):
        num_cols = len(table['table_content'])
        if self.max_length <= 128:
            cur_maxlen = min(self.max_length, 512 // num_cols - 1)
        else:
            cur_maxlen = max(1, self.max_length // num_cols - 1)

        df = table['raw_table_content']
        token_ids_list = [self.tokenizer.encode(self.tokenizer.cls_token + " " + " ".join([str(_) for _ in df[x].dropna().tolist()]), 
                                                add_special_tokens=False, max_length=cur_maxlen, truncation=True) for x in df.columns]
        
        token_ids = torch.LongTensor(reduce(operator.add,
                                            token_ids_list)).to(self.device)
        cls_index_list = [0] + np.cumsum(
            np.array([len(x) for x in token_ids_list])).tolist()[:-1]
        for cls_index in cls_index_list:
            assert token_ids[
                cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
        cls_indexes = torch.LongTensor(cls_index_list).to(self.device)
        labels = [[1 if __ in table['label_dict'][_] else 0 for __ in range(self.num_classes)] for _ in range(num_cols)]
        class_ids = torch.LongTensor(labels).to(self.device)
        return token_ids, cls_indexes, class_ids

    def __getitem__(self, idx):
        
        table_ori = self.tables[idx]
        # apply the augmentation operator
        op_list = self.augment_op.split(',')
        if len(op_list) == 1 and self.num_aug > 1:
            op_list = [self.augment_op for _ in range(self.num_aug)]
        table_aug = []
        for op in op_list:
            table_tmp = copy.deepcopy(table_ori)
            if op in ['dropout', '', 'None', None]:
                table_aug.append(table_tmp)
            else:
                table_tmp['raw_table_content'] = augment(table_tmp['raw_table_content'], op)
                table_aug.append(table_tmp)
        # convert table into string        
        x_ori, cls_indexes_ori, class_ids_ori = self.tokenize_one_table(table_ori)
        x_aug = []
        cls_indexes_aug = []
        class_ids_aug = []
        for table in table_aug:
            x, cls_indexes, class_ids = self.tokenize_one_table(table)
            x_aug.append(x)
            cls_indexes_aug.append(cls_indexes)
            class_ids_aug.append(class_ids)         
        
        return {"ori": x_ori, "cls_indexes": cls_indexes_ori, "label_ori": class_ids_ori, "aug": x_aug, "cls_indexes_aug": cls_indexes_aug, "label_aug": class_ids_aug}


class TURLRelExtTablewiseDataset(data.Dataset):
    """Table dataset for finetuning and evaluating relationship extraction on the TURL dataset"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]

        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)
        # num_train = 1000

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # It's probably already sorted but just in case.
            group_df = group_df.sort_values("column_id")

            if split == "train" and len(group_df) > max_colnum:
                continue
            label_ids = group_df["label_ids"].tolist()

            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(label_ids) - len(label_ids))
            else:
                cur_maxlen = max(1, max_length // len(label_ids) - len(label_ids))
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(label_ids).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }


class TURLRelExtTablewiseDatasetWithDA(data.Dataset):
    """Table dataset for finetuning relationship extraction on the TURL dataset under semi-supervised setting"""

    def __init__(
            self,
            filepath: str,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            all_tables_filename: str = "table_rel_extraction_serialized.alltrain.processed.pkl",
            base_dirpath: str = "/efs/task_datasets/TURL/",
            augment_op: str = "",
            num_aug: int = 1,
            num_classes: int = 121):
        
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "unlabeled"], "split must be train or unlabeled"
        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)
      
        self.augment_op = augment_op
        self.num_aug = num_aug
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.num_classes = num_classes
        self.split = split
        if split == 'train':
            with open(filepath.replace('.pkl', '.processed.pkl'), "rb") as fin_raw:
                self.table_dict = pickle.load(fin_raw)
                self.tables = list(self.table_dict[split].values())

        elif split == 'unlabeled':
            labeled_table_id_dict = {}
            for s in ['train', 'dev']:
                for i, tid in enumerate(df_dict[s]['table_id']):
                    labeled_table_id_dict[tid] = True
            raw_tables = pickle.load(open(base_dirpath + all_tables_filename, 'rb'))
            self.table_dict = {k: v for k, v in raw_tables.items() if k not in labeled_table_id_dict}
            self.tables = list(self.table_dict.values())
            
        print(len(self.tables))    
        
    def __len__(self):
        return len(self.tables)
    
    def tokenize_one_table(self, table):
        num_cols = len(table['table_content'])
        if self.max_length <= 128:
            cur_maxlen = min(self.max_length, 512 // num_cols - 1)
        else:
            cur_maxlen = max(1, self.max_length // num_cols - 1)
        df = table['raw_table_content']
        token_ids_list = [self.tokenizer.encode(self.tokenizer.cls_token + " " + " ".join([str(_) for _ in df[x].dropna().tolist()]), 
                                                add_special_tokens=False, max_length=cur_maxlen, truncation=True) for x in df.columns]
        
        token_ids = torch.LongTensor(reduce(operator.add,
                                            token_ids_list)).to(self.device)
        cls_index_list = [0] + np.cumsum(
            np.array([len(x) for x in token_ids_list])).tolist()[:-1]
        for cls_index in cls_index_list:
            assert token_ids[ 
                cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
        cls_indexes = torch.LongTensor(cls_index_list).to(self.device)
        if self.split == 'unlabeled':
            labels = [[1 if __ in table['label_dict'][_] else 0 for __ in range(self.num_classes)] for _ in range(1, num_cols)]
        else:
            labels = [[1 if __ in table['label_dict'][_] else 0 for __ in range(self.num_classes)] for _ in range(num_cols)]

        class_ids = torch.LongTensor(labels).to(self.device)
        return token_ids, cls_indexes, class_ids

    def __getitem__(self, idx):
        
        table_ori = self.tables[idx]
        # apply the augmentation operator
        op_list = self.augment_op.split(',')
        if len(op_list) == 1 and self.num_aug > 1:
            op_list = [self.augment_op for _ in range(self.num_aug)]
            
        table_aug = []
        for op in op_list:
            table_tmp = copy.deepcopy(table_ori)
            if op in ['dropout', '', 'None', None]:
                table_aug.append(table_tmp)
            else:
                table_tmp['raw_table_content'] = augment(table_tmp['raw_table_content'], op)
                table_aug.append(table_tmp)
        # convert table into string        
        x_ori, cls_indexes_ori, class_ids_ori = self.tokenize_one_table(table_ori)
        x_aug = []
        cls_indexes_aug = []
        class_ids_aug = []
        for table in table_aug:
            x, cls_indexes, class_ids = self.tokenize_one_table(table)
            x_aug.append(x)
            cls_indexes_aug.append(cls_indexes)
            class_ids_aug.append(class_ids)
            
        return {"ori": x_ori, "cls_indexes": cls_indexes_ori, "label_ori": class_ids_ori, "aug": x_aug, "cls_indexes_aug": cls_indexes_aug, "label_aug": class_ids_aug}

class ColPoplTablewiseDataset(data.Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 use_content: bool = True,
                 use_metadata: bool = False,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

    
        data = load_jsonl(filepath, n_rows=1000000)

        self.df = pd.DataFrame.from_dict(data)
        num_tables = len(self.df)
        num_train = int(train_ratio * num_tables)
        data_list = []
        for i, line in self.df.iterrows():
            if i >= num_train:
                break
            table = line['table_data']
            col_num = len(table[0])

            if split == "train" and col_num > max_colnum:
                continue
            label_str = line["label"]
            label_ids = line["label_idx"]
            for j in range(len(label_ids)):
                if label_ids[j] >= 127656:
                    label_ids[j] = 0

            if max_length <= 128:
                if col_num > 0:
                    cur_maxlen = min(max_length, 512 // col_num - col_num)
                else:
                    cur_maxlen = max_length
            else:
                cur_maxlen = max(1, max_length // col_num - col_num)
            if use_metadata and use_content:
                columns = [' '.join(line['orig_header'][j].split() + [table[_][j] for _ in range(len(table))]) for j in range(col_num)]
            elif use_metadata:
                columns = [' '.join(line['orig_header'][j].split()) for j in range(col_num)]
            elif use_content:
                columns = [' '.join([table[_][j] for _ in range(len(table))]) for j in range(col_num)]
            if use_metadata:
                pgTitle = line['pgTitle']
                secTitle = line['sectionTitle']
                caption = line['caption']
                metadata = pgTitle.split() + secTitle.split()
                if caption != secTitle:
                    metadata += caption.split()
                columns= [' '.join(metadata)] + columns
            token_ids_list = list(map(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True), columns))
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(label_ids).to(device)
            data_list.append(
                [i, col_num, token_ids, class_ids, label_str, cls_indexes])
       
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "idx", "num_col", "data_tensor",
                                         "label_tensor", "label_str", "cls_indexes"
                                     ])
        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "idx": self.table_df.iloc[idx]["idx"],
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"]
        }


class SotabCVTablewiseDataset(data.Dataset):
    """Table dataset for finetuning and evaluating semantic type detection on the Sotab dataset"""

    def __init__(
            self,
            # cv: int,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            gt_only: bool = False,
            max_length: int = 256,
            # multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "/data/yongkang/TU/SOTAB",
            small_tag: str = "",
            label_encoder: LabelEncoder = None,
            max_unlabeled=8,
            random_sample=False # TODO
            ):
        # ?
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid", "test"]
        if split == "train":
            gt_df = pd.read_csv(os.path.join(base_dirpath, "CTA_training_small_gt.csv"))
            data_folder = "Train"
        elif split == "valid":
            gt_df = pd.read_csv(os.path.join(base_dirpath, "CTA_validation_gt.csv"))
            data_folder = "Validation"
        else:  # test
            gt_df = pd.read_csv(os.path.join(base_dirpath, "CTA_test_gt.csv"))
            data_folder = "Test"

        # 初始化或加载 LabelEncoder
        if label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder.fit(gt_df['label'])
        self.label_encoder = label_encoder

        # gt_set = set(zip(gt_df["table_name"], gt_df["column_index"]))

        table_files = [f for f in os.listdir(os.path.join(base_dirpath, data_folder)) if f in gt_df['table_name'].values]

        mapping_file_path = os.path.join(base_dirpath, "label_mapping.txt")
        df_csv_path = os.path.join(base_dirpath, f"comma_{split}_fully_deduplicated_sotab.csv")

        # 检查是否存在之前保存的标签映射关系
        if os.path.exists(mapping_file_path):
            # 如果存在，直接从文件中读取映射
            with open(mapping_file_path, 'r') as f:
                label_mapping = json.load(f)
            next_label_id = max(label_mapping.values()) + 1
            print(f"标签映射关系从 {mapping_file_path} 读取成功")
        else:
            raise FileNotFoundError(f"{mapping_file_path} 文件不存在，请确认文件路径")

        # 检查 CSV 文件是否存在
        if os.path.exists(df_csv_path):
            # 直接从 CSV 文件中读取数据
            df = pd.read_csv(df_csv_path)
            print(f"数据已从 {df_csv_path} 加载")
        else:
            raise FileNotFoundError(f"{df_csv_path} 文件不存在，请确认文件路径")
        
        if split != "train" and gt_only:
            df = df[df["label"] > -1]

        data_list = []

        df['label'] = df['label'].astype(int)
        df.drop(df[(df['data'].isna()) & (df['label'] == -1)].index, inplace=True)
        df['column_index'] = df['column_index'].astype(int)
        df['data'] = df['data'].astype(str)

        print("start encoder")
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if split == "train" and ((i >= num_train) or (i >= valid_index)):
            #     break
            # if split == "valid" and i < valid_index:
            #     continue

            labeled_columns = group_df[group_df['label'] > -1]
            unlabeled_columns = group_df[group_df['label'] == -1]
            num_unlabeled = min(max(max_unlabeled-len(labeled_columns), 0), len(unlabeled_columns))
            unlabeled_columns = unlabeled_columns.sample(num_unlabeled) if random_sample else unlabeled_columns[0:num_unlabeled]
            group_df = pd.concat([group_df[group_df['label'] > -1], unlabeled_columns]) # TODO
            group_df.sort_values(by=['column_index'], inplace=True)

            num_labels = len(list(group_df["label"].values))
            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // num_labels - 1)
            else:
                cur_maxlen = max(1, max_length // num_labels - 1)

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)

            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            labels = torch.LongTensor(group_df["label"].values).to(device)

            data_list.append(
                [index,
                 len(group_df), token_ids, labels, cls_indexes])
                # 每处理1000条数据时输出一个标识和最近处理的 DataFrame
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} entries. Recent dataframe sample:")
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of singesle-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}


class SemtableCVTablewiseDataset(data.Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "./data/semtable2019",
            small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"
        if split in ["train", "valid"]:
            if multicol_only:
                basename = "mreca_all_{}_fold_{}.csv"
            else:
                basename = "reca_all_{}_fold_{}.csv"
        else:
            if multicol_only:
                basename = "mreca_all_{}.csv"
            else:
                basename = "reca_all_{}.csv"

        filepath = os.path.join(base_dirpath, basename.format(split, cv))
    
        df = pd.read_csv(filepath)
        df = df[df["class_id"] > -1]
        
        num_tables = len(df.groupby("table_id"))
        
        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if len(data_list) > 500:
            #     break

            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(list(group_df["class_id"].values)) - 1)
            else:
                cur_maxlen = max(1, max_length // len(list(group_df["class_id"].values)) - 1)
                
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
        
class SemtableCVColwiseDataset(data.Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "./data/semtable2019",
            small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"
         
        if split in ["train", "valid"]:
            if multicol_only:
                basename = "mreca_all_{}_fold_{}.csv"
            else:
                basename = "reca_all_{}_fold_{}.csv"
        else:
            if multicol_only:
                basename = "mreca_all_{}.csv"
            else:
                basename = "reca_all_{}.csv"

        filepath = os.path.join(base_dirpath, basename.format(split, cv))
    
        df = pd.read_csv(filepath)
        df = df[df["class_id"] > -1]
        
        num_tables = len(df.groupby("table_id"))
        
        row_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if len(data_list) > 500:
            #     break
            for _, row in group_df.iterrows():
                row_list.append(row)


        self.df = pd.DataFrame(row_list)

        # Convert into torch.Tensor
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(x,
                                 add_special_tokens=True,
                                 max_length=max_length + 2)).to(device))
        self.df["label_tensor"] = self.df["class_id"].apply(
            lambda x: torch.LongTensor([x]).to(device)
        )  # Can we reduce the size?

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }
        

class GittablesTablewiseDataset(data.Dataset):

    def __init__(
            self,
            cv: int,
            split: str,
            src: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "/data/zhihao/TU/GitTables/semtab_gittables/2022",
            base_tag: str = '', # blank, comma
            small_tag: str = "",
            train_ratio: float = 1.0,
            max_unlabeled=8,
            random_sample=False, # TODO
            train_only=False,
            seed=None): # TODO
        if device is None:
            device = torch.device('cpu')
        basename = small_tag+ "_cv_{}.csv"
    
        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
                print(split, i)
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)
            print(split)


        if gt_only:
            df = df[df["class_id"] > -1]
        if train_only and split != "train":
            df = df[df["class_id"] > -1]

        
        data_list = []
        
        df['class_id'] = df['class_id'].astype(int)
        df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
        df['col_idx'] = df['col_idx'].astype(int)
        df['data'] = df['data'].astype(str)
        
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)        
        
        # df.drop(df[(df['data'] == '') & (df['class_id'] == -1)].index, inplace=True)
        total_num_cols = 0
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue
            #     break
            labeled_columns = group_df[group_df['class_id'] > -1]
            unlabeled_columns = group_df[group_df['class_id'] == -1]
            num_unlabeled = min(max(max_unlabeled-len(labeled_columns), 0), len(unlabeled_columns)) if max_unlabeled > 0 else len(unlabeled_columns)
            unlabeled_columns = unlabeled_columns.sample(num_unlabeled, random_state=seed) if random_sample else unlabeled_columns[0:num_unlabeled]
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns.sample(min(10-len(labeled_columns), len(unlabeled_columns)))])
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns[0:min(max(10-len(labeled_columns), 0), len(unlabeled_columns))]])
            group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns]) # TODO
            group_df.sort_values(by=['col_idx'], inplace=True)

            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(list(group_df["class_id"].values)) - 1)
            else:
                cur_maxlen = max(1, max_length // len(list(group_df["class_id"].values)) - 1)
                
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}





from watchog.utils import preprocess_text
from rank_bm25 import BM25Okapi

class GittablesColwiseDataset(data.Dataset):
    def __init__(
            self,
            cv: int,
            split: str,
            tokenizer: AutoTokenizer,
            max_length: int = 64,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "/data/zhihao/TU/GitTables/semtab_gittables/2022",
            base_tag: str = '', # blank, comma
            small_tag: str = "",
            train_ratio: float = 1.0,
            max_num_col=8,
            sampling_method=None, # TODO
            random_seed=0,
            train_only=False,
            context_encoding_type="v1",
            adaptive_max_length=False,
            ): # TODO: not used
        if device is None:
            device = torch.device('cpu')
        basename = small_tag+ "_cv_{}.csv"
        print(basename,  max_num_col, max_length, sampling_method, context_encoding_type, adaptive_max_length)
        assert max_num_col >= 1, "max_num_col must be greater than 1"
        # assert 512//max_num_col >= max_length  , "max_length cannot be smaller than 512//max_num_col"
        # assert random_sample and split == "train" can't be true at the same time
        # assert not (sampling_method=="random" and split != "train"), "random_sample and split != train can't be true at the same time"
        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
                # print(split, i)
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)
            # print(split)


        if gt_only or max_num_col == 1:
            df = df[df["class_id"] > -1]


        
        data_list = []
        
        df['class_id'] = df['class_id'].astype(int)
        df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
        df['col_idx'] = df['col_idx'].astype(int)
        df['data'] = df['data'].astype(str)
        
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)        
        
        # df.drop(df[(df['data'] == '') & (df['class_id'] == -1)].index, inplace=True)
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue
            #     break
            
            group_df.sort_values(by=['col_idx'], inplace=True)
            labeled_columns = group_df[group_df['class_id'] > -1]
            # if len(labeled_columns) > 1:
            #     print("Here")
            unlabeled_columns = group_df[group_df['class_id'] == -1]
            # Preprocessing for bm25, sentenceBERT sampling
            if sampling_method == "bm25":
                processed_texts = group_df["data"].apply(lambda x: preprocess_text(x)).to_list()
                processed_texts = [x.split() for x in processed_texts]
                try:
                    bm25 = BM25Okapi(processed_texts)
                except:
                    processed_texts = group_df["data"].apply(lambda x: x.split()).to_list()
                    bm25 = BM25Okapi(processed_texts)
            
            for idx in labeled_columns.index:
                token_ids_list = []
                target_column = labeled_columns.loc[idx]
                rest_columns = labeled_columns[labeled_columns.index != idx] # the rest of labeled columns
                # tokenize target column
                if context_encoding_type == "v1.2": # padding each column to make sure that each column is of equal length
                    token_ids_list.append(tokenizer.encode(
                    tokenizer.cls_token + " " + target_column["data"] + "[PAD]"*512, add_special_tokens=False, max_length=max_length, truncation=True))                    
                else:
                    token_ids_list.append(tokenizer.encode(
                    tokenizer.cls_token + " " + target_column["data"], add_special_tokens=False, max_length=max_length, truncation=True))
                # tokenize context
                if not gt_only:
                    unlabeled_columns = unlabeled_columns
                else:
                    unlabeled_columns  = unlabeled_columns.sample(0)
                temp_group_df = pd.concat([rest_columns, unlabeled_columns])
                
                # Sampling context columns
                if sampling_method is None or sampling_method == "None":
                    temp_group_df = temp_group_df[: max_num_col-1]
                    temp_group_df.sort_values(by=['col_idx'], inplace=True)
                elif sampling_method == "random":
                    temp_group_df = temp_group_df.sample(min(max_num_col-1, len(temp_group_df)), random_state=random_seed)
                    temp_group_df.sort_values(by=['col_idx'], inplace=True)
                elif sampling_method == "bm25":
                    target_text = group_df["data"].loc[idx]
                    target_text = preprocess_text(target_text).split()
                    temp_group_df = group_df.iloc[torch.tensor(bm25.get_scores(target_text)).argsort(descending=True, stable=True)[1:max_num_col].tolist()]
                else:
                    raise ValueError("sampling_method {} is not supported".format(sampling_method))
                    
                    
                    
                
                
                if adaptive_max_length and (len(temp_group_df)+1) < max_num_col:
                    cur_maxlen = 512 // (len(temp_group_df)+1)
                else:
                    cur_maxlen = max_length
                    
                if context_encoding_type == "v0":
                    token_ids_list += temp_group_df["data"].apply(lambda x: tokenizer.encode(
                    tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                    )
                elif context_encoding_type == "v0.1": # only one cls token to separate context and target
                    token_ids_list.append([tokenizer.cls_token_id])
                    token_ids_list += temp_group_df["data"].apply(lambda x: tokenizer.encode(
                    " " + x, add_special_tokens=False, max_length=cur_maxlen-1, truncation=True)).tolist(
                    )                    
                elif context_encoding_type == "v1":
                    token_ids_list += temp_group_df["data"].apply(lambda x: tokenizer.encode(
                    tokenizer.sep_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                    )                    
                elif context_encoding_type == "v1.1": # only one cls token to separate context and target
                    token_ids_list.append([tokenizer.sep_token_id])
                    token_ids_list += temp_group_df["data"].apply(lambda x: tokenizer.encode(
                    " " + x, add_special_tokens=False, max_length=cur_maxlen-1, truncation=True)).tolist(
                    )   
                elif context_encoding_type == "v1.2": # padding each column to make sure that each column is of equal length
                    assert len(token_ids_list[0]) == max_length
                    token_ids_list += temp_group_df["data"].apply(lambda x: tokenizer.encode(
                    tokenizer.sep_token + " " + x + "[PAD]"*512, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                    )                  
                else:
                    raise ValueError("context_encoding_type {} is not supported".format(context_encoding_type))
                token_ids = torch.LongTensor(reduce(operator.add,
                                                    token_ids_list)).to(device)
                token_type_ids = torch.LongTensor([0 if i < max_length else 1 for i in range(len(token_ids))]).to(device)
                cls_index_list = [0] 
                for cls_index in cls_index_list:
                    assert token_ids[
                        cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                class_ids = torch.LongTensor([target_column["class_id"]]
                    ).to(device)
                data_list.append(
                    [index,
                    len(temp_group_df), token_ids, class_ids, cls_indexes, token_type_ids])
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "token_type_ids"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "token_type_ids": self.table_df.iloc[idx]["token_type_ids"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
        
from sentence_transformers import SentenceTransformer
class GittablesColwiseMaxDataset(data.Dataset):
    def __init__(
            self,
            cv: int,
            split: str,
            tokenizer: AutoTokenizer,
            max_length: int = 64,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "/data/zhihao/TU/GitTables/semtab_gittables/2022",
            base_tag: str = '', # blank, comma
            small_tag: str = "",
            train_ratio: float = 1.0,
            max_num_col=8,
            random_sample=False, # TODO
            train_only=False,
            context_encoding_type="v1",
            adaptive_max_length=False,
            return_table_embedding=False,
            ): # TODO: not used
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.return_table_embedding = return_table_embedding
        if return_table_embedding:
            setence_encoder = SentenceTransformer("all-mpnet-base-v2", cache_folder="/data/zhihao/TU/Watchog/sentence_transformers_cache", device=device)
        basename = small_tag+ "_cv_{}.csv"
        print(basename,  max_num_col)
        assert max_num_col >= 1, "max_num_col must be greater than 1"
        # assert 512//max_num_col >= max_length  , "max_length cannot be smaller than 512//max_num_col"
        # assert random_sample and split == "train" can't be true at the same time
        assert not (random_sample and split != "train"), "random_sample and split != train can't be true at the same time"
        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
                # print(split, i)
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)
            # print(split)


        if gt_only or max_num_col == 1:
            df = df[df["class_id"] > -1]


        
        data_list = []
        
        df['class_id'] = df['class_id'].astype(int)
        df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
        df['col_idx'] = df['col_idx'].astype(int)
        df['data'] = df['data'].astype(str)
        
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)        
        
        # df.drop(df[(df['data'] == '') & (df['class_id'] == -1)].index, inplace=True)
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue
            #     break
            group_df.sort_values(by=['col_idx'], inplace=True)
            labeled_columns = group_df[group_df['class_id'] > -1]
            # if len(labeled_columns) > 1:
            #     print("Here")
            unlabeled_columns = group_df[group_df['class_id'] == -1]
            for idx in labeled_columns.index:
                token_ids_list = []
                target_column = labeled_columns.loc[idx]
                rest_columns = labeled_columns[labeled_columns.index != idx] # the rest of labeled columns
                # tokenize target column
                if context_encoding_type == "v1.2": # padding each column to make sure that each column is of equal length
                    token_ids_list.append(tokenizer.encode(
                    tokenizer.cls_token + " " + target_column["data"] + "[PAD]"*512, add_special_tokens=False, max_length=max_length, truncation=True))                    
                else:
                    token_ids_list.append(tokenizer.encode(
                    tokenizer.cls_token + " " + target_column["data"], add_special_tokens=False, max_length=max_length, truncation=True))
                
                # tokenize context
                if not gt_only:
                    unlabeled_columns = unlabeled_columns
                else:
                    unlabeled_columns  = unlabeled_columns.sample(0)
                group_df = pd.concat([rest_columns, unlabeled_columns])
                group_df = group_df[: max_num_col-1] if not random_sample  else group_df.sample(min(max_num_col-1, len(group_df))) # TODO
                group_df.sort_values(by=['col_idx'], inplace=True)
                
                if adaptive_max_length and (len(group_df)+1) < max_num_col:
                    cur_maxlen = 512 // (len(group_df)+1)
                else:
                    cur_maxlen = max_length
                    
                if context_encoding_type == "v0":
                    token_ids_list += group_df["data"].apply(lambda x: tokenizer.encode(
                    tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                    )
                elif context_encoding_type == "v0.1": # only one cls token to separate context and target
                    token_ids_list.append([tokenizer.cls_token_id])
                    token_ids_list += group_df["data"].apply(lambda x: tokenizer.encode(
                    " " + x, add_special_tokens=False, max_length=cur_maxlen-1, truncation=True)).tolist(
                    )                    
                elif context_encoding_type == "v1":
                    token_ids_list += group_df["data"].apply(lambda x: tokenizer.encode(
                    tokenizer.sep_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                    )                    
                elif context_encoding_type == "v1.1": # only one cls token to separate context and target
                    token_ids_list.append([tokenizer.sep_token_id])
                    token_ids_list += group_df["data"].apply(lambda x: tokenizer.encode(
                    " " + x, add_special_tokens=False, max_length=cur_maxlen-1, truncation=True)).tolist(
                    )   
                elif context_encoding_type == "v1.2": # padding each column to make sure that each column is of equal length
                    assert len(token_ids_list[0]) == max_length
                    token_ids_list += group_df["data"].apply(lambda x: tokenizer.encode(
                    tokenizer.sep_token + " " + x + "[PAD]"*512, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                    )                  
                else:
                    raise ValueError("context_encoding_type {} is not supported".format(context_encoding_type))
                if len(token_ids_list) < max_num_col:
                    for _ in range(max_num_col - len(token_ids_list)):
                        token_ids_list.append([tokenizer.pad_token_id]*max_length)
                token_ids = torch.LongTensor(reduce(operator.add,
                                                    token_ids_list)).to(device)
                token_type_ids = torch.LongTensor([0 if i < max_length else 1 for i in range(len(token_ids))]).to(device)
                cls_index_list = [0] 
                for cls_index in cls_index_list:
                    assert token_ids[
                        cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                class_ids = torch.LongTensor([target_column["class_id"]]
                    ).to(device)
                if return_table_embedding:
                    target_embedding = setence_encoder.encode([target_column["data"]])

                    if len(group_df) > 0:
                        context_embedding = setence_encoder.encode(group_df["data"].to_list())
                    else:
                        context_embedding = np.zeros_like(target_embedding)
                    if len(context_embedding) < max_num_col-1:
                        context_embedding = np.concatenate([context_embedding, np.zeros((max_num_col-1-len(context_embedding), target_embedding.shape[1]))], axis=0)

                    table_embedding = torch.tensor(np.concatenate([target_embedding, context_embedding], axis=0)).float()
                    data_list.append(
                        [index,
                        len(group_df), token_ids, class_ids, cls_indexes, token_type_ids, table_embedding])
                else:
                    data_list.append(
                        [index,
                        len(group_df), token_ids, class_ids, cls_indexes, token_type_ids])
        print(split, len(data_list))
        if return_table_embedding:
            self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "token_type_ids", "table_embedding"
                                     ])
        else:
            self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "token_type_ids"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        results = {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "token_type_ids": self.table_df.iloc[idx]["token_type_ids"]
        }
        if self.return_table_embedding:
            results["table_embedding"] = self.table_df.iloc[idx]["table_embedding"].to(self.device)
        return results
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
        
        
        
class GittablesColwiseDatasetDecoder(data.Dataset):
    """This is for decoder-only archetecture LMs like Llama.
    The input first is the context, seperated by eos token, the last tokens are the target column.
    """
    def __init__(
            self,
            cv: int,
            split: str,
            src: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "/data/zhihao/TU/GitTables/semtab_gittables/2022",
            base_tag: str = '', # blank, comma
            small_tag: str = "",
            train_ratio: float = 1.0,
            max_num_col=8,
            random_sample=False, # TODO
            train_only=False,
            context_encoding_type="v1"
            ): # TODO: not used
        if device is None:
            device = torch.device('cpu')
        basename = small_tag+ "_cv_{}.csv"
        assert max_num_col >= 1, "max_num_col must be greater than 1"
        assert 512//max_num_col >= max_length  , "max_length cannot be smaller than 512//max_num_col"
        # assert random_sample and split == "train" can't be true at the same time
        assert not (random_sample and split != "train"), "random_sample and split != train can't be true at the same time"
        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
                print(split, i)
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)
            print(split)


        if gt_only or max_num_col == 1:
            df = df[df["class_id"] > -1]


        
        data_list = []
        
        df['class_id'] = df['class_id'].astype(int)
        df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
        df['col_idx'] = df['col_idx'].astype(int)
        df['data'] = df['data'].astype(str)
        
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)        
        
        # df.drop(df[(df['data'] == '') & (df['class_id'] == -1)].index, inplace=True)
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue
            #     break
            group_df.sort_values(by=['col_idx'], inplace=True)
            labeled_columns = group_df[group_df['class_id'] > -1]
            # if len(labeled_columns) > 1:
            #     print("Here")
            unlabeled_columns = group_df[group_df['class_id'] == -1]
            for idx in labeled_columns.index:
                token_ids_list = []
                target_column = labeled_columns.loc[idx]
                rest_columns = labeled_columns[labeled_columns.index != idx] # the rest of labeled columns
                # tokenize context first
                if not gt_only:
                    unlabeled_columns = unlabeled_columns
                else:
                    unlabeled_columns  = unlabeled_columns.sample(0)
                group_df = pd.concat([rest_columns, unlabeled_columns])
                group_df = group_df[: max_num_col-1] if not random_sample  else group_df.sample(min(max_num_col-1, len(group_df))) # TODO
                group_df.sort_values(by=['col_idx'], inplace=True)                
                if context_encoding_type == "v1":
                    # token_ids_list += [token_seq + [tokenizer.eos_token_id] for token_seq in group_df["data"].apply(lambda x: tokenizer.encode(
                    # tokenizer.bos_token + " " + x, add_special_tokens=False, max_length=max_length-1, truncation=True)).tolist(
                    # ) ]    
                    token_ids_list += group_df["data"].apply(lambda x: tokenizer.encode(
                    tokenizer.bos_token + " " + x, add_special_tokens=False, max_length=max_length, truncation=True)).tolist(
                    )
                else:
                    raise ValueError("context_encoding_type {} is not supported".format(context_encoding_type))
                # tokenize target column then
                token_ids_list.append(tokenizer.encode(
                tokenizer.bos_token + " " + target_column["data"], add_special_tokens=False, max_length=max_length, truncation=True))
                token_ids = torch.LongTensor(reduce(operator.add,
                                                    token_ids_list)).to(device)
                cls_index_list = [0] 
                for cls_index in cls_index_list:
                    assert token_ids[
                        cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                class_ids = torch.LongTensor([target_column["class_id"]]
                    ).to(device)
                data_list.append(
                    [index,
                    len(group_df), token_ids, class_ids, cls_indexes])
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
        
class GittablesCVTablewiseDataset(data.Dataset):

    def __init__(
            self,
            cv: int,
            split: str, # Train/valid
            src: str,  # 'dbpedia_property'
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "./data/gittables",
            small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

    
        basename = "{}_all_{}_cv{}.csv"

        filepath = os.path.join(base_dirpath, basename.format(src, split, cv))
    
        df = pd.read_csv(filepath)
        df['class_id'] = df['class_id'].astype(int)
        if gt_only:
            df = df[df["class_id"] > -1]
        
        df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
        df['col_idx'] = df['col_idx'].astype(int)
        df['data'] = df['data'].astype(str)
        # df.drop(df[(df['data'] == '') & (df['class_id'] == -1)].index, inplace=True)
        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if len(data_list) > 10:
            #     break
            labeled_columns = group_df[group_df['class_id'] > -1]
            unlabeled_columns = group_df[group_df['class_id'] == -1]
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns.sample(min(10-len(labeled_columns), len(unlabeled_columns)))])
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns[0:min(max(10-len(labeled_columns), 0), len(unlabeled_columns))]])
            group_df = pd.concat([labeled_columns, unlabeled_columns[0:min(max(8-len(labeled_columns), 0), len(unlabeled_columns))]]) # TODO
            group_df.sort_values(by=['col_idx'], inplace=True)

            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(list(group_df["class_id"].values)) - 1)
            else:
                cur_maxlen = max(1, max_length // len(list(group_df["class_id"].values)) - 1)
            # print(tokenizer.cls_token)
            # print(group_df["data"])
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + str(x), add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
        
        
from sklearn.cluster import KMeans
import torch.nn.functional as F
class GittablesTablewiseIterateClusterDataset(data.Dataset):

    def __init__(
            self,
            cv: int,
            split: str,
            src: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "/data/zhihao/TU/GitTables/semtab_gittables/2022",
            base_tag: str = '', # blank, comma
            small_tag: str = "",
            train_ratio: float = 1.0,
            max_unlabeled=8,
            adaptive_max_length=True,
            random_sample=False, # TODO
            train_only=False,
            model=None): # TODO
        if device is None:
            device = torch.device('cpu')
        basename = small_tag+ "_cv_{}.csv"
    
        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
                print(split, i)
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)
            print(split)


        if gt_only:
            df = df[df["class_id"] > -1]
        if train_only and split != "train":
            df = df[df["class_id"] > -1]

        
        data_list = []
        
        df['class_id'] = df['class_id'].astype(int)
        df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
        df['col_idx'] = df['col_idx'].astype(int)
        df['data'] = df['data'].astype(str)
        
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)        
        
        # df.drop(df[(df['data'] == '') & (df['class_id'] == -1)].index, inplace=True)
        total_num_cols = 0
        self.correctness_checklist = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue
            #     break
            
            group_df = group_df.reset_index(drop=True)
            # scores = bm25.get_scores(test_dataset_gt[idx]["data"].cpu().tolist())
            # semantic_scores = torch.tensor(group_df["data"].apply(lambda x: max(bm25.get_scores(tokenizer.encode(
            #         tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=max_length, truncation=True)))))
            
            # labeled_columns_index = group_df[group_df['class_id'] > -1].index.to_list()
            # assert len(labeled_columns_index) < 8
            # if set(labeled_columns_index).issubset(set(semantic_scores.topk(min(len(group_df), max_unlabeled)).indices.tolist())):
            #     self.correctness_checklist.append(1)
            # else:
            #     self.correctness_checklist.append(0)
            if len(group_df) > max_unlabeled:
                # col_embs = sb_model.encode(group_df["data"].to_list())
                
                with torch.no_grad():
                    model.eval()
                    model = model.to(device)
                    token_list = group_df["data"].apply(lambda x: tokenizer.encode(
                            tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=max_length, truncation=True)).tolist()
                    token_list = torch.nn.utils.rnn.pad_sequence(
                    [torch.LongTensor(tokens) for tokens in token_list], padding_value=0).T.to(device)
                    cls_indexes = torch.nonzero(
                                token_list == tokenizer.cls_token_id)
                    logits, embs = model(token_list, cls_indexes=cls_indexes, get_enc=True)
                    col_embs = embs.cpu()
                    if len(embs.size()) == 1:
                        col_embs = col_embs.unsqueeze(0)
                        
                kmeans = KMeans(n_clusters=max_unlabeled-1, random_state=0).fit(col_embs)
                centers = torch.tensor(kmeans.cluster_centers_)
                col_embs = torch.tensor(col_embs)
                cluster_labels = torch.tensor(kmeans.labels_)
                col_embs_cluster = []
                col_embs_cluster_index = []
                for cluster_i in np.unique(kmeans.labels_):
                    col_embs_cluster.append(col_embs[cluster_labels == cluster_i])
                    col_embs_cluster_index.append((cluster_labels == cluster_i).nonzero().reshape(-1))
                num_clusters = len(col_embs_cluster)
            labeled_columns = group_df[group_df['class_id'] > -1]
            labeled_columns.sort_values(by=['col_idx'], inplace=True)
            for index, target_column in labeled_columns.iterrows():
                target_col_idx = target_column["col_idx"]
                if len(group_df) > max_unlabeled:
                    closest_cols = []
                    for i in range(num_clusters):
                        # try:
                        sim = F.cosine_similarity(col_embs_cluster[i], col_embs[index].reshape(1, -1))
                        # except:
                        #     raise ValueError("here")
                        # try:
                        closest_index = col_embs_cluster_index[i][sim.argmax().item()].item()
                        # except:
                        #     print("here")
                        if closest_index == index and len(col_embs_cluster[i]) > 1:
                            closest_index = col_embs_cluster_index[i][sim.argsort(descending=True)[1].item()].item()
                        closest_cols.append(closest_index)
                    other_columns = group_df.iloc[closest_cols]
                else:
                    other_columns = group_df.drop(index)
                target_table = pd.concat([target_column.to_frame().T, other_columns], ignore_index=True)
                
                # if len(group_df) >= 8:
                #     assert len(target_table) == max_unlabeled
                # else:
                #     assert len(target_table) == len(group_df)
    
                
                target_cls = target_column["class_id"]
                target_table.sort_values(by=['col_idx'], inplace=True)
                col_idx_list = target_table["col_idx"].tolist()
                # target_table.sort_values(by=['col_idx'], inplace=True)

                if max_length <= 128 and adaptive_max_length:
                    cur_maxlen = min(max_length, 512 // len(target_table) - 1)
                else:
                    cur_maxlen = max_length
                    
                token_ids_list = target_table["data"].apply(lambda x: tokenizer.encode(
                    tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                    )
                token_ids = torch.LongTensor(reduce(operator.add,
                                                    token_ids_list)).to(device)

                target_col_mask = []
                cls_index_value = 0
                context_id = 1
                meet_target = False
                for idx, col_i in enumerate(col_idx_list):
                    if col_i == target_col_idx:
                        target_col_mask += [0] * len(token_ids_list[idx])
                        meet_target = True
                    else:
                        target_col_mask += [context_id] * len(token_ids_list[idx])
                        context_id += 1
                    if not meet_target:
                        cls_index_value += len(token_ids_list[idx])
                cls_index_list = [cls_index_value] 
                for cls_index in cls_index_list:
                    assert token_ids[
                        cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                class_ids = torch.LongTensor(
                    [target_cls]).to(device)
                target_col_mask = torch.LongTensor(target_col_mask).to(device)
                data_list.append(
                    [index,
                    len(target_table), token_ids, class_ids, cls_indexes, target_col_mask])                
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "target_col_mask"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"],
            "target_col_mask": self.table_df.iloc[idx]["target_col_mask"],
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
        
class GittablesTablewiseIterateDataset(data.Dataset):

    def __init__(
            self,
            cv: int,
            split: str,
            src: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "/data/zhihao/TU/GitTables/semtab_gittables/2022",
            base_tag: str = '', # blank, comma
            small_tag: str = "",
            train_ratio: float = 1.0,
            max_unlabeled=8,
            random_sample=False, # TODO
            train_only=False,
            seed=None): # TODO
        if device is None:
            device = torch.device('cpu')
        basename = small_tag+ "_cv_{}.csv"
    
        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
                print(split, i)
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)
            print(split)


        if gt_only:
            df = df[df["class_id"] > -1]
        if train_only and split != "train":
            df = df[df["class_id"] > -1]

        
        data_list = []
        
        df['class_id'] = df['class_id'].astype(int)
        df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
        df['col_idx'] = df['col_idx'].astype(int)
        df['data'] = df['data'].astype(str)
        
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)        
        
        # df.drop(df[(df['data'] == '') & (df['class_id'] == -1)].index, inplace=True)
        total_num_cols = 0
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue
            #     break
            labeled_columns = group_df[group_df['class_id'] > -1]
            unlabeled_columns = group_df[group_df['class_id'] == -1]
            num_unlabeled = min(max(max_unlabeled-len(labeled_columns), 0), len(unlabeled_columns)) if max_unlabeled > 0 else len(unlabeled_columns)
            unlabeled_columns = unlabeled_columns.sample(num_unlabeled, random_state=seed) if random_sample else unlabeled_columns[0:num_unlabeled]
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns.sample(min(10-len(labeled_columns), len(unlabeled_columns)))])
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns[0:min(max(10-len(labeled_columns), 0), len(unlabeled_columns))]])
            group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns]) # TODO
            group_df.sort_values(by=['col_idx'], inplace=True)

            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(list(group_df["class_id"].values)) - 1)
            else:
                cur_maxlen = max(1, max_length // len(list(group_df["class_id"].values)) - 1)
                
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            for col_i in range(len(token_ids_list)):
                if group_df["class_id"].values[col_i] == -1:
                    continue
                target_col_mask = []
                cls_index_value = 0
                context_id = 1
                for col_j in range(len(token_ids_list)):
                    if col_j == col_i:
                        target_col_mask += [0] * len(token_ids_list[col_j])
                    else:
                        target_col_mask += [context_id] * len(token_ids_list[col_j])
                        context_id += 1
                    if col_j < col_i:
                        cls_index_value += len(token_ids_list[col_j])
                cls_index_list = [cls_index_value] 
                for cls_index in cls_index_list:
                    assert token_ids[
                        cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                class_ids = torch.LongTensor(
                    [group_df["class_id"].values[col_i]]).to(device)
                target_col_mask = torch.LongTensor(target_col_mask).to(device)
                data_list.append(
                    [index,
                    len(group_df), token_ids, class_ids, cls_indexes, target_col_mask])                
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "target_col_mask"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"],
            "target_col_mask": self.table_df.iloc[idx]["target_col_mask"],
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
        
        
        
class VerificationDataset(data.Dataset):
    def __init__(
            self,
            data_path: str = "/data/zhihao/TU/Watchog/verification/veri_data.pth",
            max_list_length: int = 10,
            pos_ratio : int = 0.5, # None: only control pos_ratio to be less than 0.5
            label_padding_value: int = -1,
            data_padding_value: int = 101,
            ): 
        data_raw = torch.load(data_path)
        veri_label = data_raw["label"]
        veri_data = data_raw["data"]
        veri_cls_indexes = data_raw["cls_indexes"]
        self.label_padding_value = label_padding_value
        self.data_padding_value = data_padding_value
        self.max_list_length = max_list_length
        self.pos_ratio = pos_ratio
        self.veri_label = {}
        self.veri_data = {}
        self.veri_cls_indexes = {}
        i = 0
        for veri_i in veri_label:
            labels_i = torch.tensor(veri_label[veri_i]).reshape(-1)
            
            if 0 not in labels_i or 1 not in labels_i:
                continue
            self.veri_data[i] = veri_data[veri_i]
            self.veri_label[i] = labels_i
            self.veri_cls_indexes[i] = veri_cls_indexes[veri_i]
            i += 1
    def sample(self, labels):
        labels = labels.tolist()  # Convert tensor to list for easier manipulation
        positive_indices = [i for i, label in enumerate(labels) if label == 1]
        negative_indices = [i for i, label in enumerate(labels) if label == 0]

        # Determine how many positives we can sample, respecting the ratio requirement
        max_num_positives = max_num_negatives = min(len(positive_indices), len(negative_indices), self.max_list_length // 2)
        
        # Randomly sample the positives and negatives
        sampled_positives = random.sample(positive_indices, max_num_positives)
        sampled_negatives = random.sample(negative_indices, max_num_negatives)

        # Combine and shuffle the indices
        sampled_indices = sampled_positives + sampled_negatives

        return sampled_indices
        # {"data": veri_data, "label": veri_label, "cls_indexes": veri_cls_indexes}
    def __len__(self):
        return len(self.veri_data)

    def __getitem__(self, idx):
        labels = self.veri_label[idx]
        sampled_indices = self.sample(labels)
        sampled_labels = torch.tensor([labels[i] for i in sampled_indices]).reshape(-1)
        sampled_data = [self.veri_data[idx][i].reshape(-1) for i in sampled_indices]
        sampled_cls_indexes = torch.tensor([self.veri_cls_indexes[idx][i] for i in sampled_indices], dtype=torch.long)
        if len(sampled_indices) < self.max_list_length:
            sampled_labels = torch.cat([sampled_labels, torch.ones(self.max_list_length - len(sampled_labels))*self.label_padding_value])
            sampled_data.extend([torch.tensor([self.data_padding_value]) for _ in range(self.max_list_length - len(sampled_data))])
            sampled_cls_indexes = torch.cat([sampled_cls_indexes, torch.zeros(self.max_list_length - len(sampled_cls_indexes))])
        return {
            "data": sampled_data,
            "label": sampled_labels,
            "cls_indexes": sampled_cls_indexes, 
        }
        
class VerificationBinaryDataset(data.Dataset):
    def __init__(
            self,
            data_path: str = "/data/zhihao/TU/Watchog/verification/veri_data.pth",
            pos_ratio = None, # clone the negative samples
            context = None,
            ): 
        data_raw = torch.load(data_path)
        veri_label = data_raw["label"].long()
        # veri_data = data_raw["data"]
        # veri_cls_indexes = data_raw["cls_indexes"]
        veri_embs = data_raw["embs"]
        # veri_target_embs = data_raw["target_embs"]
        # veri_logits = data_raw["logits"]
        
        self.neg_expand = int(1/pos_ratio)-1 if pos_ratio is not None else 1
        self.veri_label = {}
        self.veri_data = {}
        self.veri_embs = {}
        self.veri_cls_indexes = {}
        self.veri_logits = {}
        i = 0
        for veri_i in veri_label:
            for veri_j in range(len(veri_label[veri_i])):
                if veri_label[veri_i][veri_j] == 1:
                    labels_i = torch.tensor([1]).reshape(-1)
                    # self.veri_data[i] = veri_data[veri_i][veri_j]
                    self.veri_label[i] = labels_i
                    # self.veri_cls_indexes[i] = veri_cls_indexes[veri_i][veri_j] 
                    
                    if context is None or context == "None":
                        self.veri_embs[i] = veri_embs[veri_i][veri_j]
                    elif context == "init":
                        self.veri_embs[i] = torch.cat([veri_embs[veri_i][0].reshape(-1), veri_embs[veri_i][veri_j].reshape(-1)])
                    elif context == "target":
                        self.veri_embs[i] = torch.cat([veri_target_embs[veri_i][0].reshape(-1), veri_embs[veri_i][veri_j].reshape(-1)])
                    else:
                        raise ValueError("context {} is not supported".format(context))
                        
                    # self.veri_logits[i] = veri_logits[veri_i][veri_j]
                    i += 1
                else:
                    for _ in range(self.neg_expand):
                        labels_i = torch.tensor([0]).reshape(-1)
                        # self.veri_data[i] = veri_data[veri_i][veri_j]
                        self.veri_label[i] = labels_i
                        # self.veri_cls_indexes[i] = veri_cls_indexes[veri_i][veri_j]
                        if context is None or context == "None":
                            self.veri_embs[i] = veri_embs[veri_i][veri_j]
                        elif context == "init":
                            self.veri_embs[i] = torch.cat([veri_embs[veri_i][0].reshape(-1), veri_embs[veri_i][veri_j].reshape(-1)])
                        elif context == "target":
                            self.veri_embs[i] = torch.cat([veri_target_embs[veri_i][0].reshape(-1), veri_embs[veri_i][veri_j].reshape(-1)])
                        else:
                            raise ValueError("context {} is not supported".format(context))
                        
                        # self.veri_logits[i] = veri_logits[veri_i][veri_j]
                        i += 1

    def __len__(self):
        return len(self.veri_data)

    def __getitem__(self, idx):
        return {
            # "data": self.veri_data[idx].reshape(-1),
            "embs": self.veri_embs[idx].reshape(-1),
            "label": self.veri_label[idx],
            # "logits": self.veri_logits[idx],
            # "cls_indexes": self.veri_cls_indexes[idx], 
        }

class VerificationBinaryFastDataset(data.Dataset):
    def __init__(
            self,
            data_path: str = "/data/zhihao/TU/Watchog/verification/veri_data.pth",
            pos_ratio = None, # clone the negative samples
            context = None,
            ): 
        data_raw = torch.load(data_path)
        veri_label = data_raw["label"].long()
        # veri_data = data_raw["data"]
        # veri_cls_indexes = data_raw["cls_indexes"]
        veri_embs = data_raw["embs"]
        # veri_target_embs = data_raw["target_embs"]
        # veri_logits = data_raw["logits"]
        
        self.neg_expand = int(1/pos_ratio)-1 if pos_ratio is not None else 1
        self.veri_label = {}
        self.veri_embs = {}

        assert len(veri_label) == len(veri_embs)
        self.veri_label = veri_label
        self.veri_embs = veri_embs

    def __len__(self):
        return len(self.veri_embs)

    def __getitem__(self, idx):
        return {
            # "data": self.veri_data[idx].reshape(-1),
            "embs": self.veri_embs[idx].reshape(-1),
            "label": self.veri_label[idx].reshape(-1),
            # "logits": self.veri_logits[idx],
            # "cls_indexes": self.veri_cls_indexes[idx], 
        }
        
        
        
class VerificationBinaryDropDataset(data.Dataset):
    def __init__(
            self,
            data_path: str = "/data/zhihao/TU/Watchog/verification/veri_data_drop_0.pth",
            context = None,
            label_version = 0,
            ): 
        data_raw = torch.load(data_path)
        veri_label = data_raw["label"]
        veri_data = data_raw["data"]
        veri_cls_indexes = data_raw["cls_indexes"]
        veri_embs = data_raw["embs"]
        veri_target_embs = data_raw["target_embs"]
        veri_logits = data_raw["logits"]
        
        self.veri_label = {}
        self.veri_data = {}
        self.veri_embs = {}
        self.veri_cls_indexes = {}
        self.veri_logits = {}
        i = 0
        for veri_i in veri_label:
            context_embs = veri_embs[veri_i][0].reshape(-1)
            for veri_j in range(1, len(veri_label[veri_i])):
                if label_version == 0:
                    if veri_label[veri_i][veri_j] == 2 or veri_label[veri_i][veri_j-1] == 3:
                        continue
                    label_i = veri_label[veri_i][veri_j].reshape(-1)
                elif label_version == 1:
                    if veri_label[veri_i][veri_j] == 2 or veri_label[veri_i][veri_j] == 3:
                        label_i = torch.tensor([0]).reshape(-1)
                    else:
                        label_i = veri_label[veri_i][veri_j].reshape(-1)
                elif label_version == 2:
                    if veri_label[veri_i][veri_j] == 2 or veri_label[veri_i][veri_j] == 3:
                        label_i = torch.tensor([1]).reshape(-1)
                    else:
                        label_i = veri_label[veri_i][veri_j].reshape(-1)
                elif label_version == 3:
                    if veri_label[veri_i][veri_j] == 2:
                        label_i = torch.tensor([1]).reshape(-1)
                    elif veri_label[veri_i][veri_j] == 3:
                        label_i = torch.tensor([0]).reshape(-1)
                    else:
                        label_i = veri_label[veri_i][veri_j].reshape(-1)
                else:
                    raise ValueError("label_version {} is not supported".format(label_version))
                self.veri_data[i] = veri_data[veri_i][veri_j]
                self.veri_label[i] = label_i
                self.veri_cls_indexes[i] = veri_cls_indexes[veri_i][veri_j] 
                

                if context == "concat":
                    self.veri_embs[i] = torch.cat([context_embs, veri_embs[veri_i][veri_j].reshape(-1)])
                elif context == "substract":
                    self.veri_embs[i] = context_embs - veri_embs[veri_i][veri_j].reshape(-1)
                elif context == "substract_abs":
                    self.veri_embs[i] = torch.abs(context_embs - veri_embs[veri_i][veri_j].reshape(-1))
                elif context == "sbert":
                    self.veri_embs[i] = torch.cat([context_embs, veri_embs[veri_i][veri_j].reshape(-1), torch.abs(context_embs - veri_embs[veri_i][veri_j].reshape(-1))])
                else:
                    raise ValueError("context {} is not supported".format(context))
                    
                self.veri_logits[i] = veri_logits[veri_i][veri_j]
                i += 1


    def __len__(self):
        return len(self.veri_embs)

    def __getitem__(self, idx):
        return {
            "data": self.veri_data[idx].reshape(-1),
            "embs": self.veri_embs[idx].reshape(-1),
            "label": self.veri_label[idx],
            "logits": self.veri_logits[idx],
            "cls_indexes": self.veri_cls_indexes[idx], 
        }

from collections import defaultdict
class VerificationCompareDataset(data.Dataset):
    def __init__(
            self,
            data_path: str = "/data/zhihao/TU/Watchog/verification/gt-semtab22-dbpedia-all0_veri_data.pth",
            pos_ratio = None, # clone the negative samples
            context = None,
            ): 
        
        self.num_neg = int((1-pos_ratio)/pos_ratio)
        self.context = context
        
        data_raw = torch.load(data_path)
        veri_label = data_raw["label"]
        veri_data = data_raw["data"]
        veri_cls_indexes = data_raw["cls_indexes"]
        veri_embs = data_raw["embs"]
        veri_target_embs = data_raw["target_embs"]
        veri_logits = data_raw["logits"]
        
        
        self.veri_pos_embs = defaultdict(list)
        self.veri_neg_embs = defaultdict(list)
        self.veri_anchor_embs = defaultdict(list)
        
        # self.veri_label = {}
        # self.veri_logits = {}
        # self.veri_cls_indexes = {}
        # self.veri_logits = {}
        i = 0
        for veri_i in veri_label:
            if 1 not in veri_label[veri_i] or 0 not in veri_label[veri_i]:
                continue
            # save anchor emb
            if context is None or context == "None" or context == "init":
                self.veri_anchor_embs[i].append(veri_embs[veri_i][0])
            elif context == "target":
                self.veri_anchor_embs[i].append(veri_target_embs[veri_i][0])        
            else:
                raise ValueError("context {} is not supported".format(context))
            # save pos and neg embs
            for veri_j in range(len(veri_label[veri_i])):
                if veri_label[veri_i][veri_j] == 1:
                    self.veri_pos_embs[i].append(veri_embs[veri_i][veri_j])
                else:
                    self.veri_neg_embs[i].append(veri_embs[veri_i][veri_j])
            i += 1

        # Maintain the iteration state for each anchor
        self.pos_iter = {anchor_idx: iter(pos_embs) for anchor_idx, pos_embs in self.veri_pos_embs.items()}
        self.neg_iter = {anchor_idx: iter(neg_embs) for anchor_idx, neg_embs in self.veri_neg_embs.items()}


    def __len__(self):
        return len(self.veri_anchor_embs)

    def reset_iterators(self, anchor_idx, type="pos"):
        """Reset the iterator for the given anchor when all positives or negatives are iterated through."""
        if type == "pos":
            self.pos_iter[anchor_idx] = iter(self.veri_pos_embs[anchor_idx])
            random.shuffle(self.veri_pos_embs[anchor_idx])
        else:
            self.neg_iter[anchor_idx] = iter(self.veri_neg_embs[anchor_idx])
            random.shuffle(self.veri_neg_embs[anchor_idx])


    def __getitem__(self, idx):
        
        anchor_emb = self.veri_anchor_embs[idx]

        # Get one positive embedding
        try:
            pos_emb = next(self.pos_iter[idx])
        except StopIteration:
            # If we've exhausted all positives, reset and shuffle
            self.reset_iterators(idx, type="pos")
            pos_emb = next(self.pos_iter[idx])

        # Get N negative embeddings
        neg_embs = []
        for _ in range(self.num_neg):
            try:
                neg_emb = next(self.neg_iter[idx])
                neg_embs.append(neg_emb)
            except StopIteration:
                # If we've exhausted all negatives, reset and shuffle
                self.reset_iterators(idx, type="neg")
                neg_emb = next(self.neg_iter[idx])
                neg_embs.append(neg_emb)
        if self.context is None or self.context == "None":
            embs = [pos_emb] + neg_embs
        else:
            embs = [anchor_emb, pos_emb] + neg_embs
        embs = torch.stack(embs, dim=0)
        
        
        
        return {
            # "data": self.veri_data[idx].reshape(-1),
            "embs": embs,
            # "label": self.veri_label[idx],
            # "logits": self.veri_logits[idx],
            # "cls_indexes": self.veri_cls_indexes[idx], 
        }


class SOTABTablewiseIterateDataset(data.Dataset):

    def __init__(
            self,
            # cv: int,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            # multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "/data/yongkang/TU/SOTAB",
            small_tag: str = "",
            label_encoder: LabelEncoder = None,
            max_unlabeled=8,
            random_sample=False, # TODO
            gt_only=False,
            ):
        # ?
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid", "test"]
        if split == "train":
            gt_df = pd.read_csv(os.path.join(base_dirpath, "CTA_training_small_gt.csv"))
            data_folder = "Train"
        elif split == "valid":
            gt_df = pd.read_csv(os.path.join(base_dirpath, "CTA_validation_gt.csv"))
            data_folder = "Validation"
        else:  # test
            gt_df = pd.read_csv(os.path.join(base_dirpath, "CTA_test_gt.csv"))
            data_folder = "Test"

        # 初始化或加载 LabelEncoder
        if label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder.fit(gt_df['label'])
        self.label_encoder = label_encoder

        # gt_set = set(zip(gt_df["table_name"], gt_df["column_index"]))

        table_files = [f for f in os.listdir(os.path.join(base_dirpath, data_folder)) if f in gt_df['table_name'].values]

        mapping_file_path = os.path.join(base_dirpath, "label_mapping.txt")
        df_csv_path = os.path.join(base_dirpath, f"comma_{split}_fully_deduplicated_sotab.csv")

        # 检查是否存在之前保存的标签映射关系
        if os.path.exists(mapping_file_path):
            # 如果存在，直接从文件中读取映射
            with open(mapping_file_path, 'r') as f:
                label_mapping = json.load(f)
            next_label_id = max(label_mapping.values()) + 1
            print(f"标签映射关系从 {mapping_file_path} 读取成功")
        else:
            raise FileNotFoundError(f"{mapping_file_path} 文件不存在，请确认文件路径")

        # 检查 CSV 文件是否存在
        if os.path.exists(df_csv_path):
            # 直接从 CSV 文件中读取数据
            df = pd.read_csv(df_csv_path)
            print(f"数据已从 {df_csv_path} 加载")
        else:
            raise FileNotFoundError(f"{df_csv_path} 文件不存在，请确认文件路径")
        
        if gt_only:
            df = df[df["label"] > -1]

        data_list = []

        df['label'] = df['label'].astype(int)
        df.drop(df[(df['data'].isna()) & (df['label'] == -1)].index, inplace=True)
        df['column_index'] = df['column_index'].astype(int)
        df['data'] = df['data'].astype(str)

        print("start encoder")
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if split == "train" and ((i >= num_train) or (i >= valid_index)):
            #     break
            # if split == "valid" and i < valid_index:
            #     continue

            labeled_columns = group_df[group_df['label'] > -1]
            unlabeled_columns = group_df[group_df['label'] == -1]
            num_unlabeled = min(max(max_unlabeled-len(labeled_columns), 0), len(unlabeled_columns))
            unlabeled_columns = unlabeled_columns.sample(num_unlabeled) if random_sample else unlabeled_columns[0:num_unlabeled]
            group_df = pd.concat([group_df[group_df['label'] > -1], unlabeled_columns]) # TODO
            group_df.sort_values(by=['column_index'], inplace=True)

            num_labels = len(list(group_df["label"].values))
            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // num_labels - 1)
            else:
                cur_maxlen = max(1, max_length // num_labels - 1)

            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            for col_i in range(len(token_ids_list)):
                if group_df["label"].values[col_i] == -1:
                    continue
                target_col_mask = []
                cls_index_value = 0
                context_id = 1
                new_token_ids_list = []
                for col_j in range(len(token_ids_list)):
                    if len(set(target_col_mask)) == max_unlabeled-1 and 0 not in target_col_mask and col_j != col_i:
                        # skip the rest of the columns until the target one
                        continue
                    
                    if col_j == col_i:
                        target_col_mask += [0] * len(token_ids_list[col_j])
                    else:
                        target_col_mask += [context_id] * len(token_ids_list[col_j])
                        context_id += 1
                    if col_j < col_i:
                        cls_index_value += len(token_ids_list[col_j])
                    new_token_ids_list.append(token_ids_list[col_j])
                    if len(set(target_col_mask)) == max_unlabeled and 0 in target_col_mask:
                        break
                new_token_ids_list = torch.LongTensor(reduce(operator.add,
                                                new_token_ids_list)).to(device)
                cls_index_list = [cls_index_value] 
                for cls_index in cls_index_list:
                    assert token_ids[
                        cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                cls_indexes = torch.LongTensor(cls_index_list).to(device)
                class_ids = torch.LongTensor(
                    [group_df["label"].values[col_i]]).to(device)
                target_col_mask = torch.LongTensor(target_col_mask).to(device)
                data_list.append(
                    [index,
                    len(group_df), new_token_ids_list, class_ids, cls_indexes, target_col_mask])                
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes", "target_col_mask"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"],
            "target_col_mask": self.table_df.iloc[idx]["target_col_mask"],
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}