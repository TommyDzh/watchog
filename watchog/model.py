import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class UnsupCLforTable(nn.Module):
    """ BarlowTwins/SimCLR encoder for contrastive learning
    """
    def __init__(self, hp, device='cuda', lm='roberta'):
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        self.temperature = hp.temperature
        hidden_size = 768

        # projector
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # contrastive
        # self.criterion = nn.CrossEntropyLoss().to(device)
        self.criterion = nn.CrossEntropyLoss().cuda()
        # cls token id
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id


    def info_nce_loss(self, features,
            batch_size,
            n_views):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        return logits, labels

    def _extract_columns(self, x, z, cls_indices=None):
        """Helper function for extracting column vectors from LM outputs.
        """
        x_flat = x.view(-1)
        column_vectors = z.view((x_flat.shape[0], -1))

        if cls_indices is None:
            indices = [idx for idx, token_id in enumerate(x_flat) \
                if token_id == self.cls_token_id]
        else:
            indices = []
            seq_len = x.shape[-1]
            for rid in range(len(cls_indices)):
                indices += [idx + rid * seq_len for idx in cls_indices[rid]]

        return column_vectors[indices]
    
    def _extract_table(self, x, z, cls_indices=None):
        """Helper function for extracting all column vectors in a table from LM outputs.
        """
        x_flat = x.view(-1)
        column_vectors = z.view((x_flat.shape[0], -1))

        if cls_indices is None:
            indices = [idx for idx, token_id in enumerate(x_flat) \
                if token_id == self.cls_token_id]
        else:
            indices = []
            seq_len = x.shape[-1]
            for rid in range(len(cls_indices)):
                indices += [idx + rid * seq_len for idx in cls_indices[rid]]

        return torch.mean(torch.stack(column_vectors[indices]))


    def inference(self, x):
        """Apply the model on a serialized table.

        Args:
            x (LongTensor): a batch of serialized tables

        Returns:
            Tensor: the column vectors for all tables
        """
        # x = x.to(self.device)
        x = x.cuda()
        z = self.bert(x)[0]
        z = self.projector(z) # optional
        return self._extract_columns(x, z)


    def forward(self, x_ori, x_aug, cls_indices, mode="simclr", task="None"):
        """Apply the model for contrastive learning.

        Args:
            x_ori (LongTensor): the first views of a batch of tables
            x_aug (LongTensor): the second views of a batch of tables
            cls_indices (tuple of List): the cls_token alignment
            mode (str, optional): the name of the contrastive learning algorithm
            task (str, optional): the supervision signal, unsupervised if task == "None"

        Returns:
            Tensor: the loss
        """
        if mode in ["simclr", "barlow_twins"]:
            # pre-training
            # encode
            batch_size = len(x_ori)
            x_ori = x_ori.cuda() # original, (batch_size, seq_len)
            x_aug = x_aug.cuda() # augment, (batch_size, seq_len)

            # encode each table (all columns)
            x = torch.cat((x_ori, x_aug)) # (2*batch_size, seq_len)
            z = self.bert(x)[0] # (2*batch_size, seq_len, hidden_size)

            # assert that x_ori and x_aug have the same number of columns
            z_ori = z[:batch_size] # (batch_size, seq_len, hidden_size)
            z_aug = z[batch_size:] # (batch_size, seq_len, hidden_size)
            
            cls_ori, cls_aug = cls_indices

            z_ori_col = self._extract_columns(x_ori, z_ori, cls_ori) # (total_num_columns, hidden_size)
            z_aug_col = self._extract_columns(x_aug, z_aug, cls_aug) # (total_num_columns, hidden_size)
            assert z_ori_col.shape == z_aug_col.shape

            z_col = torch.cat((z_ori_col, z_aug_col))
            z_col = self.projector(z_col) # (2*total_num_columns, projector_size)

            if mode == "simclr":
                # simclr
                logits, labels = self.info_nce_loss(z_col, len(z_col) // 2, 2)
                loss = self.criterion(logits, labels)                            
                return loss
            
            elif mode == "barlow_twins":
                # barlow twins
                z1 = z[:len(z) // 2]
                z2 = z[len(z) // 2:]

                # empirical cross-correlation matrix
                c = (self.bn(z1).T @ self.bn(z2)) / (len(z1))

                # use --scale-loss to multiply the loss by a constant factor
                on_diag = ((torch.diagonal(c) - 1) ** 2).sum() * self.hp.scale_loss
                off_diag = (off_diagonal(c) ** 2).sum() * self.hp.scale_loss
                loss = on_diag + self.hp.lambd * off_diag
                return loss
       


class SupCLforTable(nn.Module):
    """Supervised contrastive learning encoder for tables.
    """
    def __init__(self, hp, device='cuda', lm='roberta'):
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        self.temperature = hp.temperature
        hidden_size = 768
        # projector
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)

        # a fully connected layer for fine tuning
        # self.fc = nn.Linear(hidden_size * 2, 2)

        # contrastive
        self.criterion = nn.CrossEntropyLoss().cuda()

        # cls token id
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id

    def load_from_pretrained_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model = UnsupCLforTable(ckpt['hp'], device=self.device, lm=ckpt['hp'].lm)
        model.load_state_dict(ckpt['model'])
        self.bert = model.bert
        self.cls_token_id = model.cls_token_id
        del model

    def info_nce_loss(self, features,
            batch_size,
            n_views,
            temperature=0.07):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / temperature
        return logits, labels
    
    def supcon_loss(self, features,
            batch_size,
            n_views,
            signals=None):
        
        features = F.normalize(features, dim=1)
        # discard the main diagonal from both: labels and similarities matrix
        similarity_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()
        if len(signals.shape) == 1:
            mask_similar_class = (signals.unsqueeze(1).repeat(1, signals.shape[0]) == signals).cuda()
        else:
            mask_similar_class = (torch.sum(torch.eq(signals.unsqueeze(1).repeat(1,signals.shape[0],1),
                                                     signals.unsqueeze(0).repeat(signals.shape[0],1,1)), dim=-1) == signals.shape[-1]).cuda()
            
        mask_anchor_out = (1 - torch.eye(logits.shape[0])).cuda()
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        
        # log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        exp_logits = torch.exp(logits) * mask_anchor_out
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = -torch.mean(supervised_contrastive_loss_per_sample)
    
        return supervised_contrastive_loss

    def _extract_columns(self, x, z, cls_indices=None):
        """Helper function for extracting column vectors from LM outputs.
        """
        x_flat = x.view(-1)
        column_vectors = z.view((x_flat.shape[0], -1))

        if cls_indices is None:
            indices = [idx for idx, token_id in enumerate(x_flat) \
                if token_id == self.cls_token_id]
        else:
            indices = []
            seq_len = x.shape[-1]
            for rid in range(len(cls_indices)):
                indices += [idx + rid * seq_len for idx in cls_indices[rid]]

        return column_vectors[indices]


    def inference(self, x):
        """Apply the model on a serialized table.

        Args:
            x (LongTensor): a batch of serialized tables

        Returns:
            Tensor: the column vectors for all tables
        """
        # x = x.to(self.device)
        x = x.cuda()
        z = self.bert(x)[0]
        # z = self.projector(z) # optional
        return self._extract_columns(x, z)


    def forward(self, x_ori, x_aug, cls_indices, supervised_signals, mode="simclr", task="None"):
        """Apply the model for contrastive learning.

        Args:
            x_ori (LongTensor): the first views of a batch of tables
            x_aug (LongTensor): the second views of a batch of tables
            cls_indices (tuple of List): the cls_token alignment
            mode (str, optional): the name of the contrastive learning algorithm
            task (str, optional): the supervision signal, unsupervised if task == "None"

        Returns:
            Tensor: the loss or the column embeddings of x_ori, x_aug
        """
        if mode in ["simclr", "barlow_twins", "supcl", "supcon"]:
            # pre-training
            # encode
            batch_size = len(x_ori)
            x_ori = x_ori.cuda() # original, (batch_size, seq_len)
            x_aug = x_aug.cuda() # augment, (batch_size, seq_len)

            # encode each table (all columns)
            x = torch.cat((x_ori, x_aug)) # (2*batch_size, seq_len)
            z = self.bert(x)[0] # (2*batch_size, seq_len, hidden_size)

            # assert that x_ori and x_aug have the same number of columns
            z_ori = z[:batch_size] # (batch_size, seq_len, hidden_size)
            z_aug = z[batch_size:] # (batch_size, seq_len, hidden_size)

            if cls_indices is None:
                cls_ori = None
                cls_aug = None
            else:
                cls_ori, cls_aug = cls_indices
                
            z_ori_col = self._extract_columns(x_ori, z_ori, cls_ori) # (total_num_columns, hidden_size)
            z_aug_col = self._extract_columns(x_aug, z_aug, cls_aug) # (total_num_columns, hidden_size)
            assert z_ori_col.shape == z_aug_col.shape

            z_col = torch.cat((z_ori_col, z_aug_col))
            z_col = self.projector(z_col) # (2*total_num_columns, projector_size)

            if mode == "simclr":
                # simclr
                logits, labels = self.info_nce_loss(z_col, len(z_col) // 2, 2)
                loss = self.criterion(logits, labels)            
                return loss
            elif mode == "barlow_twins":
                # barlow twins
                z1 = z[:len(z) // 2]
                z2 = z[len(z) // 2:]
                # empirical cross-correlation matrix
                c = (self.bn(z1).T @ self.bn(z2)) / (len(z1))
                # use --scale-loss to multiply the loss by a constant factor
                on_diag = ((torch.diagonal(c) - 1) ** 2).sum() * self.hp.scale_loss
                off_diag = (off_diagonal(c) ** 2).sum() * self.hp.scale_loss
                loss = on_diag + self.hp.lambd * off_diag
                return loss
            elif mode in ["supcon", "supcon_ddp"]:
                if supervised_signals is not None:
                    loss = self.supcon_loss(z_col, len(z_col) // 2, 2, signals=supervised_signals)
                else:
                    # only return embedding, compute loss outside
                    return z_col
                return loss
                
        else:
            pass


class SupclLoss(nn.Module):
    """The class of loss function for computing the CL loss when using DDP and gather batches from all workers"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, signals=None):
        features = F.normalize(features, dim=1)
        # discard the main diagonal from both: labels and similarities matrix
        similarity_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        logits = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()
        if len(signals.shape) == 1:
            mask_similar_class = (signals.unsqueeze(1).repeat(1, signals.shape[0]) == signals).cuda()
        else:
            mask_similar_class = (torch.sum(torch.eq(signals.unsqueeze(1).repeat(1,signals.shape[0],1),
                                                     signals.unsqueeze(0).repeat(signals.shape[0],1,1)), dim=-1) == signals.shape[-1]).cuda()
            
        mask_anchor_out = (1 - torch.eye(logits.shape[0])).cuda()
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        
        # log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        exp_logits = torch.exp(logits) * mask_anchor_out
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = -torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

def pool_sub_sentences(hidden_states, cls_indexes, table_length=None):
    pooled_outputs = []
    B = hidden_states.size(0)
    
    for i in range(B):
        cls_indices = cls_indexes[cls_indexes[:,0]==i]
        sub_sentences = []
        max_length = table_length[i] if table_length is not None else hidden_states.size(1)
        # Extract sub-sentence embeddings based on CLS tokens
        for j in range(len(cls_indices)):
            start_idx = cls_indices[j][1].item()
            end_idx = cls_indices[j+1][1].item() if j+1 < len(cls_indices) else max_length
            
            # Pooling (e.g., mean pooling) the tokens in the sub-sentence
            sub_sentence_embedding = hidden_states[i, start_idx:end_idx, :].mean(dim=0)
            sub_sentences.append(sub_sentence_embedding)
        
        pooled_outputs.append(torch.stack(sub_sentences))
    pooled_outputs = torch.cat(pooled_outputs, dim=0)
    return pooled_outputs

# Function to extract CLS token embeddings for each sub-sentence
def extract_cls_tokens(hidden_states, cls_indexes, head=False):
    cls_embeddings = []
    for i, j in cls_indexes:
        sub_sentence_cls_embeddings = hidden_states[i, 0, :] if head else hidden_states[i, j, :]
        cls_embeddings.append(sub_sentence_cls_embeddings)
    cls_embeddings = torch.stack(cls_embeddings)
    return cls_embeddings

class BertMultiPooler(nn.Module):

    def __init__(self, hidden_size, version='v0'):
        super().__init__()
        self.version = version
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense_tab = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pooler_output=None, cls_indexes=None, table_length=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        #token_tensor = torch.index_select(hidden_states, 1,
        #                                  cls_indexes)
        # Apply
        #pooled_outputs = self.dense(token_tensor)
        if self.version == "v0":
            pooled_outputs = extract_cls_tokens(hidden_states, cls_indexes)
            pooled_outputs = self.dense(pooled_outputs)
        elif self.version == "v0.1":
            pooled_outputs = extract_cls_tokens(hidden_states, cls_indexes)
            tab_outputs = pooler_output[cls_indexes[:,0]] # (B, hidden_size)
            pooled_outputs = self.dense(pooled_outputs) + self.dense_tab(tab_outputs)
        elif self.version == "v0.2":
            pooled_outputs = hidden_states[:, 0, :]
            pooled_outputs = self.dense(pooled_outputs)
        elif self.version == "v0.3":
            pooled_outputs = pooler_output
            pooled_outputs = self.dense(pooled_outputs)            
        elif self.version == "v1":
            pooled_outputs = pool_sub_sentences(hidden_states, cls_indexes, table_length)
            pooled_outputs = self.dense(pooled_outputs)
        elif self.version == "v1.1": # TODO: project before pooling
            hidden_states = self.dense(hidden_states)
            pooled_outputs = pool_sub_sentences(hidden_states, cls_indexes, table_length)
        elif self.version == "v2":
            pooled_outputs = pool_sub_sentences(hidden_states, cls_indexes, table_length)
            pooled_outputs = self.dense(pooled_outputs)
            tab_outputs = extract_cls_tokens(hidden_states, cls_indexes)
            pooled_outputs = pooled_outputs + self.dense_tab(tab_outputs)
        elif self.version == "v3": # tab token is the first
            pooled_outputs = pool_sub_sentences(hidden_states, cls_indexes, table_length)
            pooled_outputs = self.dense(pooled_outputs)
            tab_outputs = extract_cls_tokens(hidden_states, cls_indexes, head=True)
            pooled_outputs = pooled_outputs + self.dense_tab(tab_outputs)
        elif self.version == "v4": # use Bert pooler_output as tab token
            pooled_outputs = pool_sub_sentences(hidden_states, cls_indexes, table_length)
            pooled_outputs = self.dense(pooled_outputs)
            tab_outputs = pooler_output[cls_indexes[:,0]] # (B, hidden_size)
            pooled_outputs = pooled_outputs + self.dense_tab(tab_outputs)
        elif self.version == "v4.1": 
            hidden_states = self.dense(hidden_states)
            pooled_outputs = pool_sub_sentences(hidden_states, cls_indexes, table_length)
            tab_outputs = pooler_output[cls_indexes[:,0]] # (B, hidden_size)
            pooled_outputs = pooled_outputs + self.dense_tab(tab_outputs)            
        else:
            raise ValueError(f"Invalid version: {self.version}")
        pooled_outputs = self.activation(pooled_outputs)
        return pooled_outputs


class BertMultiPairPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        #token_tensor = torch.index_select(hidden_states, 1,
        #                                  cls_indexes)
        # Apply
        #pooled_outputs = self.dense(token_tensor)
        # print(hidden_states.shape)
        hidden_states_first_cls = hidden_states[:, 0].unsqueeze(1).repeat(
            [1, hidden_states.shape[1], 1])
        pooled_outputs = self.dense(
            torch.cat([hidden_states_first_cls, hidden_states], 2))
        pooled_outputs = self.activation(pooled_outputs)
        # pooled_outputs = pooled_outputs.squeeze(0)
        # print(pooled_outputs.shape)
        return pooled_outputs

class BertForMultiOutputClassification(nn.Module):

    def __init__(self, hp, device='cuda', lm='roberta', col_pair='None', version='v0'):
        
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        self.col_pair = col_pair
        self.version = version
        hidden_size = 768

        # projector
        self.pooler = BertMultiPooler(hidden_size, version=version)
        self.projector = nn.Linear(hidden_size, hp.projector)
        '''Require all models using the same CLS token'''
        # self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp['roberta']).cls_token_id
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id
        self.num_labels = hp.num_labels
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.hp.num_labels)
    
    def load_from_CL_model(self, model):
        '''load from models pre-trained with contrastive learning'''
        self.bert = model.bert
        self.projector = model.projector
        self.cls_token_id = model.cls_token_id
    
    def forward(
        self,
        input_ids=None,
        get_enc=False,
        cls_indexes=None,
        token_type_ids=None,
    ):
        # BertModelMultiOutput
        bert_output = self.bert(input_ids, token_type_ids=token_type_ids)
        table_length = [len(input_ids[i].nonzero()) for i in range(len(input_ids))]
        # Note: returned tensor contains pooled_output of all tokens (to make the tensor size consistent)
        pooled_output = self.pooler(bert_output[0], pooler_output=bert_output[1], cls_indexes=cls_indexes, table_length=table_length).squeeze(0)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if get_enc:
            outputs = (logits, pooled_output)
        else:
            outputs = logits
        return outputs  # (loss), logits, (hidden_states), (attentions)




class BertForMultiOutputClassificationColPopl(nn.Module):

    def __init__(self, hp, device='cuda', lm='roberta', col_pair='None', n_seed_cols=-1, cls_for_md=False):
        
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        self.col_pair = col_pair
        self.n_seed_cols = 3 if n_seed_cols == -1 else n_seed_cols
        self.cls_for_md = cls_for_md
        if self.cls_for_md:
            self.n_seed_cols += 1
        hidden_size = 768

        # projector
        self.projector = nn.Linear(hidden_size, hp.projector)

        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp['roberta']).cls_token_id
        self.num_labels = hp.num_labels
        if self.n_seed_cols > 1:
            self.dense = nn.Linear(hidden_size * self.n_seed_cols, hidden_size)
            self.activation = nn.Tanh()
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.hp.num_labels)

    def load_from_CL_model(self, model):
        self.bert = model.bert
        self.projector = model.projector
        self.cls_token_id = model.cls_token_id
    

    def forward(
        self,
        input_ids=None,
        cls_indexes=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):
        # print(295, input_ids.shape)
        # BertModelMultiOutput
        if "distilbert" in self.hp.__dict__['shortcut_name']:
            bert_output = self.bert(
                input_ids,
                #cls_indexes,
                attention_mask=attention_mask
            )
        else:
            bert_output = self.bert(
                input_ids,
                #cls_indexes,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        if self.n_seed_cols == 1:
            pooled_output = bert_output[0][:, 0]
        else:
            hidden_states = bert_output[0]
            cls_outputs = hidden_states[cls_indexes[:,0], cls_indexes[:, 1]].reshape(hidden_states.shape[0], self.n_seed_cols, 768)
            pooled_output = cls_outputs[:, 0]
            for j in range(1, self.n_seed_cols):
                pooled_output = torch.cat([pooled_output, cls_outputs[:, j]], dim=1)
            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
      
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
      

        outputs = (logits, )

        
        return outputs  # (loss), logits, (hidden_states), (attentions)




import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler

import random
import time
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
EPSILON = np.finfo(np.float32).tiny

class SubsetOperator(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        assert len(scores.shape) == 2
        # Add noise by sampling from Gumbel distribution
        # v0: training & eval with noise
        # v1: training with noise, eval without noise
        if self.training:
            m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
            g = m.sample()
            scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).to(scores.device))
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res

import torch.nn.functional as F
from torch_scatter import scatter
from typing import List, Optional, Tuple, Union
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
class BertForMultiSelectionClassification(nn.Module):

    def __init__(self, hp, device='cuda', lm='roberta', col_pair='None', version='v0', 
                 tau=1.0, max_num_cols=10, target_num_cols=8, num_tokens_per_col=64, gate_version='v0.1'):
        
        super().__init__()
        assert num_tokens_per_col * target_num_cols <= 512
        self.N = max_num_cols
        self.M = target_num_cols
        self.num_tokens_per_col = num_tokens_per_col
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        self.col_pair = col_pair
        self.version = version
        self.gate_version = gate_version
        self.hidden_size = hidden_size = 768

        # projector
        self.pooler = BertMultiPooler(hidden_size, version=version)
        self.projector = nn.Linear(hidden_size, hp.projector)
        '''Require all models using the same CLS token'''
        # self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp['roberta']).cls_token_id
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id
        self.num_labels = hp.num_labels
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.hp.num_labels)
        # For selection module
        self.sampler = SubsetOperator(k=self.M, tau=tau, hard=True)
        if gate_version == 'v0':
            self.gate = nn.Linear(2*hidden_size, 1)
        elif gate_version == 'v0.1':
            self.gate = nn.Sequential(
                nn.Linear(2*hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        elif gate_version == 'v0.2':
            self.gate = nn.Sequential(
                nn.Linear(2*hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_size, 1)
            )          
        elif gate_version == 'v1.1' or gate_version == 'v2.1':
            self.gate = nn.Sequential(
                nn.Linear(3*hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        elif gate_version == 'v1.2' or gate_version == 'v2.2':
            self.gate = nn.Sequential(
                nn.Linear(3*hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_size, 1)
            )
        elif gate_version == 'v3.1':
            self.gate = nn.Sequential(
                nn.Linear(4*hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )    
        elif gate_version == 'v3.2':
            self.gate = nn.Sequential(
                nn.Linear(4*hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_size, 1)
            )
        elif gate_version == 'v4.1' or gate_version == 'v5.1':
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )            
        elif gate_version == 'v4.2' or gate_version == 'v5.2':
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_size, 1)
            )
        else:
            raise ValueError(f"Invalid gate version: {gate_version}")

    def load_from_CL_model(self, model):
        '''load from models pre-trained with contrastive learning'''
        self.bert = model.bert
        self.projector = model.projector
        self.cls_token_id = model.cls_token_id

    def forward(
        self,
        input_ids=None,
        get_enc=False,
        return_gates=False,
        cls_indexes=None,
        token_type_ids=None,
        column_embeddings=None,
    ):
        device = input_ids.device
        B = input_ids.size(0)
        N = self.N
        M = self.M
        num_tokens_per_col = self.num_tokens_per_col
        assert num_tokens_per_col * N <= input_ids.size(1) 
        D = self.hidden_size
        
        # Select top M columns
        split_ids = input_ids.split(num_tokens_per_col, dim=1)
        token_embedding = []
        for i in range(N):
            if i == 0:
                token_embedding.append(self.bert.embeddings(split_ids[i], token_type_ids=torch.zeros_like(split_ids[i], device=device)))
            else:
                token_embedding.append(self.bert.embeddings(split_ids[i], token_type_ids=torch.ones_like(split_ids[i], device=device)))
        token_embedding = torch.cat(token_embedding, dim=1)
        col_to_token = torch.eye(N, device=device, dtype=torch.float) \
                    .unsqueeze(-1).repeat(1, 1, num_tokens_per_col).transpose(1, 2).flatten(0, 1) \
                    .transpose(0, 1)[1:, :].unsqueeze(0).repeat(B, 1, 1) # (B, N-1, L) [1, ,1 ,1 , ..., 1, 0, 0, ..., 0, 0, 0], ...,  [0, 0, 0, ..., 0, 1, 1, ..., 1]
        if column_embeddings is None:
            token_col_indexes = torch.cat([torch.tensor([i]*num_tokens_per_col) for i in range(N)]).to(device) # [0, 0, ..., 1, 1, ..., 2, 2, ..., N-1, N-1, ..., N-1]
            column_embeddings = scatter(token_embedding, token_col_indexes, dim=1, reduce="mean") # (B, N, D)
        # TODO: add PE for column embeddings
        target_column_embeddings, context_column_embeddings = column_embeddings[:, 0, :].unsqueeze(1),  column_embeddings[:, 1:, :]# (B, M, D)
        target_column_embeddings = target_column_embeddings.repeat(1, context_column_embeddings.shape[1], 1) # (B, N, D)
        if self.gate_version.startswith('v0'):
            context_embeddings = torch.cat([target_column_embeddings, context_column_embeddings], dim=-1) # (B, N, D)
        elif self.gate_version.startswith('v1'):
            diff_embeddings = torch.abs(target_column_embeddings - context_column_embeddings)
            context_embeddings = torch.cat([target_column_embeddings, context_column_embeddings, diff_embeddings], dim=-1) # (B, N, D)
        elif self.gate_version.startswith('v2'):
            dot_embeddings = target_column_embeddings * context_column_embeddings
            context_embeddings = torch.cat([target_column_embeddings, context_column_embeddings, dot_embeddings], dim=-1) # (B, N, D)
        elif self.gate_version.startswith('v3'):
            diff_embeddings = torch.abs(target_column_embeddings - context_column_embeddings)
            dot_embeddings = target_column_embeddings * context_column_embeddings
            context_embeddings = torch.cat([target_column_embeddings, context_column_embeddings, diff_embeddings, dot_embeddings], dim=-1) # (B, N, D)
        elif self.gate_version.startswith('v4'):
            context_embeddings = torch.abs(target_column_embeddings - context_column_embeddings)
        elif self.gate_version.startswith('v5'):
            context_embeddings = target_column_embeddings * context_column_embeddings
        else:
            raise ValueError(f"Invalid gate version: {self.gate_version}")
        column_logits = self.gate(context_embeddings).squeeze() # (B, N)
        gates = self.sampler(column_logits) # (B, M)
        chosen_mask = torch.matmul(gates.unsqueeze(1), col_to_token).squeeze(1).unsqueeze(-1) # (B, N, 1)
        chosen_embeddings = token_embedding * chosen_mask 
        chosen_embeddings = chosen_embeddings[chosen_mask.detach().bool().expand_as(chosen_embeddings)].view(B, -1, D)
        embedding_output = torch.cat([token_embedding[:, :num_tokens_per_col, :], chosen_embeddings], dim=1)
        
        bert_output = self.attn_forward(embedding_output)
        
        # BertModelMultiOutput
        table_length = [len(input_ids[i].nonzero()) for i in range(len(input_ids))]
        # Note: returned tensor contains pooled_output of all tokens (to make the tensor size consistent)
        pooled_output = self.pooler(bert_output[0], pooler_output=bert_output[1], cls_indexes=cls_indexes, table_length=table_length).squeeze(0)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = [logits] 
        if get_enc:
            outputs.append(pooled_output)
        if return_gates:
            outputs.append(gates.clone().detach().cpu())

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
    def attn_forward(
        self,
        embedding_output: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )
        return_dict = False

        if self.bert.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.bert.config.use_cache
        else:
            use_cache = False

        input_shape = embedding_output.shape[:2]

        batch_size, seq_length = input_shape
        device = embedding_output.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0


        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.bert.attn_implementation == "sdpa"
            and self.bert.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.bert.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output) if self.bert.pooler is not None else None

        # if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )
