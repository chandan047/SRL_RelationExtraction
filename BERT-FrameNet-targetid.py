#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[2]:


import json
import numpy as np
from os import path
from sesame.evaluation import *
from sesame.dataio import *
from sesame.globalconfig import VERSION, TRAIN_FTE, UNK, DEV_CONLL, TEST_CONLL
from sesame.conll09 import lock_dicts, post_train_lock_dicts, LEMDICT
import sys
from tqdm import tqdm


# In[3]:


# from .conll09 import lock_dicts, post_train_lock_dicts, VOCDICT, POSDICT, LEMDICT, LUDICT, LUPOSDICT
# from .dataio import create_target_lu_map, get_wvec_map, read_conll
# from .evaluation import calc_f, evaluate_example_targetid
# from .frame_semantic_graph import LexicalUnit
# from .housekeeping import unk_replace_tokens
# from .raw_data import make_data_instance
# from .semafor_evaluation import convert_conll_to_frame_elements


# In[4]:


train_conll = TRAIN_FTE
dev_conll = DEV_CONLL
test_conll = TEST_CONLL


# In[5]:


def combine_examples(corpus_ex):
    """
    Target ID needs to be trained for all targets in the sentence jointly, as opposed to
    frame and arg ID. Returns all target annotations for a given sentence.
    """
    combined_ex = [corpus_ex[0]]
    for ex in tqdm(corpus_ex[1:]):
        if ex.sent_num == combined_ex[-1].sent_num:
            current_sent = combined_ex.pop()
            target_frame_dict = current_sent.targetframedict.copy()
            target_frame_dict.update(ex.targetframedict)
            current_sent.targetframedict = target_frame_dict
            combined_ex.append(current_sent)
            continue
        combined_ex.append(ex)
    sys.stderr.write("Combined {} instances in data into {} instances.\n".format(
        len(corpus_ex), len(combined_ex)))
    return combined_ex


# In[6]:


train_examples, _, _ = read_conll(train_conll)
combined_train = combine_examples(train_examples)

target_lu_map, lu_names = create_target_lu_map()
post_train_lock_dicts()
lock_dicts()

dev_examples, _, _ = read_conll(dev_conll)
combined_dev = combine_examples(dev_examples)

test_examples, _, _ = read_conll(test_conll)
combined_test = combine_examples(test_examples)


# In[7]:


def check_if_potential_target(lemma):
    """
    Simple check to see if this is a potential position to even consider, based on
    the LU index provided under FrameNet. Note that since we use NLTK lemmas,
    this might be lossy.
    """
    nltk_lem_str = LEMDICT.getstr(lemma)
    return nltk_lem_str in target_lu_map or nltk_lem_str.lower() in target_lu_map


# In[8]:


from torch.utils import data


class TargetDataset(data.Dataset):
    def __init__(self, sentences, pos, lemmas, labels):
        self.sentences = sentences
        self.pos = pos
        self.lemmas = lemmas
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.pos[index], self.lemmas[index], self.labels[index]

    
def read_dataset(corpus_ex):
    sentences = []
    postags = []
    all_lemmas = []
    all_labels = []
    
    for ex in tqdm(corpus_ex):
        words = [VOCDICT.getstr(tok) for tok in ex.tokens]
        pos = ex.postags
        lemmas = ex.lemmas
        labels = [-1] * len(words)
        
        for index in range(len(lemmas)):
            if check_if_potential_target(lemmas[index]):
                labels[index] = 0
        
        for index, (lu, fr) in ex.targetframedict.items():
            if labels[index] != 0:
                print ("Not a potential target but labeled as target", LUDICT.getstr(lu.id))
            labels[index] = 1
        
        sentences.append(words)
        postags.append(pos)
        all_lemmas.append(lemmas)
        all_labels.append(labels)

    dataset = TargetDataset(sentences, postags, all_lemmas, all_labels)
    return dataset


# In[9]:


train_dataset = read_dataset(combined_train)
dev_dataset = read_dataset(combined_dev)
test_dataset = read_dataset(combined_test)


# In[10]:


from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# In[11]:


def prepare_bert_batch(batch):
    x = []
    pos = []
    lem = []
    lengths = []
    y = []
    
    for sentences, x_pos, x_lemmas, labels in batch:
        length = 0
        tokens = []
        x_pos_ids = []
        x_lem_ids = []
        label_ids = []
        
        for word, x_p, x_l, label in zip(sentences, x_pos, x_lemmas, labels):
            word_tokens = bert_tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            x_pos_ids.extend([x_p] + [POSDICT.getid("UNK")] * (len(word_tokens) - 1))
            x_lem_ids.extend([x_p] + [LEMDICT.getid("UNK")] * (len(word_tokens) - 1))
            label_ids.extend([label] + [-1] * (len(word_tokens) - 1))
            length += len(word_tokens)
    
        x.append(tokens)
        pos.append(x_pos_ids)
        lem.append(x_lem_ids)
        lengths.append(length)
        y.append(label_ids)
    
    max_len = max(lengths)
    for i in range(len(y)):
        y[i] = y[i] + [-1] * (max_len - len(y[i]))
        pos[i] = pos[i] + [POSDICT.getid("UNK")] * (max_len - len(pos[i]))
        lem[i] = lem[i] + [LEMDICT.getid("UNK")] * (max_len - len(lem[i]))
    
    return x, pos, lem, lengths, y


# # Model

# In[12]:


import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# In[13]:


torch.cuda.device_count(), torch.cuda.current_device()


# In[14]:


import yaml

with open('config_targetid.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['base_path'] = '.'
config['stamp'] = "targetid"


# In[15]:


train_dataloader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                             collate_fn=lambda pb: prepare_bert_batch(pb))

dev_dataloader = data.DataLoader(dev_dataset, batch_size=config['eval_batch_size'],
                                             collate_fn=lambda pb: prepare_bert_batch(pb))

test_dataloader = data.DataLoader(test_dataset, batch_size=config['eval_batch_size'],
                                             collate_fn=lambda pb: prepare_bert_batch(pb))


# In[16]:


from sklearn import metrics
from torch.nn import functional as F

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        
        pred = pred[gold != trg_pad_idx]
        gold = gold[gold != trg_pad_idx]
        
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.mean()
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def calculate_metrics(predictions, labels, binary=False):
    averaging = 'binary' if binary else 'macro'
    predictions = torch.stack(predictions).cpu().numpy()
    labels = torch.stack(labels).cpu().numpy()
    unique_labels = np.unique(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging, labels=unique_labels)
    recall = metrics.recall_score(labels, predictions, average=averaging, labels=unique_labels)
    f1_score = metrics.f1_score(labels, predictions, average=averaging, labels=unique_labels)
    return accuracy, precision, recall, f1_score


def make_prediction(output):
    with torch.no_grad():
        if output.size(1) == 1:
            pred = (output > 0).int()
        else:
            pred = output.max(-1)[1]
    return pred


# In[17]:


from torch import nn
from transformers import BertModel


class BERTSequenceModel(nn.Module):

    def __init__(self, model_params):
        super(BERTSequenceModel, self).__init__()
        self.embed_dim = model_params['embed_dim']
        self.hidden_size = model_params['hidden_size']
        # self.hidden_size = model_params['embed_dim']
        self.dropout_ratio = model_params.get('dropout_ratio', 0)
        self.n_tunable_layers = model_params.get('fine_tune_layers', None)
        self.bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
        # self.bert = BertModel.from_pretrained('span-bert')
        self.linear = nn.Sequential(nn.Linear(self.embed_dim, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(p=self.dropout_ratio))

        self.bert.pooler.dense.weight.requires_grad = False
        self.bert.pooler.dense.bias.requires_grad = False

        if self.n_tunable_layers is not None:
            tunable_layers = {str(l) for l in range(12 - self.n_tunable_layers, 12)}
            for name, param in self.bert.named_parameters():
                if not set.intersection(set(name.split('.')), tunable_layers):
                    param.requires_grad = False

    def forward(self, input, input_len):
        attention_mask = (input.detach() != 0).float()
        output, _, _ = self.bert(input, attention_mask=attention_mask)
        
        cls_token = output[:, 0, :].unsqueeze(1)
        output = output[:, 1:-1, :]  # Ignore the output of the CLS and SEP tokens
        output = self.linear(output)
        
        return output


# In[18]:


import coloredlogs
import logging

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# In[19]:


from transformers import BertTokenizer, AdamW, get_constant_schedule_with_warmup

class SeqSupervisedNetwork(nn.Module):
    def __init__(self, config):
        super(SeqSupervisedNetwork, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.lr = config.get('meta_lr', 1e-3)
        self.weight_decay = config.get('weight_decay', 0.0)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

        self.learner = BERTSequenceModel(config['learner_params'])
        self.pos_embedding = nn.Embedding(POSDICT.size(), config['pos_dim'])
        
        self.classifier = nn.Linear(config['learner_params']['hidden_size'] + config['pos_dim'], 
                                    config['learner_params']['num_outputs']['ner'])
        
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
        if config.get('trained_learner', False):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_learner'])
            ))
            logger.info('Loaded trained learner model {}'.format(config['trained_learner']))
        
        self.device = torch.device(config.get('device', 'cpu'))
        logger.info('Model loaded to {}'.format(self.device))
        self.to(self.device)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.initialize_optimizer_scheduler()

    def initialize_optimizer_scheduler(self):
        learner_params = [p for p in self.learner.parameters() if p.requires_grad]
        learner_params += self.pos_embedding.parameters()
        learner_params += self.classifier.parameters()
        
        self.optimizer = AdamW(learner_params, lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=100)

    def vectorize(self, batch_x, batch_pos, batch_len, batch_y):
        with torch.no_grad():
            max_batch_len = max(batch_len) + 2
            input_ids = torch.zeros((len(batch_x), max_batch_len)).long()
            for i, sent in enumerate(batch_x):
                sent_token_ids = self.bert_tokenizer.encode(sent, add_special_tokens=True)
                input_ids[i, :len(sent_token_ids)] = torch.tensor(sent_token_ids)
            batch_x = input_ids.to(self.device)
            
            batch_pos = torch.LongTensor(batch_pos).to(self.device)
            batch_pos = self.pos_embedding(batch_pos)

        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_pos, batch_len, batch_y
    
    def forward(self, dataloader, testing=False):
        if not testing:
            self.train()
        else:
            self.eval()
        
        eval_every = 100
        
        avg_loss = 0
        all_predictions, all_labels = [], []
        
        for batch_id, batch in enumerate(dataloader):
            batch_x, batch_pos, batch_lem, batch_len, batch_y = batch
            
            batch_x, batch_pos, batch_len, batch_y = self.vectorize(batch_x, batch_pos, batch_len, batch_y)

            batch_x_repr = self.learner(batch_x, batch_len)
            output = torch.cat((batch_x_repr, batch_pos), 2)
            output = self.classifier(output)

            batch_size, seq_len = output.shape[0], output.shape[1]
            
            output = output.view(batch_size * seq_len, -1)
            
            batch_y = batch_y.view(-1)
            
            classification_loss = self.loss_fn(output, batch_y)
            # classification_loss = cal_loss(output, batch_y, -1, smoothing=True)
            loss = classification_loss
            
            avg_loss += loss.item()
            
            if not testing:
                loss.backward(retain_graph=True)
                if (batch_id + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.lr_scheduler.step()
            
            predictions, labels = [], []
            
            relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
            predictions.extend(make_prediction(output[relevant_indices]).cpu())
            labels.extend(batch_y[relevant_indices].cpu())
            
            accuracy, precision, recall, f1_score = calculate_metrics(predictions,
                                                                            labels, binary=False)

            if (batch_id + 1) % eval_every == 0:
                logger.info('Batch {}/{}, [supervised]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                    'recall = {:.5f}, F1 score = {:.5f}'.format(batch_id + 1, len(dataloader),
                                                                classification_loss.item(),
                                                                accuracy, precision, recall, f1_score))
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)

        avg_loss /= len(dataloader)

        # Calculate metrics
        accuracy, precision, recall, f1_score = calculate_metrics(all_predictions,
                                                                        all_labels, binary=False)

        return avg_loss, accuracy, precision, recall, f1_score


# In[20]:


class SupervisedNetwork:
    def __init__(self, config):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.tensorboard_writer = SummaryWriter(log_dir='runs/Supervised-' + date_time)
        
        self.base_path = config['base_path']
        self.stamp = config['stamp']
        self.meta_epochs = config['num_epochs']
        self.early_stopping = config['early_stopping']
        self.stopping_threshold = config.get('stopping_threshold', 1e-3)

        self.model = SeqSupervisedNetwork(config)

        logger.info('Supervised network instantiated')

    def training(self, train_dataloader, val_dataloader):
        best_loss = float('inf')
        best_f1 = 0
        patience = 0
        model_path = os.path.join(self.base_path, 'saved_models', 'Supervised-{}.h5'.format(self.stamp))
        classifier_path = os.path.join(self.base_path, 'saved_models', 'SupervisedClassifier-{}.h5'.format(self.stamp))
        logger.info('Model name: Supervised-{}.h5'.format(self.stamp))
        for epoch in range(self.meta_epochs):
            logger.info('Starting epoch {}/{}'.format(epoch + 1, self.meta_epochs))
            avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1 = self.model(train_dataloader)

            logger.info('Train epoch {}: Avg loss = {:.5f}, avg accuracy = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_loss, avg_accuracy,
                                                                            avg_precision, avg_recall, avg_f1))

            self.tensorboard_writer.add_scalar('Loss/train', avg_loss, global_step=epoch + 1)
            self.tensorboard_writer.add_scalar('F1/train', avg_f1, global_step=epoch + 1)

            avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1 = self.model(val_dataloader, 
                                                                                   testing=True)

            logger.info('Val epoch {}: Avg loss = {:.5f}, avg accuracy = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_loss, avg_accuracy,
                                                                            avg_precision, avg_recall, avg_f1))

            self.tensorboard_writer.add_scalar('Loss/val', avg_loss, global_step=epoch + 1)
            self.tensorboard_writer.add_scalar('F1/val', avg_f1, global_step=epoch + 1)
            
            if avg_f1 > best_f1 + self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                best_f1 = avg_f1
                logger.info('Saving the model since the F1 improved')
                torch.save(self.model.learner.state_dict(), model_path)
                torch.save(self.model.classifier.state_dict(), classifier_path)
                
                logger.info('')
            else:
                patience += 1
                logger.info('F1 did not improve')
                logger.info('')
                if patience == self.early_stopping:
                    break

            # Log params and grads into tensorboard
#             for name, param in self.model.named_parameters():
#                 if param.requires_grad and param.grad is not None:
#                     self.tensorboard_writer.add_histogram('Params/' + name, param.data.view(-1),
#                                                      global_step=epoch + 1)
#                     self.tensorboard_writer.add_histogram('Grads/' + name, param.grad.data.view(-1),
#                                                      global_step=epoch + 1)

        self.model.learner.load_state_dict(torch.load(model_path))
        self.model.classifier.load_state_dict(torch.load(classifier_path))
        return best_f1

    def testing(self, test_dataloader):
        logger.info('---------- Supervised testing starts here ----------')
        _, accuracy, precision, recall, f1_score = self.model(test_dataloader, 
                                                              testing=True)

        logger.info('Avg meta-testing metrics: Accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(accuracy,
                                               precision,
                                               recall,
                                               f1_score))
        return f1_score


# In[21]:


learner = SupervisedNetwork(config)


# In[ ]:


learner.training(train_dataloader, dev_dataloader)
logger.info('Supervised learning completed')


# In[ ]:


learner.testing(test_dataloader)
logger.info('Supervised testing completed')

