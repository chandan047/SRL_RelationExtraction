#!/usr/bin/env python
# coding: utf-8

# ### incomplete!!!

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"  


# In[2]:


import json
import random
import re
import numpy as np
from os import path
import pdb
import pickle as pkl
from sesame.evaluation import *
from sesame.dataio import *
from sesame.globalconfig import VERSION, TRAIN_FTE, UNK, DEV_CONLL, TEST_CONLL, FRAME_DIR, EMPTY_FE
from sesame.conll09 import lock_dicts, post_train_lock_dicts
from sesame.housekeeping import filter_long_ex, merge_span_ex
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hier", type=bool, default=False, help="Use phrase structure features.")
args = parser.parse_args("")


# In[5]:


train_conll = TRAIN_FTE
dev_conll = DEV_CONLL
test_conll = TEST_CONLL


# In[6]:


train_examples, _, _ = read_conll(train_conll)


post_train_lock_dicts()

frmfemap, corefrmfemap, _ = read_frame_maps()

if args.hier:
    frmrelmap, feparents = read_frame_relations()

lock_dicts()
NOTANFEID = FEDICT.getid(EMPTY_FE)  # O in CoNLL format.
USE_SPAN_CLIP = True
ALLOWED_SPANLEN = 20

# merge_span_ex(train_examples, NOTANFEID)
train_examples = filter_long_ex(train_examples, USE_SPAN_CLIP, ALLOWED_SPANLEN, NOTANFEID)

# In[7]:


def read_fes_definitions(frame_file):
    with open(frame_file, "rb") as f:
        tree = et.parse(f)
        root = tree.getroot()

        frcount = 0
        for frame in root.iter('{http://framenet.icsi.berkeley.edu}frame'):
            framename = frame.attrib["name"]
            frid = FRAMEDICT.getid(framename)
            frcount += 1

        if frcount > 1:
            raise Exception("More than one frame?", frame_file, framename)

        fes = {}
        for fe in root.findall('{http://framenet.icsi.berkeley.edu}FE'):
            fename = fe.attrib["name"]
            feid = FEDICT.getid(fename)
            definition = fe.find('{http://framenet.icsi.berkeley.edu}definition').text
            definition = definition.split('<ex>')[0].strip()
            definition = definition.split('\n')[0].strip()
            definition = re.sub('<[^<]+>', "", definition)
            
            fenamesplit = " ".join(fename.split('_'))
            fes[feid] = "If " + fenamesplit + " is " + definition + ", then [MASK] is the " + fenamesplit + "?"

    return frid, fes

# test
# read_fes_definitions(os.path.join(FRAME_DIR, "Employing.xml"))


# In[8]:


def read_frame_definitions():
    sys.stderr.write("\nReading the frame definitions from {} ...\n".format(FRAME_DIR))

    frmfedefmap = {}
    maxfesforframe = 0
    longestframe = None

    for f in tqdm(os.listdir(FRAME_DIR)):
        framef = os.path.join(FRAME_DIR, f)
        if framef.endswith("xsl"):
            continue
        frm, fe_def = read_fes_definitions(framef)
        frmfedefmap[frm] = fe_def
        if len(frmfedefmap[frm]) > maxfesforframe:
            maxfesforframe = len(frmfedefmap[frm])
            longestframe = frm

    sys.stderr.write("Max! {} frame-elements for frame: {}\n\n".format(maxfesforframe, FRAMEDICT.getstr(longestframe)))
    return frmfedefmap


DEF_SAVEFILE = 'saved_definitions.pkl'
if path.exists(DEF_SAVEFILE):
    with open(DEF_SAVEFILE, 'rb') as f:
        frmfedefmap = pkl.load(f)
else:
    frmfedefmap = read_frame_definitions()
    with open(DEF_SAVEFILE, 'wb') as f:
        pkl.dump(frmfedefmap, f)

# In[9]:


dev_examples, _, _ = read_conll(dev_conll)

test_examples, _, _ = read_conll(test_conll)


# In[10]:


print ("Total number of distinct frames", FRAMEDICT.size())
print ("Total number of dictinct frame elements", FEDICT.size())


# In[14]:


from transformers import BertTokenizer, AlbertTokenizer, AlbertForQuestionAnswering
MODEL_NAME = 'mrm8488/spanbert-finetuned-squadv2'
bert_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


# In[15]:


class DataInstance:
    def __init__(self, ex, input_tokens, segment_ids, 
                 span_start, span_end, 
                 word_start_mask, word_end_mask,
                 sent_len, max_seq_len):
        self.ex = ex
        
        self.input_tokens = input_tokens  # [[sent::ques], [sent::ques], ...]        num of FE's for the frame
        self.segment_ids = segment_ids    # [[0, 0, ..., 1, 1], [0, 0, ..., 1, 1]]   num of FE's for the frame
        self.span_start = span_start      # [0, 0, 2, ...]                           sentence length
        self.span_end = span_end          # [0, 3, 5, ...]                           sentence length
        self.sent_len = sent_len
        self.word_start_mask = word_start_mask
        self.word_end_mask = word_end_mask
        self.max_seq_len = max_seq_len


class CustomDataLoader(object):
    def __init__(self, examples):
        self.instances = self.process_batches(examples)
    
    def process_batches(self, examples):
        instances = []

        n_total_fes = 0
        n_fes_with_multispans = 0
        
        for index, ex in enumerate(tqdm(examples)):
            instance_fes = []
            ques_tokens = []
            segment_ids = []
            span_start = []
            span_end = []
            sent_len = -1
            max_seq_len = -1

            words = [VOCDICT.getstr(tok) for tok in ex.tokens]
            frame = ex.frame
            tfdict = ex.targetframedict
            valid_fes = [NOTANFEID] + frmfemap[frame.id]
            
            fes = ex.invertedfes
            aligned_fes = {}
            for fe in fes:
                aligned_fes[fe] = []
                for span in fes[fe]:
                    span_copy = (span[0]+1, span[1]+1)   # accounting for the [CLS] token
                    aligned_fes[fe].append(span_copy)
            
            sent_tokens = []
            word_start_mask = [1]
            word_end_mask = [1]
            for i, word in enumerate(words):
                word_tokens = bert_tokenizer.tokenize(word)
                word_start_mask.append(len(sent_tokens) + 1)
                sent_tokens.extend(word_tokens)
                word_end_mask.append(len(sent_tokens))
                n_extra_tokens = len(word_tokens) - 1
                for fe in aligned_fes:
                    for span_idx in range(len(fes[fe])):
                        new_start = aligned_fes[fe][span_idx][0]
                        new_end = aligned_fes[fe][span_idx][1]
                        if i < fes[fe][span_idx][0]:
                            new_start += n_extra_tokens
                        if i <= fes[fe][span_idx][1]:
                            new_end += n_extra_tokens
                        aligned_fes[fe][span_idx] = (new_start, new_end)
            sent_len = len(sent_tokens)
            
            for fe in frmfemap[frame.id]:
            # for fe in fes:
                ques_i = ["CLS"] + sent_tokens + ["SEP"]
                seg_i = [0] * len(ques_i)
                ques_len = len(ques_i)
                for word in frmfedefmap[frame.id][fe].split(" "):
                    word_tokens = bert_tokenizer.tokenize(word)
                    ques_i.extend(word_tokens)
                    seg_i.extend([1] * len(word_tokens))
                    ques_len += len(word_tokens)
                ques_i += ["SEP"]
                ques_enc = bert_tokenizer.encode(ques_i, add_special_tokens=False)
                seg_i.append(1)
                ques_len += 1
                ques_tokens.append(ques_enc)
                segment_ids.append(seg_i)
                if fe in aligned_fes:
                    n_total_fes += 1
                    if (len(aligned_fes[fe]) > 1):
                        n_fes_with_multispans += 1
                        '''
                        sys.stderr.write("## More than one span for {} ##\n".format(FEDICT.getstr(fe)))
                        for idx, (span_o, span_n) in enumerate(zip(fes[fe], aligned_fes[fe])):
                            print ("sent_len: {}, span_o: {}, argument_o: {}"
                                   .format(sent_len, span_o, " ".join(words[span_o[0]: span_o[1] + 1]))
                                  )
                            print ("sent_len: {}, span_n: {}, argument_n: {}"
                                   .format(sent_len, span_n, " ".join(ques_i[span_n[0]: span_n[1] + 1]))
                                  )
                        print ()
                        '''
                    arg_span = aligned_fes[fe][0]
                    span_start.append(arg_span[0])
                    span_end.append(arg_span[1])
                else:
                    span_start.append(0)
                    span_end.append(0)
                max_seq_len = max(max_seq_len, ques_len)
                
            instances.append(DataInstance(ex, ques_tokens, segment_ids, 
                                          span_start, span_end,
                                          word_start_mask, word_end_mask,
                                          sent_len, max_seq_len))
        
        print ("Proporation of multi-span fes {:.5f}".format(n_fes_with_multispans/n_total_fes))
        
        return instances

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        return self.instances[index]
        
# # Model

import collections

def make_arg_prediction(batch_start_logits, batch_end_logits, sent_lens):
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["start_index", "end_index", "start_logit", "end_logit"])
    
    batch_start_logits = batch_start_logits.detach().cpu().numpy()
    batch_end_logits = batch_end_logits.detach().cpu().numpy()
    
    predictions = []
    
    for (start_logits, end_logits, length) in zip(batch_start_logits, batch_end_logits, sent_lens):
        prelim_predictions = []
        start_logits, end_logits = start_logits[:length], end_logits[:length]
        bestk_start_inds = sorted(range(1, len(start_logits)), key=lambda i: start_logits[i])[-20:]
        bestk_end_inds = sorted(range(1, len(end_logits)), key=lambda i: end_logits[i])[-20:]
        for start_index in bestk_start_inds:
            for end_index in bestk_end_inds:
                if (start_index > end_index) or (end_index - start_index > 20): 
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logits[start_index],
                        end_logit=end_logits[end_index]))
        if not (0 in bestk_start_inds and 0 in bestk_end_inds):
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=0,
                    end_index=0,
                    start_logit=start_logits[0],
                    end_logit=end_logits[0]))
        
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)
        
        # if prelim_predictions[0].end_index is not 0:
        predictions.append((prelim_predictions[0].start_index-1, prelim_predictions[0].end_index-1))
        
    return predictions

# In[16]:


import torch
from torch import nn
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# In[17]:


torch.cuda.device_count(), torch.cuda.current_device()


# In[18]:


torch.manual_seed(42)
random.seed(42)


# In[19]:


import yaml

with open('config_fe_v3.yaml', 'r') as f:
    config = yaml.safe_load(f)
# config['base_path'] = '.'
# config['stamp'] = "spanbert-sqadv2-arg"
print ()
print (config)
print ()

# Span-bert loaders
train_loadername = "train_dataloader.pkl"
dev_loadername = "dev_dataloader.pkl"
test_loadername = "test_dataloader.pkl"

if not config['train']:
    train_dataloader = None
elif path.exists(train_loadername):
    with open(train_loadername, "rb") as f:
        train_dataloader = pkl.load(f)
        print (f"{train_loadername} loaded from cache")
else:
    train_dataloader = CustomDataLoader(train_examples)
    with open(train_loadername, "wb") as f:
        pkl.dump(train_dataloader, f)


if not config['train']:
    dev_dataloader = None
elif path.exists(dev_loadername):
    with open(dev_loadername, "rb") as f:
        dev_dataloader = pkl.load(f)
        print (f"{dev_loadername} loaded from cache")
else:
    dev_dataloader = CustomDataLoader(dev_examples)
    with open(dev_loadername, "wb") as f:
        pkl.dump(dev_dataloader, f)
        

if path.exists(test_loadername):
    with open(test_loadername, "rb") as f:
        test_dataloader = pkl.load(f)
        print (f"{test_loadername} loaded from cache")
else:
    test_dataloader = CustomDataLoader(test_examples)
    with open(test_loadername, "wb") as f:
        pkl.dump(test_dataloader, f)

print ()


# In[21]:


from sesame.evaluation import calc_f, evaluate_example_argid, evaluate_corpus_argid


import coloredlogs
import logging

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# In[32]:

from transformers import BertTokenizer, BertModel, BertForQuestionAnswering, AdamW, get_constant_schedule_with_warmup
from mytorch_crf import CRF
import torch.nn.functional as F

class SeqSupervisedNetwork(nn.Module):
    def __init__(self, config):
        super(SeqSupervisedNetwork, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.lr = config.get('lr', 3e-5)
        self.weight_decay = config.get('weight_decay', 0.0)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)

        self.learner = BertForQuestionAnswering.from_pretrained(MODEL_NAME)
        
        if config.get('trained_learner', False):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_learner'])
            ))
            logger.info('Loaded trained learner model {}'.format(config['trained_learner']))
        
        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)
        logger.info('Model loaded to {}'.format(self.device))
        
        self.initialize_optimizer_scheduler()

        
    def initialize_optimizer_scheduler(self):
        learner_params = [p for p in self.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(learner_params, lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=100)

        
    def vectorize(self, batch):
        with torch.no_grad():
            max_batch_len = batch.max_seq_len
            input_ids = torch.zeros((len(batch.input_tokens), max_batch_len)).long()
            input_masks = torch.zeros((len(batch.input_tokens), max_batch_len)).long()
            input_segs = torch.zeros((len(batch.input_tokens), max_batch_len)).long()
            for i, sent in enumerate(batch.input_tokens):
                input_ids[i, :len(sent)] = torch.LongTensor(sent)
                input_masks[i, :len(sent)] = torch.LongTensor([1]*len(sent))
                input_segs[i, :len(batch.segment_ids[i])] = torch.LongTensor(batch.segment_ids[i])
            batch_x = input_ids.to(self.device)
            batch_masks = input_masks.to(self.device)
            batch_segs = input_segs.to(self.device)
        
        span_start = torch.LongTensor(batch.span_start).to(self.device)
        span_end = torch.LongTensor(batch.span_end).to(self.device)
        
        return batch_x, batch_masks, batch_segs, span_start, span_end
    
    def forward(self, dataloader, testing=False):
        if not testing:
            self.train() 
        else:
            self.eval() 
        
        log_every = 500
        
        avg_loss = 0 
        avg_loss_log = None
        all_examples, all_predictions = [], []
        all_metrics = lmetrics = umetrics = tokmetrics = [0., 0., 0.]
        
        loader_t = tqdm(dataloader)
        for batch_id, batch in enumerate(loader_t):
            batch_x, batch_masks, batch_segs, span_start, span_end = self.vectorize(batch)
            batch_size = batch_x.shape[0]
            
            word_start_mask = torch.LongTensor(batch.word_start_mask).to(self.device)
            word_end_mask = torch.LongTensor(batch.word_end_mask).to(self.device)
            
            outputs = self.learner(
                input_ids=batch_x, 
                token_type_ids=batch_segs, 
                attention_mask=batch_masks,
                start_positions=span_start, 
                end_positions=span_end, 
                return_dict=True
            )
            
            loss = outputs.loss
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            if not testing:
                loss.backward()
                
            avg_loss += loss.item() * self.gradient_accumulation_steps

            if (batch_id + 1) % self.gradient_accumulation_steps == 0 and not testing:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
            
            with torch.no_grad():
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                # start_logits = F.softmax(start_logits, dim=-1)
                # end_logits = F.softmax(end_logits, dim=-1)
                
                start_logits = start_logits[:, :batch.sent_len+1]
                end_logits = end_logits[:, :batch.sent_len+1]

                start_logits = start_logits[:, word_start_mask]
                end_logits = end_logits[:, word_end_mask]
                
                preds = make_arg_prediction(start_logits, end_logits, [len(batch.ex.tokens) + 1] * batch_size)
                prediction = {fe:[pred] for fe, pred in zip(frmfemap[batch.ex.frame.id], preds) if pred[1] != -1}
                all_predictions.append(prediction)
                
                u, l, w = evaluate_example_argid(batch.ex._get_inverted_femap(), prediction, 
                                                 corefrmfemap[batch.ex.frame.id],
                                                 len(batch.ex.tokens), NOTANFEID)
                umetrics = np.add(umetrics, u)
                lmetrics = np.add(lmetrics, l)
                tokmetrics = np.add(tokmetrics, w)
                all_metrics = np.add(all_metrics, l)

                if (batch_id + 1) % log_every == 0:
                    avg_loss_log = avg_loss / log_every
                    _, _, uf1_score = calc_f(umetrics)
                    _, _, lf1_score = calc_f(lmetrics)
                    _, _, tokf1 = calc_f(tokmetrics)
                    loader_t.set_description('Batch {}/{}, [supervised]: Loss = {:.5f}, '
                        'uf1 = {:.5f}, lf1 = {:.5f}, tf1 = {:.5f}'.format(batch_id + 1, len(dataloader),
                        avg_loss_log, uf1_score, lf1_score, tokf1))
                    loader_t.refresh()
                    avg_loss = 0
                    umetrics = lmetrics = tokmetrics = [0., 0., 0.]


        # Calculate metrics
        precision, recall, f1_score = calc_f(all_metrics)

        if avg_loss_log is None:
            avg_loss_log = avg_loss / len(dataloader)

        return avg_loss_log, precision, recall, f1_score

# In[33]:


class SupervisedNetwork:
    def __init__(self, config):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.tensorboard_writer = SummaryWriter(log_dir='runs/args-without-target-' + date_time)
        
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
        model_path = os.path.join(self.base_path, 'saved_models', 'model-{}.h5'.format(self.stamp))
        logger.info('Model name: model-{}.h5'.format(self.stamp))
        for epoch in range(self.meta_epochs):
            if train_dataloader:
                random.shuffle(train_dataloader.instances)
                
            logger.info('Starting epoch {}/{}'.format(epoch + 1, self.meta_epochs))
            avg_loss, avg_precision, avg_recall, avg_f1 = self.model(train_dataloader)

            logger.info('Train epoch {}: Avg loss = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_loss,
                                                                            avg_precision, avg_recall, avg_f1))

            self.tensorboard_writer.add_scalar('Loss/train', avg_loss, global_step=epoch + 1)
            self.tensorboard_writer.add_scalar('F1/train', avg_f1, global_step=epoch + 1)
            
            avg_loss, avg_precision, avg_recall, avg_f1 = self.model(val_dataloader, 
                                                                                   testing=True)

            logger.info('Val epoch {}: Avg loss = {:.5f}, avg precision = {:.5f}, '
                        'avg recall = {:.5f}, avg F1 score = {:.5f}'.format(epoch + 1, avg_loss, 
                                                                            avg_precision, avg_recall, avg_f1))

            self.tensorboard_writer.add_scalar('Loss/val', avg_loss, global_step=epoch + 1)
            self.tensorboard_writer.add_scalar('F1/val', avg_f1, global_step=epoch + 1)
            
            if avg_f1 > best_f1 + self.stopping_threshold:
                patience = 0
                best_loss = avg_loss
                best_f1 = avg_f1
                logger.info('Saving the model since the F1 improved')
                torch.save(self.model.learner.state_dict(), model_path)
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
        return best_f1

    def testing(self, test_dataloader):
        # model_path = os.path.join(self.base_path, 'saved_models', 'model-{}.h5'.format(self.stamp))
        
        logger.info('---------- Supervised testing starts here ----------')
        _, precision, recall, f1_score = self.model(test_dataloader, 
                                                              testing=True)

        logger.info('Avg meta-testing metrics: precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(precision,
                                               recall,
                                               f1_score))
        return f1_score


# In[34]:


learner = SupervisedNetwork(config)


# In[35]:


if config['train']:
    learner.training(train_dataloader, dev_dataloader)
    logger.info('Supervised learning completed')


# In[ ]:


learner.testing(test_dataloader)
logger.info('Supervised testing completed')

