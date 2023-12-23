import itertools
import csv
import fire
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tokenization
import models as models
import optim
import train as train
from utils import set_seeds, get_device, truncate_tokens_pair


class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None
    def __init__(self, file, pipeline=[]):
        Dataset.__init__(self)
        data = []
        with open(file, "r", encoding="cp437", errors='ignore') as f:
           
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance in self.get_instances(lines): 
                for proc in pipeline: 
                    instance = proc(instance)
                data.append(instance)

        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        raise NotImplementedError


class MRPC(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): 
            yield line[0], line[3], line[4]

class COLA(CsvDataset):
    """ Dataset class for COLA """
    labels = ("0", "1") 
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None):
            yield line[1], line[3] 

def dataset_class(task):
    """ Mapping from task string to Dataset Class """
    table = {'mrpc': MRPC}
    return table[task]


class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor 
        self.tokenize = tokenize

    def __call__(self, instance):
        if(len(instance) == 3):
        	label, text_a, text_b = instance
        else:
        	label, text_a = instance
        	text_b=[]

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) if text_b else []

        return (label, tokens_a, tokens_b)


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer 
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) 
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, label_id)


class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits

def main(task='mrpc',
         train_cfg='config/train_cola.json',
         model_cfg='config/bert_base.json',
         data_file='data.tsv',
         model_file=None,
         pretrain_file='weight.pt',
         data_parallel=False,
         vocab="C:/Users/prbhatnagar/Downloads/Final_Project/py_src/uncased_L-12_H-768_A-12/bert-base-uncased-vocab.txt",
         save_dir='mrpc',
         max_len=128,
         mode='eval'):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_class(task)
    
    print('Dataset and Tokenizer loaded successfully')
    print()

    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(max_len),
                TokenIndexing(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, max_len)]
    dataset = TaskDataset(data_file, pipeline)
    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    print('Pipeline implemented successfully')
    print()

    model = Classifier(model_cfg, len(TaskDataset.labels))
    criterion = nn.CrossEntropyLoss()
    print('Model created successfully')
    print()

    trainer = train.Trainer(cfg,
                            model,
                            data_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device())

    state_dict = torch.load(pretrain_file, map_location =torch.device('cpu'))
    trainer.model.load_state_dict(state_dict, strict = False)
    print('Trainer created and weights loaded successfully')
    print()

    if mode == 'eval':
        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            print(' Starting inference')
            print()
            print('BERT says HI!')
            print()

            initial_time_seconds = time.time()
            initial_time_milliseconds = int(initial_time_seconds * 1000)
         
            logits = model(input_ids, segment_ids, input_mask)

            final_time_seconds = time.time()
            final_time_milliseconds = int(final_time_seconds * 1000)
            print()
            
            print(f"Total Inference time (GPU) in Milliseconds: {final_time_milliseconds - initial_time_milliseconds}")

            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float()
            accuracy = result.mean()
            return accuracy, result

        results = trainer.eval(evaluate, model_file, data_parallel)


if __name__ == '__main__':
    fire.Fire(main)
