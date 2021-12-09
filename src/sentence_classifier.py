import yaml
import sys
import torch
import getopt
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel ,BertForSequenceClassification
from Utils.utils import *
from Utils.finetune import *


# Constants
LABELS = {'background': 0, 'objective': 1, 'methods': 2, 'results': 3, 'conclusions': 4}
NUM_LABELS = len(LABELS)
COLUMNS = {'abstract', 'sentence', 'label'}


def main(method, cfg):
    if method == 'train':
        # get all the training data for preparaing the label set and store the labels for future use
        df = preprocess_data(cfg['data']['train'])

        sentences = df.sentence.values
        labels = df.label.values

        #tokenizer = BertTokenizer.from_pretrained(''emilyalsentzer/Bio_ClinicalBERT'', do_lower_case=True)
        #tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        training_inputs, training_masks = get_encoded_data(tokenizer, sentences)

        # convert to torch tensors
        tensor_inputs, tensor_labels, tensor_masks = get_torch_tensors(training_inputs, labels, training_masks)
        batch_size = cfg['hyperparams']['batch_size']

        # Create the DataLoader for our training set.
        train_data = TensorDataset(tensor_inputs, tensor_masks, tensor_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        model, scheduler, optimizer = initialise_model(cfg, train_dataloader)
        epochs = cfg['hyperparams']['epochs']
        model = finetunemodel(model, scheduler, epochs, train_dataloader, optimizer)
        save_model(model, cfg, tokenizer)
        sys.exit(1)

    elif method == 'eval':

        # load the labels from training data:label is key
        df = preprocess_data(cfg['data']['test'])
        sentences = df.sentence.values
        labels = df.label.values

        #tokenizer = BertTokenizer.from_pretrained(''emilyalsentzer/Bio_ClinicalBERT'', do_lower_case=True)
        #tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        #tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        test_inputs, test_masks = get_encoded_data(tokenizer, sentences)

        # convert to torch tensors
        tensor_inputs, tensor_labels, tensor_masks = get_torch_tensors(test_inputs, labels, test_masks)
        batch_size = cfg['hyperparams']['batch_size']

        # Create the DataLoader for our testing set.
        test_data = TensorDataset(tensor_inputs, tensor_masks, tensor_labels)
        test_sampler = SequentialSampler(test_data)
        prediction_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


        #load saved  model for eval
        #model = BertForSequenceClassification.from_pretrained(cfg['data']['finetuned_model'])
        model = BertForSequenceClassification.from_pretrained(cfg['pruning']['model_name'])
        
        head_mask = {0: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11], 1: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10], 2: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 3: [1, 3, 4, 6, 7, 9, 10, 11], 4: [0, 1, 2, 4, 5, 6, 7, 8, 10], 5: [0, 1, 2, 3, 5, 9, 11], 6: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11], 7: [2, 4, 5, 6, 9, 10, 11], 8: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10], 9: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], 10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11: [0, 2, 3, 6, 7, 8, 9, 11]}
        head_mask = np.load(head_mask)
        head_mask = torch.from_numpy(head_mask)
        heads_to_prune = {}
        for layer in range(len(head_mask)):
            heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
            heads_to_prune[layer] = heads_to_mask
        assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
        model.prune_heads(heads_to_prune)
        
        model.eval()

        print_classification_report(prediction_dataloader,model,LABELS)

    else:
        print("Please check the argument . Expected value train or eval")
        sys.exit(1)


if __name__ == "__main__":
    try:
        options, args = getopt.getopt(sys.argv[1:], "mh", ["mode="])
        for name, value in options:
            if name in ('-m', '--mode'):
                mode = value
                assert mode == "train" or mode == "eval"
            if name in ('-h', '--help'):
                print ('python sentence_classifier.py --mode eval\\train ')
                sys.exit(1)
    except getopt.GetoptError as err:
        print("Seems arguments are wrong..")
        print("usage:: python sentence_classifier.py --mode eval\\train")
        print ("Ex:: python sentence_classifier.py --mode eval")
        sys.exit(1)

    with open('../config/config.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    main(mode,config)
