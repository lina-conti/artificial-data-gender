import argparse
import yaml
import pandas as pd
import os.path
from model import LanguageModel
from data import Dataset
from grid_search import GridSearch
from cmd import Cmd
from tqdm import tqdm
"""
This is a command line interface for running a trained model given an hyperparameter file
and a parameter file.

The interface may also be used to train a single model given an hyperparameter file
"""

def unlist(value):
    """
    A function for preprocessing hyperparams values from the Yaml file
    Args:
        value (list or primitive  type): an hyperparam
    Returns:
        a primitive type or raises an error
    """
    if type(value) == list:
        if len(value) > 1:
            raise ValueError('Trying to perform predictions from an underspecified model file')
        return value[0]
    else:
        return value

def load_transformer_model(model_dir,cpu=False):
    """
    Loads an encoding and a model from model_dir
    Args:
     model_dir (path): the path leading to the model
    Returns:
     A couple (encoder (DataSet), LanguageModel)
    """
    yaml_file = os.path.join(model_dir, 'model.yaml')
    istream = open(yaml_file)
    hyper = yaml.safe_load(istream)
    hyper = {key: unlist(value) for key, value in hyper.items()}
    istream.close()
    encoder = Dataset('', parentencoding=model_dir)
    lm = LanguageModel(encoder,
                       hyper['context_model'],
                       hyper['model_input_size'],
                       hyper['model_output_size'],
                       hyper['num_layers'],
                       nheads=hyper['nheads'],
                       ffn_hidden=hyper['ffn_hidden'],
                       dropout=hyper['dropout'],
                       tie_weights=hyper['tie_weights'],
                       verbose=True)
    lm.load_params(model_dir,cpu)
    return (encoder,lm)





parser = argparse.ArgumentParser(description="""
Train or run a Neural Language model. 
If TRAINING_FILE and VALIDATION_FILE files are provided, trains a model. 
If TEST_FILE is provided, performs predictions. Otherwise provides an interface to generate random text

MODEL_DIR is the name of the directory where every file related to the model is stored. 
Before training it must contain an hyperparameter file called model.yaml 
""")

parser.add_argument('model_dir', metavar='MODEL_DIR', type=str,help='A directory name where the model is/will be stored')
parser.add_argument('--train_file', metavar='TRAINING_FILE', type=str,help='a training text file')
parser.add_argument('--valid_file', metavar='VALIDATION_FILE', type=str,help='a validation text file')
parser.add_argument('--test_file', metavar='TEST_FILE', type=str,help='a test text file')
parser.add_argument('--probe_ex_file', metavar='EX_FILE', type=str,help='a text file to be turned into examples for a probe')
parser.add_argument('--out_pickle', metavar='PICKLE_FILE', type=str,help='a pickle file where probing examples should be saved')
parser.add_argument('--device_name', metavar='DEVICE', type=str,nargs="+",default='cpu', help='a valid device identifier (defaults to cpu)')


if __name__ == '__main__':

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    device_list = args.device_name
    if type(device_list) != list:
        device_list = [device_list]
    device_list = [device.split(':')[-1] for device in device_list if device != 'cpu']

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(device_list)

    if args.train_file and args.valid_file: #training mode

        yaml_file = os.path.join(args.model_dir,'model.yaml')
        grid = GridSearch(yaml_file)
        grid.search(args.train_file,args.valid_file,args.model_dir,device_list)

    elif args.test_file: #prediction mode
        yaml_file = os.path.join(args.model_dir,'model.yaml')
        istream = open(yaml_file)
        hyper   = yaml.safe_load(istream)
        hyper   = {key:unlist(value) for key,value in hyper.items()}
        istream.close()
        testset = Dataset(args.test_file, parentencoding=args.model_dir)
        lm = LanguageModel(testset,
                       hyper['context_model'],
                       hyper['model_input_size'],
                       hyper['model_output_size'],
                       hyper['num_layers'],
                       nheads=hyper['nheads'],
                       ffn_hidden=hyper['ffn_hidden'],
                       dropout=hyper['dropout'],
                       tie_weights=hyper['tie_weights'],
                       positional=hyper['positional']).to(0)
        lm.load_params(args.model_dir)
        pd.set_option('display.max_rows', None)
        for df in lm.predict(testset,hyper['batch_size'],device=0):
            print(df)
            print()
        print('Test perplexity',lm.validate(testset,hyper['batch_size'],hyper['bptt_chunk'],device=0)[1])

    elif args.probe_ex_file and args.out_pickle: #probe examples mode

        yaml_file = os.path.join(args.model_dir,'model.yaml')
        istream = open(yaml_file)
        hyper   = yaml.safe_load(istream)
        hyper   = {key:unlist(value) for key,value in hyper.items()}
        istream.close()
        exset = Dataset(args.probe_ex_file, parentencoding=args.model_dir)
        lm = LanguageModel(exset,
                       hyper['context_model'],
                       hyper['model_input_size'],
                       hyper['model_output_size'],
                       hyper['num_layers'],
                       nheads=hyper['nheads'],
                       ffn_hidden=hyper['ffn_hidden'],
                       dropout=hyper['dropout'],
                       tie_weights=hyper['tie_weights'],
                       positional=hyper['positional']).to(0)
        lm.load_params(args.model_dir)
        all_examples = []
        for examples in tqdm(lm.get_examples_probe(exset,hyper['batch_size'],device=0), total=len(exset.sentences)):
            all_examples += examples
         # vectors correspond to the hidden output vectors of the transformer encoder (columns called vectors for retrocompatibility)
        pd.DataFrame(columns=['tokens', 'vectors', 'embeddings'], data=all_examples).to_pickle(args.out_pickle)
        print(f'Examples saved to {args.out_pickle}')
        print('Model perplexity on this dataset: ',lm.validate(exset,hyper['batch_size'],hyper['bptt_chunk'],device=0)[1])  

    else: #random generation mode
        class Generator(Cmd):
            prompt = 'nnlm> '
            intro = "Type the beginning of a sentence and let the model generate the rest"
            def __init__(self,model, dataencoder,device):
                super(Generator, self).__init__()
                self.model = model
                self.encoder = dataencoder
                self.device = device

            def default(self, text):
                if text in ['exit','quit']:
                    exit(0)
                gtext = self.model.generate(self.encoder, text.split(),device=self.device)
                print(' '.join(gtext))

        yaml_file = os.path.join(args.model_dir, 'model.yaml')
        istream = open(yaml_file)
        hyper = yaml.safe_load(istream)
        hyper = {key: unlist(value) for key, value in hyper.items()}
        istream.close()
        encoder = Dataset('', parentencoding=args.model_dir)
        lm = LanguageModel(encoder,
                       hyper['context_model'],
                       hyper['model_input_size'],
                       hyper['model_output_size'],
                       hyper['num_layers'],
                       nheads=hyper['nheads'],
                       ffn_hidden=hyper['ffn_hidden'],
                       dropout=hyper['dropout'],
                       tie_weights=hyper['tie_weights'],
                       positional=hyper['positional']).to(0)
        lm.load_params(args.model_dir)
        Generator(lm,encoder,0).cmdloop()

