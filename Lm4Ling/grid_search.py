import sys
import os,os.path
import yaml
import logging
import model
import data

def run(task,trainset,validset,model_dir,devicelist):
    tmp_dir = os.path.join(model_dir,'tmp_dir')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    logger = logging.getLogger('lm_log')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt='%(message)s'))
    logger.addHandler(handler)
    logger.info('-' * 80)
    logger.info('Running task %s'%(str(task),))
    lm = model.LanguageModel(trainset,
                task['context_model'],
                task['model_input_size'],
                task['model_output_size'],
                task['num_layers'],
                nheads=task['nheads'],
                ffn_hidden=task['ffn_hidden'],
                dropout=task['dropout'],
                tie_weights=task['tie_weights'],
                positional=task['positional'])
    logger.info('\nThe language model has %d parameters.\n' % (lm.num_parameters()))
    LR = task['learning_rate']
    try:
        lm.train_model( trainset,
                    validset,
                    task['batch_size'],
                    task['bptt_chunk'],
                    task['epochs'],
                    task['warmup_epochs'],
                    task['warmup_batch_size'],
                    task['restart_cycles'],
                    logger,
                    devicelist,
                    lr = LR,
                    modeldir=tmp_dir)
        return lm.validate(validset, task['batch_size'], task['bptt_chunk'], device=0)
    except RuntimeError as e:
        logger.error('Out of memory', e)
        logger.error('Training failed. skipping this configuration: %s'%(str(task)))
    except ValueError as e:
        logger.error(e)
        logger.error('Training failed. skipping this configuration: %s'%(str(task)))


class GridSearch:
    """
    Performs Grid search from params specified in a yaml file.
    """
    def __init__(self,yamlfile):
        istream = open(yamlfile)
        self.params = yaml.safe_load(istream)
        istream.close()

    def generate_task_list(self):
        tasklist = [{}]
        for param in self.params.keys():
            uptasklist = []
            for task in tasklist:
                values = self.params[param]
                if type(values) != list:
                    values = [values]
                for val in values:
                    uptask = task.copy()
                    uptask[param] = val
                    uptasklist.append(uptask)
            tasklist = uptasklist
        print('There are %d model(s) to train.' % (len(tasklist)))
        return tasklist

    def search(self,trainfilename,validfilename,model_dir,device_list):

        print('Reading data')
        trainset = data.Dataset(trainfilename, max_vocab_size=self.params['max_vocab_size'])
        validset = data.Dataset(validfilename,unk=trainset.unk_token,parentencoding=trainset.encoding)
        trainset.save(model_dir)

        print('Searching...')
        best_ppl = 1000000000000
        best_task = None
        for task in self.generate_task_list():
            nll,ppl = run(task,trainset,validset,model_dir,device_list)
            if ppl < best_ppl:
                best_ppl    = ppl
                best_task = task
                src  = os.path.join(model_dir,'tmp_dir','lm_params.pt')
                dest = os.path.join(model_dir,'lm_params.pt')
                os.system('cp %s %s'%(src,dest))

        print('\n\nBest configuration found', best_task)
        print('Perplexity %.5f' % (best_ppl,))
        print('done.')


