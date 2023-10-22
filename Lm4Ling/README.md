# Lm4Ling

This is a script for neural netword based language modelling for linguistic 
purposes. It implements word-based language models and returns word based 
probability transition statistics. 

It implements RNN, LSTM and Transformer style architectures.
It comes with an example relying on wikitext-103 stored in 
the directory `example`.

Testing
--------
Given a trained model, stored in `MODEL_DIR` the script can be used to perform 
predictions on data. The data must have a one sentence per line
formatting and expects proper tokenization (white space indicates token
boundaries).

Here is an example for performing predictions on the wikitext 103 test file
whose `MODEL_DIR`is `wiki103` using `gpu 0`:
``` shell script
python nnlm.py wiki103 --test_file wiki.test.tokens --device_name cuda:0 
```
The prediction returns log probabilities with and without teacher forcing 
for each sentence in the test file. Here is an example for a short sentence:

```
        token   ref_next  pred_next   ref_prob  pred_prob
0       <bos>         By        The  -5.185022  -1.684012
1          By       July        the  -4.223794  -1.203219
2        July       1915          ,  -4.877973  -2.527704
3        1915        all          ,  -7.119631  -0.137419
4         all  seventeen         of  -9.482289  -2.280144
5   seventeen         of        men  -2.633155  -2.563609
6          of        the        the  -0.353900  -0.353900
7         the     German  battalion  -3.821953  -3.220526
8      German   Imperial      ships  -5.488371  -2.874173
9    Imperial       Navy       Navy  -2.395504  -2.395504
10       Navy       Type        had -11.419503  -1.758142
11       Type         UB         UB  -1.352640  -1.352640
12         UB         Is         II -12.430326  -0.646976
13         Is        had      <unk>  -4.976085  -1.818442
14        had       been       been  -0.507446  -0.507446
15       been  completed    shipped  -4.723439  -2.700507
16  completed          .          ,  -1.768721  -1.387439
17          .      <eos>        The  -2.984874  -1.603312
```

Training
--------
 
 A model is located in a directory called `MODEL_DIR`. 
 The first thing to do is to create it:
``` shell script
mkdir MODEL_DIR
cd MODEL_DIR
```
Inside the model directory, you have to create a file called `model.yaml` 
that contains the range of hyperparameters you want to search for
during training. The `wiki103` directory in this git is an example of such a directory
with example hyperparameters for training a standard LSTM model

Once created and assuming that `MODEL_DIR` is currently 
in your current working directory, the training procedure is launched 
as follows:
``` shell script
python nnlm.py wiki103 --train_file wiki.train.tokens --valid_file wiki.valid.tokens --device_name cuda:0  
``` 
where we use the `wikitext-103` train and validation files and we specify that the computations will take place 
on `gpu 0`. In case no device is specified, it defaults to `cpu` which is not the recommended usage for training

**Grid Search** for training the `model.yaml` may contain lists of values for some hyperparameters.
In which case, the trainer will perform a grid search by testing all 
the hyperparameter combinations. The `model_search.yaml` is an example of such 
a file. 

**Memory management** Training language models may result in memory blowup.
In case you get a `Runtime Error` with a memory problem. Try to 
reduce the value of the `batch_size` and `bptt_chunk` that represent the number of sequences in 
a batch and the maximum length of the sub-sequence processed by the truncated backpropagation through time algorithm. 

