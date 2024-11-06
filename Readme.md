## Multimodal Depression Detection

This repository contains code and instructions to correctly implement the Bi-LSTM, GRU & Fuse_net models which were proposed in [**THIS**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746569) research paper published in ICASSP 2022.

Since, the code provided by the author is buggy & doesn't work right away and requires other files like vggish_input.py, vggish_params.py, vggish_postprocess.py, vggish_slim.py, loupe_keras.py, etc..

And these codes also required code changes to work on latest version of tf, keras.

**NOTE :** A lot of the functional code has been rewritten by me, for resolving the errors that I came across in author's code, some obvious modifications to reduce complexity and make it easier to train and infer.

### Steps For Model Evaluation

Ensure that required packages as mentioned in `requirements.txt` are installed on your device.

For **classification models**:

Since, the pretrained models are already stored in ./Model folder, `./Classification/AudioModelChecking.ipynb`, `./Classification/TextModelChecking.ipynb` can be used for evaluating Audio GRU and Text Bi-LSTM models.

Similarly, `./Classification/fuse_model_checking.ipynb` can be used for evaluating the fuse model.

**Model Evaluation Results** : 

__Text Bi-LSTM Model__:

```
precison: 0.7838920261123793 
recall: 0.5851851851851851 
f1 score: 0.6606730460153797
```
__Audio GRU Model__:

```

```

__Fuse Net Model__:
```
precison: 0.6551062091503268 
recall: 0.7508417508417509 
f1 score: 0.6788445472655998
```

**Modified Model Results** :

After changing the text embeddings from ELMo to BERT, the text Bi-LSTM model converges much faster and gives more consistent results in the regression test along with some improvements in the MAE as mentioned below :-

Text 1 : 
ELMO Embeddings - MAE : 8.42
BERT Tokenizer - MAE : 7.94
Roberta - MAE : 7.71

Text 2 : 
ELMO Embeddings - MAE : 8.23
BERT Tokenizer - MAE : 7.60
Roberta - MAE : 7.58

Text 3 : 
ELMO Embeddings - MAE : 7.64
BERT Tokenizer - MAE : 7.39
Roberta - MAE : 7.37


__Results for Fusenet__

Fuse 1 :
ELMO Embeddings(in text BiLSTM) - MAE : 7.82
BERT Tokenizer(in text BiLSTM) - MAE : 7.34
Roberta(in text BiLSTM) - MAE : 7.22

Fuse 2 :
ELMO Embeddings(in text BiLSTM) - MAE : 8.24
BERT Tokenizer(in text BiLSTM) - MAE : 8.22
Roberta(in text BiLSTM) - MAE : 7.97

Fuse 3 :
ELMO Embeddings(in text BiLSTM) - MAE : 7.52
BERT Tokenizer(in text BiLSTM) - MAE : 7.93
Roberta(in text BiLSTM) - MAE : 7.98


i.e. Initially with ELMo embeddings, the model results were not consistent and required multiple runs to get the best results on the particularly selected random indexes. But with Transformer embeddings for the text, the results are much more consistent and do not require multiple runs.

### Steps For Model Training

Ensure that required packages as mentioned in `requirements.txt` are installed on your device.

Step 1 can be skipped entirely as well, since the .npz files are already provided in this repository.

Step 1 : Run `text_features_whole.ipynb` , `audio_features_whole.ipynb` to get the required .npz files for audio and text regression and classification models.

```
If you encounter error displaying: 
`Highway.forward: return type <class 'torch.Tensor'> is not a <class 'NoneType'>`

then modify forward function in `highway.py` and add `@overrides(check_signature=False)` instead of `@overrides`.
```

Step 2 : For **regression models**:

Run `text_bilstm_perm.ipynb` & `audio_bilstm_perm.ipynb` which will save the model weights as well.

Step 3 : `AudioModelChecking.ipynb` can be used for evaluating the models, if the model accuracy is low redo the step 2.

Step 4 : Then, after you have got the weights from audio and text models, run `Fuse_net.ipynb` to get the fused model.


Step 5 : For **classification models**:

Step 6 : Run `text_bilstm_whole.ipynb` & `audio_gru_whole.ipynb` which will save the models in **../Models/ClassificationWhole** folder.

Step 7 : `AudioModelChecking.ipynb`, `TextModelChecking.ipynb` can be used for evaluating these models.

Step 8 : Then, for fused model training, run `fuse_net_whole.ipynb` which will save the model in `Fuse` directory.

Step 9 : Similarly, `fuse_model_checking.ipynb` can be used for evaluating the fuse model.