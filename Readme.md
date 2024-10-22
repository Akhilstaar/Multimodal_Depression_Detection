## Multimodal Depression Detection

This repository contains code and instructions to correctly implement the Bi-LSTM, GRU & Fuse_net models which were proposed in [**THIS**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746569) research paper published in ICASSP 2022.

Since, the code provided by the author is buggy & doesn't work right away and requires other files like vggish_input.py, vggish_params.py, vggish_postprocess.py, vggish_slim.py, loupe_keras.py, etc..

And these codes also required code changes to work on latest version of tf, keras.

**NOTE :** A lot of the functional code has been rewritten by me, for resolving the errors that I came across in author's code, some obvious modifications to reduce complexity and make it easier to train and infer.

### Steps For Model Evaluation

Ensure that required packages as mentioned in `requirements.txt` are installed on your device.

Due to size upload limit on github, 3 files `import_files/vggish/vggish_model.ckpt`,`import_files/zhs.model/encoder.pkl` and `import_files/zhs.model/token_embedder.pkl` are uploaded to google drive instead. So, before running the code, the files need to be put to their desired location.

Files Link : [drive.google.com/...](https://drive.google.com/drive/folders/1Y67ZzvZnPPX_5Z5DdXhXH-ELHq6e_aK2?usp=sharing)

For **classification models**:

Since, the pretrained models are already stored in ./Model folder, `./Classification/AudioModelChecking.ipynb`, `./Classification/TextModelChecking.ipynb` can be used for evaluating Audio GRU and Text Bi-LSTM models.

Similarly, `./Classification/fuse_model_checking.ipynb` can be used for evaluating the fuse model.

**Results** : 

__Text Bi-LSTM Model__:

```
precison: 0.7838920261123793 
recall: 0.5851851851851851 
f1 score: 0.6606730460153797
```
__Audio GRU Model__:


__Fuse Net Model__:
```
precison: 0.6551062091503268 
recall: 0.7508417508417509 
f1 score: 0.6788445472655998
```
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