## Multimodal Depression Detection

This repository contains code and instructions to correctly implement the Bi-LSTM, GRU & Fuse_net models which were proposed in [**THIS**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746569) research paper published in ICASSP 2022.

Since, the code provided by the author is buggy & doesn't work right away and requires other files like vggish_input.py, vggish_params.py, vggish_postprocess.py, vggish_slim.py, loupe_keras.py, etc..

And these codes also required code changes to work on latest version of tf, keras.

**NOTE :** A lot of the functional code has been rewritten by me, for resolving the errors that I encountered in author's code, some obvious modifications to reduce complexity and make it easier to train and infer.

### Steps

Ensure that required packages as mentioned in `requirements.txt` are installed on your device.

Then, run `text_features_whole.ipynb` , `audio_features_whole.ipynb` to get the required .npz files for audio and text regression and classification models.

```
If you encounter error displaying: 
`Highway.forward: return type <class 'torch.Tensor'> is not a <class 'NoneType'>`

then modify forward function in `highway.py` and add `@overrides(check_signature=False)` instead of `@overrides`.
```

Then, for regression models:

Run `text_bilstm_perm.ipynb` & `audio_bilstm_perm.ipynb` which will save the model weights as well.

`AudioModelChecking.ipynb` can be used for evaluating the models.

Then, after you have got the weights from audio and text models, run `Fuse_net.ipynb` to get the fused model.


For **classification models**:

Run `text_bilstm_whole.ipynb` & `audio_gru_whole.ipynb` which will save the models in **../Models/ClassificationWhole** folder.

`AudioModelChecking.ipynb`, `TextModelChecking.ipynb` can be used for evaluating these models.

Then, for fused model training, run `fuse_net_whole.ipynb` which will save the model in `Fuse` directory.

Similarly, `fuse_model_checking.ipynb` can be used for evaluating the fuse model.