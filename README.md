# Text classification model using Tensorflow 2.0 and Tensorflow serving
 
 model.py contains final classifier training Pipeline
 
 Graphs, comments, and further steps discussion can be found here:
 https://colab.research.google.com/drive/1YKjTJhYOI8CvrkPUXqeD-JQkKtQffSIL
  

### File Structure:
```
.
|--model.py
|--predict_online.py
|--customer_chat_sample.csv
|--classifier
|  |--saved_models
|     |--model
|        |--keras_metadata.pb
|        |--saved_model.pb
|        |--assets
|           |--bert_vocab_500k.external.111lang.txt
|           |--cased_vocab.txt
|        |--variables
|           |--variables.data-00000-of-00001
|           |--variables.index                       
|--requirements.txt
```

### Usage Requirements:

Install all Python dependencies using `pip install -r requirements.txt`

To train the model and output the SavedModel object, run `python model.py`. 
This will output the SavedModel files to `classifier/saved_models`.

Google colab was used to train models since it provides GPU.
In order to run provided notebook upload `customer_chat_sample.csv` to Google colab.


### TensorFlow serving [supposed docker is installed and running]

In terminal run:

docker run -p 8501:8501 --mount type=bind,source=~/Code/Projects/DL_Classifier/classifier/saved_models/model,target=/models/classifier/1 -e MODEL_NAME=classifier -t tensorflow/serving

```
docker run -p 8501:8501 --mount type=bind,source=<HERE_INSERT_FULL_PATH_TO_MODEL>,target=/models/classifier/1 -e MODEL_NAME=classifier -t tensorflow/serving
```

Run `python predict_online.py`


### Note

`classifier` folder already contains trained model -> file `predict_online.py` can be run without training. After running `model.py` this folder will be replaced with a newly trained model.

