# import torch
# from vncorenlp import VnCoreNLP
# from transformers import AutoModel, AutoTokenizer

# phobert = AutoModel.from_pretrained("vinai/phobert-base")

# # For transformers v4.x+: 
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# rdrsegmenter = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

# # Input 
# text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

# # To perform word (and sentence) segmentation
# sentences = rdrsegmenter.tokenize(text) 
# for sentence in sentences:
#     input_ids = torch.tensor([tokenizer.encode(" ".join(sentence))])
# with torch.no_grad():
#     features = phobert(input_ids)  # Models outputs are now tuples

#######################################
### -------- Load libraries ------- ###

# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
from sklearn.model_selection import train_test_split


#######################################
### --------- Import data --------- ###

# Import data from csv
data = pd.read_csv('data/mebe_tiki.csv')

# Select required columns
data = data[['cmt', 'giá', 'dịch_vụ', 'an_toàn', 'chất_lượng', 'ship', 'other', 'chính_hãng']]

# Remove a row if any of the three remaining columns are missing
data = data.dropna()

# Set your model output as categorical and save in new label col
data['Gia_label'] = pd.Categorical(data['giá'])
data['Dich_vu_label'] = pd.Categorical(data['dịch_vụ'])
data['An_toan_label'] = pd.Categorical(data['an_toàn'])
data['Chat_luong_label'] = pd.Categorical(data['chất_lượng'])
data['Ship_label'] = pd.Categorical(data['ship'])
data['Chinh_hang_label'] = pd.Categorical(data['chính_hãng'])


# Transform your output to numeric
data['giá'] = data['Gia_label'].cat.codes
data['dịch_vụ'] = data['Dich_vu_label'].cat.codes
data['an_toàn'] = data['An_toan_label'].cat.codes
data['chất_lượng'] = data['Chat_luong_label'].cat.codes
data['ship'] = data['Ship_label'].cat.codes
data['chính_hãng'] = data['Chinh_hang_label'].cat.codes


# Split into train and test - stratify over 
data, data_test = train_test_split(data, test_size = 0.2, stratify = data[['giá']])


#######################################
### --------- Setup BERT ---------- ###

# Name of the BERT model to use
model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 100

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

# Load the Transformers BERT model
transformer_model = TFBertModel.from_pretrained(model_name, config = config)


#######################################
### ------- Build the model ------- ###

# TF Keras documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Model

# Load the MainLayer
bert = transformer_model.layers[0]

# Build your model input
input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
# attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
# inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
inputs = {'input_ids': input_ids}

# Load the Transformers BERT model as a layer in a Keras model
bert_model = bert(inputs)[1]
dropout = Dropout(config.hidden_dropout_prob, name='pooled_output')
pooled_output = dropout(bert_model, training=False)

# Then build your model output
gia = Dense(units=len(data.Gia_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='gia')(pooled_output)
dich_vu = Dense(units=len(data.Dich_vu_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='dich_vu')(pooled_output)
an_toan = Dense(units=len(data.An_toan_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='an_toan')(pooled_output)
chat_luong = Dense(units=len(data.Chat_luong_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='chat_luong')(pooled_output)
ship = Dense(units=len(data.Ship_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='ship')(pooled_output)
chinh_hang = Dense(units=len(data.Chinh_hang_label.value_counts()), kernel_initializer=TruncatedNormal(stddev=config.initializer_range), name='chinh_hang')(pooled_output)
outputs = {'gia': gia, 'dich_vu': dich_vu, 'an_toan': an_toan, 'chat_luong': chat_luong, 'ship': ship, 'chinh_hang': chinh_hang}

# And combine it all in a model object
model = Model(inputs=inputs, outputs=outputs, name='BERT_MultiLabel_MultiClass')

# Take a look at the model
model.summary()


#######################################
### ------- Train the model ------- ###

# Set an optimizer
optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss = {'gia': CategoricalCrossentropy(from_logits = True), 'dich_vu': CategoricalCrossentropy(from_logits = True), 'an_toan': CategoricalCrossentropy(from_logits = True), 'chat_luong': CategoricalCrossentropy(from_logits = True), 'ship': CategoricalCrossentropy(from_logits = True), 'chinh_hang': CategoricalCrossentropy(from_logits = True)}
metric = {'gia': CategoricalAccuracy('accuracy'), 'dich_vu': CategoricalAccuracy('accuracy'), 'an_toan': CategoricalAccuracy('accuracy'), 'chat_luong': CategoricalAccuracy('accuracy'), 'ship': CategoricalAccuracy('accuracy'), 'chinh_hang': CategoricalAccuracy('accuracy')}

# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

# Ready output data for the model
y_gia = to_categorical(data['giá'])
y_dich_vu = to_categorical(data['dịch_vụ'])
y_an_toan = to_categorical(data['an_toàn'])
y_chat_luong = to_categorical(data['chất_lượng'])
y_ship = to_categorical(data['ship'])
y_chinh_hang = to_categorical(data['chính_hãng'])


# Tokenize the input (takes some time)
x = tokenizer(
    text=data['cmt'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

# Fit the model
history = model.fit(
    # x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
    x={'input_ids': x['input_ids']},
    y={'gia': y_gia, 'dich_vu': y_dich_vu, 'an_toan': y_an_toan, 'chat_luong': y_chat_luong, 'ship': y_ship, 'chinh_hang': y_chinh_hang},
    validation_split=0.2,
    batch_size=128,
    epochs=10)


#######################################
### ----- Evaluate the model ------ ###

# Ready test data
test_y_gia = to_categorical(data_test['giá'])
test_y_dich_vu = to_categorical(data_test['dịch_vụ'])
test_y_an_toan = to_categorical(data_test['an_toàn'])
test_y_chat_luong = to_categorical(data_test['chất_lượng'])
test_y_ship = to_categorical(data_test['ship'])
test_y_chinh_hang = to_categorical(data_test['chính_hãng'])

test_x = tokenizer(
    text=data_test['cmt'].to_list(),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = False,
    verbose = True)

# Run evaluation
model_eval = model.evaluate(
    x={'input_ids': test_x['input_ids']},
    y={'gia': test_y_gia, 'dich_vu': test_y_dich_vu, 'an_toan': test_y_an_toan, 'chat_luong': test_y_chat_luong, 'ship': test_y_ship, 'chinh_hang': test_y_chinh_hang}
)