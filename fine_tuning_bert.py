#%%
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks
import json

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2")
print("preprocessor loaded", preprocessor)
encoder_inputs = preprocessor(text_input)

glue, info = tfds.load('glue/mrpc', with_info=True,
                       # It's small, load the whole dataset
                       batch_size=-1)
list(glue.keys())
info.features
info.features['label'].names
glue_train = glue['train']

for key, value in glue_train.items():
  print(f"{key:9s}: {value[0].numpy()}")

tokenizer = bert.tokenization.FullTokenizer(
    vocab_file="assets/vocab.txt",
     do_lower_case=True)

print("Vocab size:", len(tokenizer.vocab))


def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(glue_dict, tokenizer):
    num_examples = len(glue_dict["sentence1"])

    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence1"])])
    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence2"])])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs


glue_train = bert_encode(glue['train'], tokenizer)
glue_train_labels = glue['train']['label']

glue_validation = bert_encode(glue['validation'], tokenizer)
glue_validation_labels = glue['validation']['label']

glue_test = bert_encode(glue['test'], tokenizer)
glue_test_labels = glue['test']['label']

config_dict = json.loads(tf.io.gfile.GFile("assets/bert_config.json").read())

bert_config = bert.configs.BertConfig.from_dict(config_dict)



bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)
'''
inputs = tf.keras.layers.Input(shape=(2,), dtype=tf.int32, name='input_classifier')
output = tf.keras.layers.Dense(1, activation="sigmoid", name='classifier')(inputs)
bert_classifier = tf.keras.Model( inputs=bert_classifier, outputs=output)
'''

glue_batch = {key: val[:10] for key, val in glue_train.items()}

print(bert_classifier(
    glue_batch, training=True
).numpy())

epochs = 3
batch_size = 32
eval_batch_size = 32

train_data_size = len(glue_train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)
export_dir='./myburt-md'
'''bert_classifier = tf.saved_model.load(export_dir)'''
tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=64)
bert_classifier.fit(
      glue_train, glue_train_labels,
      validation_data=(glue_validation, glue_validation_labels),
      batch_size=32,
      epochs=epochs)
      
tf.saved_model.save(bert_classifier, export_dir)

my_examples_ = bert_encode(
    glue_dict = {
        'sentence1':[
            'The rain in Spain falls mainly on the plain.'],
        'sentence2':[
            'It never rains on the flat lands of Spain.']
    },
    tokenizer=tokenizer)

my_examples = [my_examples_['input_word_ids'],
                            my_examples_['input_mask'],
                            my_examples_['input_type_ids']]
result = bert_classifier(my_examples, training=False)

print(result.numpy())
result = tf.argmax(result).numpy()
print(result)

my_examples_ = bert_encode(
    glue_dict = {
        'sentence1':[
            'When someone is crazy, he''s not responsible for his acts',
            'When someone is crazy, he''s not responsible for his acts'],
        'sentence2':[
            'The craziness is the key to be more responsible',
            'The craziness turn anybody to non responsible'
            ]
    },
    tokenizer=tokenizer)
my_examples = [my_examples_['input_word_ids'],
                            my_examples_['input_mask'],
                            my_examples_['input_type_ids']]

result = bert_classifier(my_examples, training=False)
print(result.numpy())
result = tf.argmax(result).numpy()
print(result)

my_examples_ = bert_encode(
    glue_dict = {
        'sentence1':[
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.'],
        'sentence2':[
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.']
    },
    tokenizer=tokenizer)
my_examples = [my_examples_['input_word_ids'],
                            my_examples_['input_mask'],
                            my_examples_['input_type_ids']]

result = bert_classifier(my_examples, training=False)
print(result.numpy())
result = tf.argmax(result).numpy()
print(result)