# -*- coding: utf-8 -*-
"""fine tune small bert

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18FmmghYb598GH7eWhopxq98f5v4WUkAV
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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

#preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2")
#print("preprocessor loaded", preprocessor)
#encoder_inputs = preprocessor(text_input)

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
     vocab_file="gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt",
     do_lower_case=True)

print("Vocab size:", len(tokenizer.vocab))

max_seq_length =128
def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)
 #bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2")
def bert_encode(glue_dict, tokenizer):
    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence1"])])
    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence2"])])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)
    input_word_ids = tf.stack([tf.pad(e, [[0, max_seq_length - len(e.numpy())]]) for e in input_word_ids], axis=0)

    input_mask = tf.ones_like(input_word_ids)

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1)
    input_type_ids = tf.stack([tf.pad(e, [[0, max_seq_length - len(e.numpy())]]) for e in input_type_ids], axis=0)
    inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}
    #for key, value in inputs.items():
    #    print(f'{key:15s} shape: {value.shape}')
    return inputs

glue_train = bert_encode(glue['train'], tokenizer)
glue_train_labels = glue['train']['label']

glue_validation = bert_encode(glue['validation'], tokenizer)
glue_validation_labels = glue['validation']['label']

glue_test = bert_encode(glue['test'], tokenizer)
glue_test_labels = glue['test']['label']

#config_dict = json.loads(tf.io.gfile.GFile("assets/bert_config.json").read())
#bert_config = bert.configs.BertConfig.from_dict(config_dict)

#bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)

def build_classifier_model(num_classes):
    inputs = dict(
      input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
      input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
      input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )

    #encoder = bert.bert_models.get_transformer_encoder(bert_config, max_seq_length, output_range=1)
    #inputs = encoder.inputs
    #_, net = encoder(inputs)

    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1", trainable=True, name='encoder')
    net = encoder(inputs)['pooled_output']
    net = tf.keras.layers.Dropout(rate=0.1)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    return tf.keras.Model(inputs, net, name='prediction')

bert_classifier = build_classifier_model(2)

epochs = 16
batch_size = 32
eval_batch_size = batch_size

train_data_size = len(glue_train_labels)
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

# creates an optimizer with learning rate schedule
optimizer = nlp.optimization.create_optimizer(
    2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)
export_dir='./myburt-md'

#bert_classifier = tf.saved_model.load(export_dir)

tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=64)

bert_classifier.fit(
      glue_train, glue_train_labels,
      validation_data=(glue_validation, glue_validation_labels),
      batch_size=batch_size,
      epochs=epochs)

#tf.saved_model.save(bert_classifier, export_dir)

glue_example = {
        'sentence1': [
            'The rain in Spain falls mainly on the plain.',
            'The rain in Spain falls mainly on the plain.',
            'After i run, i feel good',
            'After i run, i feel good',
            'Look I fine tuned BERT.'],
        'sentence2': [
            'It never rains in spain.',
            'It mostly rains on the flat lands of Spain.',
            'I feel hungry when i run',
            'I feel well when i run',
            'Is it working? This does not match.'],
        'labels': [0, 1, 0, 1, 0]
    }
my_examples = bert_encode(
    glue_dict=glue_example,
    tokenizer=tokenizer)

result = tf.math.softmax(bert_classifier(my_examples, training=False))
result = tf.argmax(input=result, axis=1)
for i in range(len(result)):
    print(result[i], glue_example['labels'][i])

glue_batch = {key: val[:10] for key, val in glue_validation.items()}

result = tf.math.softmax(bert_classifier(
    glue_batch, training=True
)).numpy()
result = tf.argmax(input=result, axis=1)
for i in range(len(result)):
    print(glue['validation']['sentence1'].numpy()[i])
    print(glue['validation']['sentence2'].numpy()[i])
    print(result[i], glue_validation_labels.numpy()[i])