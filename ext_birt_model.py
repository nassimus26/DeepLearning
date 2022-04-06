from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import official.nlp.bert.bert_models
import gin
import tensorflow as tf
import tensorflow_hub as hub
from official.nlp import bert
from official.modeling import tf_utils
from official.nlp.albert import configs as albert_configs
from official.nlp.bert import configs
from official.nlp.modeling import models
from official.nlp.modeling import networks
def classifier_model(bert_config,
                     num_labels,
                     max_seq_length=None,
                     final_layer_initializer=None,
                     hub_module_url=None,
                     hub_module_trainable=True):
  """BERT classifier model in functional API style.

  Construct a Keras model for predicting `num_labels` outputs from an input with
  maximum sequence length `max_seq_length`.

  Args:
    bert_config: BertConfig or AlbertConfig, the config defines the core BERT or
      ALBERT model.
    num_labels: integer, the number of classes.
    max_seq_length: integer, the maximum input sequence length.
    final_layer_initializer: Initializer for final dense layer. Defaulted
      TruncatedNormal initializer.
    hub_module_url: TF-Hub path/url to Bert module.
    hub_module_trainable: True to finetune layers in the hub module.

  Returns:
    Combined prediction model (words, mask, type) -> (one-hot labels)
    BERT sub-model (words, mask, type) -> (bert_outputs)
  """
  if final_layer_initializer is not None:
    initializer = final_layer_initializer
  else:
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)

  if not hub_module_url:
    bert_encoder = bert.bert_models.get_transformer_encoder(
        bert_config, max_seq_length, output_range=1)
    return models.BertClassifier(
        bert_encoder,
        num_classes=num_labels,
        dropout_rate=bert_config.hidden_dropout_prob,
        initializer=initializer), bert_encoder

  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
  bert_model = hub.KerasLayer(hub_module_url, trainable=hub_module_trainable)
  pooled_output, _ = bert_model([input_word_ids, input_mask, input_type_ids])
  output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
      pooled_output)

  output = tf.keras.layers.Dense(
      num_labels/2, kernel_initializer=initializer, name='output')(
          output)
  return tf.keras.Model(
      inputs={
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'input_type_ids': input_type_ids
      },
      outputs=output), bert_model
