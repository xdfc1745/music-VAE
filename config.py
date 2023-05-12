# Copyright 2023 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configurations for MusicVAE models."""
import collections

from magenta.common import merge_hparams
from magenta.contrib import training as contrib_training
from magenta.models.music_vae import data
from magenta.models.music_vae import data_hierarchical
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae.base_model import MusicVAE
import note_seq

HParams = contrib_training.HParams


class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
    'train_examples_path', 'eval_examples_path', 'tfds_name'])):

    def values(self):
        return self._asdict()

Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
    config_dict = config.values()
    config_dict.update(update_dict)
    return Config(**config_dict)


CONFIG_MAP = {}

# Drums
CONFIG_MAP['cat-drums_2bar_small'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(), lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=2,
        steps_per_quarter=4,
        roll_input=True),
    train_examples_path=None,
    eval_examples_path=None,
)

# CONFIG_MAP['cat-drums_2bar_big'] = Config(
#     model=MusicVAE(lstm_models.BidirectionalLstmEncoder(), lstm_models.CategoricalLstmDecoder()),
#     hparams=merge_hparams(
#         lstm_models.get_default_hparams(),
#         HParams(
#             batch_size=512,
#             max_seq_len=32,  # 2 bars w/ 16 steps per bar
#             z_size=512,
#             enc_rnn_size=[2048],
#             dec_rnn_size=[2048, 2048, 2048],
#             free_bits=48,
#             max_beta=0.2,
#             sampling_schedule='inverse_sigmoid',
#             sampling_rate=1000,
#         )),
#     note_sequence_augmenter=None,
#     data_converter=data.DrumsConverter(
#         max_bars=100,  # Truncate long drum sequences before slicing.
#         slice_bars=2,
#         steps_per_quarter=4,
#         roll_input=True),
#     train_examples_path=None,
#     eval_examples_path=None,
# )

CONFIG_MAP['cat-drums_2bar_big'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(), lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 4,  # 4 bars w/ 16 steps per bar
            z_size=512,
            enc_rnn_size=[2048],
            dec_rnn_size=[2048, 2048, 2048],
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=4,
        steps_per_quarter=4,
        roll_input=True),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['nade-drums_2bar_reduced'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(), lstm_models.MultiLabelRnnNadeDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[1024],
            dec_rnn_size=[512, 512],
            nade_num_hidden=128,
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=2,
        steps_per_quarter=4,
        roll_input=True,
        roll_output=True),
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['nade-drums_2bar_full'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(), lstm_models.MultiLabelRnnNadeDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=32,  # 2 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[1024],
            dec_rnn_size=[512, 512],
            nade_num_hidden=128,
            free_bits=48,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        pitch_classes=data.FULL_DRUM_PITCH_CLASSES,
        slice_bars=2,
        steps_per_quarter=4,
        roll_input=True,
        roll_output=True),
    train_examples_path=None,
    eval_examples_path=None,
)

# GrooVAE configs
CONFIG_MAP['groovae_4bar'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 4,  # 4 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=4, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES),
    tfds_name='groove/4bar-midionly',
)