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

"""MusicVAE generation script."""

# TODO(adarob): Add support for models with conditioning.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from magenta.models.music_vae import configs
from magenta.models.music_vae import TrainedModel
import note_seq
import numpy as np
import tensorflow.compat.v1 as tf

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
flags.DEFINE_string(
    'checkpoint_file', None,
    'Path to the checkpoint file. run_dir will take priority over this flag.')
flags.DEFINE_string(
    'output_dir', '/tmp/music_vae/generated',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'config', None,
    'The name of the config to use.')
flags.DEFINE_integer(
    'num_outputs', 5,
    'In `sample` mode, the number of samples to produce.')
flags.DEFINE_integer(
    'max_batch_size', 8,
    'The maximum batch size to use. Decrease if you are seeing an OOM.')
flags.DEFINE_float(
    'temperature', 0.5,
    'The randomness of the decoding process.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')


def _slerp(p0, p1, t):
    """Spherical linear interpolation."""
    omega = np.arccos(
        np.dot(np.squeeze(p0/np.linalg.norm(p0)),
                np.squeeze(p1/np.linalg.norm(p1))))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def run(config_map):
    """Load model params, save config file and start trainer.
    Args:
        config_map: Dictionary mapping configuration name to Config object.

    Raises:
        ValueError: if required flags are missing or invalid.
    """
    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    
    # checking flag value
    if FLAGS.run_dir is None == FLAGS.checkpoint_file is None:
        raise ValueError(
            'Exactly one of `--run_dir` or `--checkpoint_file` must be specified.')
    if FLAGS.output_dir is None:
        raise ValueError('`--output_dir` is required.')
    tf.gfile.MakeDirs(FLAGS.output_dir) # make forder if not have output_dir

    if FLAGS.config not in config_map:
        raise ValueError('Invalid config name: %s' % FLAGS.config)
    config = config_map[FLAGS.config]
    config.data_converter.max_tensors_per_item = None

    logging.info('Loading model...')
    if FLAGS.run_dir:
        checkpoint_dir_or_path = os.path.expanduser(os.path.join(FLAGS.run_dir, 'train'))
    else:
        checkpoint_dir_or_path = os.path.expanduser(FLAGS.checkpoint_file)
    # using checkpoint for upload model
    model = TrainedModel(
        config, batch_size=min(FLAGS.max_batch_size, FLAGS.num_outputs),
        checkpoint_dir_or_path=checkpoint_dir_or_path)

    # decode random point in the latent spacing of the model.
    logging.info('Sampling...')
    results = model.sample(
        n=FLAGS.num_outputs,
        length=config.hparams.max_seq_len,
        temperature=FLAGS.temperature)

    # tfrecord to midi file
    basename = os.path.join(
        FLAGS.output_dir,
        '%s_%s_%s-*-of-%03d.mid' %
        (FLAGS.config, FLAGS.mode, date_and_time, FLAGS.num_outputs))
    logging.info('Outputting %d files as `%s`...', FLAGS.num_outputs, basename)
    for i, ns in enumerate(results):
        note_seq.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

    logging.info('Done.')


def main(unused_argv):
    logging.set_verbosity(FLAGS.log)
    run(configs.CONFIG_MAP)


def console_entry_point():
    tf.disable_v2_behavior()
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()