# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from tacotron2.text import text_to_sequence
import models
import torch
import argparse
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

import sys
import os

import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from apex import amp

from waveglow.denoiser import Denoiser

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--tacotron2', type=str, default="/mnt/data/pretrained/tacotron2/tacotron2_1032590_6000_amp", help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--waveglow', type=str, default="/mnt/data/pretrained/tacotron2/waveglow_1076430_14000_amp",
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.666, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--amp-run', action='store_true', default=True,
                        help='inference with AMP')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true', default=True,
                        help='Include warmup')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--cpu-run', action='store_true', default=False,
                        help='Run inference on CPU')

    return parser

def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_setup_model(model_name, parser, checkpoint, amp_run, cpu_run, forward_is_infer=False):
  model_parser = models.parse_model_args(model_name, parser, add_help=False)
  model_args, _ = model_parser.parse_known_args()
  model_config = models.get_model_config(model_name, model_args)
  model = models.get_model(model_name, model_config, cpu_run, forward_is_infer=forward_is_infer)

  if checkpoint is not None:
    if cpu_run:
      state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']
    else:
      state_dict = torch.load(checkpoint)['state_dict']

    if checkpoint_from_distributed(state_dict):
      state_dict = unwrap_distributed(state_dict)

    model.load_state_dict(state_dict)

  if model_name == "WaveGlow":
    model = model.remove_weightnorm(model)

  model.eval()

  if amp_run:
    model.half()

  return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts, cpu_run=False):

    d = []
    for i, text in enumerate(texts):
        d.append(torch.IntTensor(
            #TODO: eng or kor 
            text_to_sequence(text, ['english_cleaners'])[:]))
            #text_to_sequence(text, ['transliteration_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if torch.cuda.is_available() and not cpu_run:
        text_padded = torch.autograd.Variable(text_padded).cuda().long()
        input_lengths = torch.autograd.Variable(input_lengths).cuda().long()
    else:
        text_padded = torch.autograd.Variable(text_padded).long()
        input_lengths = torch.autograd.Variable(input_lengths).long()

    return text_padded, input_lengths


class MeasureTime():
  def __init__(self, measurements, key, cpu_run):
    self.measurements = measurements
    self.key = key
    self.cpu_run = cpu_run

  def __enter__(self):
    if self.cpu_run == False:
      torch.cuda.synchronize()
    self.t0 = time.perf_counter()

  def __exit__(self, exc_type, exc_value, exc_traceback):
    if self.cpu_run == False:
      torch.cuda.synchronize()
    self.measurements[self.key] = time.perf_counter() - self.t0

# API
# TODO: refactor with main()
class TTS:
  def __init__(self, parser=None):
    if parser is None:
      parser = argparse.ArgumentParser(
          description='PyTorch Tacotron 2 Inference')

      parser = parse_args(parser)

      args, _ = parser.parse_known_args()
      args.tacotron2 = os.environ.get('TACOTRON_PATH', args.tacotron2)

    self.args = args
    self.parser = parser

  def load_model(self):
    print("loading model...")
    args = self.args
    parser = self.parser

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     args.amp_run, args.cpu_run, forward_is_infer=True)
    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    args.amp_run, args.cpu_run, forward_is_infer=True)

    if args.cpu_run:
      denoiser = Denoiser(waveglow, args.cpu_run)
    else:
      denoiser = Denoiser(waveglow, args.cpu_run).cuda()

    jitted_tacotron2 = torch.jit.script(tacotron2)

    print("warming up...")
    if args.include_warmup:
      if args.cpu_run:
        sequence = torch.randint(low=0, high=148, size=(1,50),
                             dtype=torch.long)
        input_lengths = torch.IntTensor([sequence.size(1)]).long()
      else:
        sequence = torch.randint(low=0, high=148, size=(1,50),
                             dtype=torch.long).cuda()
        input_lengths = torch.IntTensor([sequence.size(1)]).cuda().long()

      for i in range(3):
        with torch.no_grad():
            mel, mel_lengths, _ = jitted_tacotron2(sequence, input_lengths)
            _ = waveglow(mel)

    self.jitted_tacotron2  = jitted_tacotron2
    self.waveglow = waveglow
    self.denoiser = denoiser
    print("done...")

  def forward(self, texts):
    args = self.args
    jitted_tacotron2 = self.jitted_tacotron2
    waveglow = self.waveglow
    denoiser = self.denoiser

    measurements = {}

    self.log("texts=", texts)
    sequences_padded, input_lengths = prepare_input_sequence(texts, args.cpu_run)
    self.log(sequences_padded)

    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time", args.cpu_run):
      mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths)

    with torch.no_grad(), MeasureTime(measurements, "waveglow_time", args.cpu_run):
      audios = waveglow(mel, sigma=args.sigma_infer)
      audios = audios.float()
      audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

    self.log("Stopping after", mel.size(2), "decoder steps")

    tacotron2_infer_perf = mel.size(0) * mel.size(2) / measurements['tacotron2_time']
    waveglow_infer_perf = audios.size(0) * audios.size(1) / measurements['waveglow_time']

    self.log("tacotron2_items_per_sec:", tacotron2_infer_perf)
    self.log("tacotron2_latency:", measurements['tacotron2_time'])
    self.log("waveglow_items_per_sec:", waveglow_infer_perf)
    self.log("waveglow_latency:", measurements['waveglow_time'])
    self.log("latency:", (measurements['tacotron2_time'] + measurements['waveglow_time']))

    output_audios = []
    for i, audio in enumerate(audios):
      audio = audio[:mel_lengths[i]*args.stft_hop_length]
      audio = audio/torch.max(torch.abs(audio))
      output_audios.append(audio)

    return output_audios, alignments

  def log(self, *args):
    print(*args)

def main(args):

  tts = TTS(args)
  tts.load_model()

  try:
    f = open(args.input, 'r')
    texts = f.readlines()
  except:
    print("Could not read file")
    sys.exit(1)

  audios, alignments = tts.forward(texts)

  for i, audio in enumerate(audios):
    audio_path = args.output+"audio_"+str(i)+"_"+args.suffix+".wav"
    write(audio_path, args.sampling_rate, audio.cpu().numpy())

    #plt.imshow(alignments[i].float().data.cpu().numpy().T, aspect="auto", origin="lower")
    #figure_path = args.output+"alignment_"+str(i)+"_"+args.suffix+".png"
    #plt.savefig(figure_path)

    DLLogger.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--suffix', type=str, default="", help="output filename suffix")

    main(parser)
