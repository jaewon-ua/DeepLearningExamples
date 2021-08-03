import os
import numpy as np
import sys
import time
import argparse
import torch
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence
sys.path.append('..')
from waveglow.mel2samp import MAX_WAV_VALUE
from denoiser import Denoiser
import json

class T2S:
  def __init__(self, lang):
    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     args.amp_run, args.cpu_run, forward_is_infer=True)
    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    args.amp_run, args.cpu_run, forward_is_infer=True)

    if args.cpu_run:
        denoiser = Denoiser(waveglow, args.cpu_run)
    else:
        denoiser = Denoiser(waveglow, args.cpu_run).cuda()

    jitted_tacotron2 = torch.jit.script(tacotron2)

    self.language = lang

  def tts(self, texts):
    if not filename:
      filename = str(time.time())

    sequences_padded, input_lengths = prepare_input_sequence(texts, args.cpu_run)

    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time", args.cpu_run):
        mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths)

    with torch.no_grad(), MeasureTime(measurements, "waveglow_time", args.cpu_run):
        audios = waveglow(mel, sigma=args.sigma_infer)
        audios = audios.float()
        audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

    print("Stopping after",mel.size(2),"decoder steps")

    tacotron2_infer_perf = mel.size(0)*mel.size(2)/measurements['tacotron2_time']
    waveglow_infer_perf = audios.size(0)*audios.size(1)/measurements['waveglow_time']

    output_audios = []
    for i, audio in enumerate(audios):
      audio = audio[:mel_lengths[i] * args.stft_hop_length]
      audio = audio/torch.max(torch.abs(audio))
      otuput_audios.append(audio)

    return output_audios

  def update_model(self, lang):
    if lang == 'en':
      self.checkpoint_path = self.config.get('model').get('en')
      self.cleaner = 'english_cleaners'
      self.language = lang
    else:
      self.checkpoint_path = self.config.get('model').get('kr')
      self.cleaner =  'transliteration_cleaners'
      self.language = lang

    self.model = self.load_model()
    self.model.load_state_dict(torch.load(self.checkpoint_path)['state_dict'])
    _ = self.model.cuda().eval().half()
