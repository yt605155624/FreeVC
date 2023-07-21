import argparse
import logging
import os

import librosa
import torch
from mel_processing import mel_spectrogram_torch
from models import SynthesizerTrn
from scipy.io.wavfile import write
from speaker_encoder.voice_encoder import SpeakerEncoder
from timer import timer

import utils
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('timer').setLevel(logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpfile",
        type=str,
        default="configs/freevc.json",
        help="path to json config file")
    parser.add_argument(
        "--ptfile",
        type=str,
        default="checkpoints/freevc.pth",
        help="path to pth file")
    parser.add_argument(
        "--txtpath", type=str, default="convert.txt", help="path to txt file")
    parser.add_argument(
        "--outdir",
        type=str,
        default="output/freevc",
        help="path to output dir")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    print("args.ptfile:", args.ptfile)
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)

    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder(
            'speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    N = 0
    T = 0
    with torch.no_grad():
        for line in zip(titles, srcs, tgts):
            title, src, tgt = line
            # tgt
            with timer() as t:
                wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
                wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

                # 会将一个长音频切成多个片后求均值
                if hps.model.use_spk:
                    g_tgt = smodel.embed_utterance(wav_tgt)
                    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
                else:
                    wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
                    mel_tgt = mel_spectrogram_torch(
                        wav_tgt, hps.data.filter_length,
                        hps.data.n_mel_channels, hps.data.sampling_rate,
                        hps.data.hop_length, hps.data.win_length,
                        hps.data.mel_fmin, hps.data.mel_fmax)
                # src
                wav_np, _ = librosa.load(src, sr=hps.data.sampling_rate)
                wav_src = torch.from_numpy(wav_np).unsqueeze(0).cuda()
                c = utils.get_content(cmodel, wav_src)
                # print("title:",title,"c.shape:",c.shape)
                if hps.model.use_spk:
                    audio = net_g.infer(c, g=g_tgt)
                else:
                    audio = net_g.infer(c, mel=mel_tgt)
                audio = audio[0][0].data.cpu().float().numpy()
            N += wav_np.size
            T += t.elapse
            speed = wav_np.size / t.elapse
            rtf = hps.data.sampling_rate / speed
            print(
                f"{title},  wave: {wav_np.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
            )
            write(
                os.path.join(args.outdir, f"{title}.wav"),
                hps.data.sampling_rate, audio)
            print(f"{title} done!")
    print(f"convert speed: {N / T}Hz, RTF: {hps.data.sampling_rate / (N / T) }")
