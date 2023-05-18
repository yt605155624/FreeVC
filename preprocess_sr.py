import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import librosa
import torch
from mel_processing import mel_spectrogram_torch
from scipy.io import wavfile
from tqdm import tqdm
from wavlm import WavLM
from wavlm import WavLMConfig

import utils
logging.getLogger('numba').setLevel(logging.WARNING)


def process(filename):
    basename = os.path.basename(filename)
    speaker = filename.split("/")[-2]
    wav_dir = os.path.join(args.wav_dir, speaker)
    ssl_dir = os.path.join(args.ssl_dir, speaker)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(ssl_dir, exist_ok=True)
    wav, _ = librosa.load(filename, sr=hps.sampling_rate)
    wav = torch.from_numpy(wav).unsqueeze(0).cuda()
    mel = mel_spectrogram_torch(wav, hps.n_fft, hps.num_mels, hps.sampling_rate,
                                hps.hop_size, hps.win_size, hps.fmin, hps.fmax)

    for i in range(args.min, args.max + 1):
        mel_rs = utils.transform(mel, i)
        wav_rs = vocoder(mel_rs)[0][0].detach().cpu().numpy()
        _wav_rs = librosa.resample(
            wav_rs, orig_sr=hps.sampling_rate, target_sr=args.sr)
        wav_rs = torch.from_numpy(_wav_rs).cuda().unsqueeze(0)
        c = utils.get_content(cmodel, wav_rs)
        ssl_path = os.path.join(ssl_dir, basename.replace(".wav", f"_{i}.pt"))
        torch.save(c.cpu(), ssl_path)
        wav_path = os.path.join(wav_dir, basename.replace(".wav", f"_{i}.wav"))
        wavfile.write(wav_path, args.sr, _wav_rs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument("--min", type=int, default=68, help="min")
    parser.add_argument("--max", type=int, default=92, help="max")
    parser.add_argument(
        "--config",
        type=str,
        default="hifigan/config.json",
        help="path to config file")
    parser.add_argument(
        "--in_dir",
        type=str,
        default="dataset/vctk-22k",
        help="path to input dir")
    parser.add_argument(
        "--wav_dir",
        type=str,
        default="dataset/sr/wav",
        help="path to output wav dir")
    parser.add_argument(
        "--ssl_dir",
        type=str,
        default="dataset/sr/wavlm",
        help="path to output ssl dir")
    parser.add_argument('--num_workers', type=int, default=20)
    args = parser.parse_args()

    print("Loading WavLM for content...")
    checkpoint = torch.load('wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    cmodel = WavLM(cfg).cuda()
    cmodel.load_state_dict(checkpoint['model'])
    cmodel.eval()
    cmodel = cmodel.cuda()
    print("Loaded WavLM.")

    print("Loading vocoder...")
    vocoder = utils.get_vocoder(0)
    vocoder.eval()

    print("Loaded vocoder.")

    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hps = utils.HParams(**config)

    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)  #[:10]

    with ThreadPoolExecutor(args.num_workers) as pool:
        with tqdm(total=len(filenames), desc="preprocess_sr") as pbar:
            futures = []
            for i, filename in enumerate(filenames):
                future = pool.submit(process, filename)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)
            results = []
            for ft in futures:
                results.append(ft.result())
