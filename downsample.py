import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm


def process(wav_path):
    # speaker 'p280', 'p315' are excluded (no mic2)
    wav_path_list = str(wav_path).split('/')
    speaker = wav_path_list[-2]
    wav_name=wav_path_list[-1]
    if '_mic2.flac' in str(wav_path):
        spk_dir1 = args.out_dir1 / speaker
        spk_dir1.mkdir(parents=True, exist_ok=True)
        spk_dir2 = args.out_dir2 / speaker
        spk_dir2.mkdir(parents=True, exist_ok=True)
        # 这里其实有问题，因为 load 的时候默认按照 22.05k, 如果模型采样率 > 22.05k 此处会有压缩
        # 应该 load 的时候就按照配置来
        wav, sr = librosa.load(wav_path)
        # 此处有裁剪
        wav, _ = librosa.effects.trim(wav, top_db=20)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav1 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr1)
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
        save_name = wav_name.replace("_mic2.flac", ".wav")
        save_path1 = spk_dir1 / save_name
        save_path2 = spk_dir2 / save_name

        sf.write(str(save_path1), wav1, samplerate=args.sr1, subtype='PCM_16')
        sf.write(str(save_path2), wav2, samplerate=args.sr2, subtype='PCM_16')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr1", type=int, default=16000, help="sampling rate")
    parser.add_argument("--sr2", type=int, default=22050, help="sampling rate")
    parser.add_argument("--in_dir", type=str, help="path to source dir")
    parser.add_argument("--out_dir1", type=str, default="./dataset/vctk-16k", help="path to target dir")
    parser.add_argument("--out_dir2", type=str, default="./dataset/vctk-22k", help="path to target dir")
    parser.add_argument(
        "--num-cpu", type=int, default=20, help="number of process.")
    args = parser.parse_args()

    pool = Pool(processes=cpu_count()-2)
    args.in_dir = Path(args.in_dir).expanduser()

    args.out_dir1 = Path(args.out_dir1).expanduser()
    args.out_dir1.mkdir(parents=True, exist_ok=True)
    
    args.out_dir2 = Path(args.out_dir2).expanduser()
    args.out_dir2.mkdir(parents=True, exist_ok=True)

    wav_paths= []

    for speaker in os.listdir(args.in_dir):
        if speaker not in {'log.txt', 'p315', 'p280'}:
            spk_dir = args.in_dir / speaker
            if os.path.isdir(spk_dir):
                for wav_name in os.listdir(spk_dir):
                    wav_paths.append(spk_dir / wav_name)


    with ThreadPoolExecutor(args.num_cpu) as pool:
        with tqdm(total=len(wav_paths), desc="resampling") as pbar:
            futures = []
            for i, wav_path in enumerate(wav_paths):
                future = pool.submit(process, wav_path)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)
            results = []
            for ft in futures:
                results.append(ft.result())
