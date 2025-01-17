import argparse
import os
from pathlib import Path
from random import shuffle

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_dir", type=str, default="./filelists/", help="path to lists")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="./dataset/vctk-16k",
        help="path to source dir")
    parser.add_argument("--valnum_per_spk", type=int, default=2)
    parser.add_argument("--testnum_per_spk", type=int, default=10)
    args = parser.parse_args()

    train_paths = []
    val_paths = []
    test_paths = []

    source_dir = Path(args.source_dir)
    list_dir = Path(args.list_dir)
    list_dir.mkdir(parents=True, exist_ok=True)

    for speaker in tqdm(os.listdir(source_dir)):
        wavs = os.listdir(os.path.join(source_dir, speaker))
        wav_paths = [str(source_dir / speaker / wav) for wav in wavs]
        shuffle(wav_paths)
        train_paths += wav_paths[args.valnum_per_spk:-args.testnum_per_spk]
        val_paths += wav_paths[:args.valnum_per_spk]
        test_paths += wav_paths[-args.testnum_per_spk:]

    shuffle(train_paths)
    shuffle(val_paths)
    shuffle(test_paths)

    for item in {'train', 'val', 'test'}:
        with open(list_dir / (item + '.txt'), "w") as f:
            for path in eval(item + '_paths'):
                f.write(path + "\n")
