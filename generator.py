import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
import json

from .utils.audio import Audio
from .utils.hparams import HParam

def train_test_split_folder(subj_folder, test_size=0.2, **kwargs):
    '''Does a train test split of all chapter folders in subj_folder and returns folder names as lists'''
    from sklearn.model_selection import train_test_split
    chapter_folders = [x for x in glob.glob(os.path.join(subj_folder, '*')) if os.path.isdir(x)]
    return train_test_split(chapter_folders, test_size=test_size, **kwargs)

def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def mix(hp, args, audio, num, s1_dvec, s1_target, s2, train):
    srate = hp.audio.sample_rate
    dir_ = os.path.join(args.out_dir, 'train' if train else 'test')

    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    assert len(w1.shape) == len(w2.shape) == 1, \
        'wav files must be mono, not stereo'

    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)

    # LibriSpeech dataset have many silent interval, so let's vad-merge them
    # VoiceFilter paper didn't do that. To test SDR in same way, don't vad-merge.
    if args.vad == 1:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1, w2 = w1[:L], w2[:L]

    mixed = w1 + w2

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1/norm, w2/norm, mixed/norm

    # save vad & normalized wav files
    target_wav_path = formatter(dir_, hp.form.target.wav, num)
    mixed_wav_path = formatter(dir_, hp.form.mixed.wav, num)
    librosa.output.write_wav(target_wav_path, w1, srate)
    librosa.output.write_wav(mixed_wav_path, mixed, srate)

    # save magnitude spectrograms
    target_mag, _ = audio.wav2spec(w1)
    mixed_mag, _ = audio.wav2spec(mixed)
    target_mag_path = formatter(dir_, hp.form.target.mag, num)
    mixed_mag_path = formatter(dir_, hp.form.mixed.mag, num)
    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

    # save selected subject name as text file
    dvec_text_path = formatter(dir_, hp.form.dvec, num)
    with open(dvec_text_path, 'w') as f:
        f.write(s1_dvec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-d', '--libri_dir', type=str, required=True,
                        help="Directory of LibriSpeech dataset, containing folders of dev-clean.")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-p', '--process_num', type=int, default=None,
                        help='number of processes to run. default: cpu_count')
    parser.add_argument('--vad', type=int, default=0,
                        help='apply vad to wav file. yes(1) or no(0, default)')
    parser.add_argument('--train_size', type=int, default=100, help='Size of training set to be generated')
    parser.add_argument('--test_size', type=int, default=100, help='Size of test set to be generated')
    parser.add_argument('--test_proportion', type=float, default=0.2, help='Proportion of chapters of each subject to be used for the test set.')
    parser.add_argument('--subjects', nargs='+', default=None, help='Names of subjects to use')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)

    hp = HParam(args.config)

    cpu_num = cpu_count() if args.process_num is None else args.process_num

    subj_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'dev-clean', '*'))
                        if os.path.isdir(x)]
    if args.subjects is not None:
        subjects_to_use = [sub_folder for sub_folder in subj_folders if sub_folder.split('/')[-1] in args.subjects]
    else:
        subjects_to_use = np.random.sample(subj_folders, 2)
    # TODO: test for multiple chapters
    # train/test split based on chapter -> use only one chapter for testing
    train_folders, test_folders = zip(*[train_test_split_folder(sub_folder, test_size=args.test_proportion)
                                        for sub_folder in subjects_to_use])
    #train_folders = ['LibriSpeech/dev-clean/2902/9008', 'LibriSpeech/dev-clean/1673/143397']
    #test_folders = ['LibriSpeech/dev-clean/2902/9006', 'LibriSpeech/dev-clean/1673/143396']

    background_folders = [x for x in subj_folders
                            if x not in subjects_to_use]

    train_spk = [[glob.glob(os.path.join(chapter_spk, '**', hp.form.input), recursive=True)
                  for chapter_spk in spk]
                 for spk in train_folders]
    train_spk = [[x for chpt in spk for x in chpt]
                  for spk in train_spk]
    test_spk = [[glob.glob(os.path.join(chapter_spk, '**', hp.form.input), recursive=True)
                  for chapter_spk in spk]
                 for spk in test_folders]
    test_spk = [[x for chpt in spk for x in chpt]
                  for spk in test_spk]


    background_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True)
                        for spk in background_folders]
    assert not np.any([nm in train_spk for nm in background_spk])
    audio = Audio(hp)

    # save dictionary of unique subject names
    subj_names = {subj: idx for idx, subj in enumerate(
        np.unique([spk[0].split('/')[-3] for spk in train_spk]+[spk[0].split('/')[-3] for spk in test_spk]))}
    with open(os.path.join(args.out_dir, 'subj_dict.json'), 'w+') as fl:
        json.dump(subj_names, fl)

    def train_wrapper(num):
        spk1 = random.sample(train_spk, 1)[0]
        spk2 = random.sample(background_spk, 1)[0]
        s1_target = random.choice(spk1)
        s2 = random.choice(spk2)
        s1_name = s1_target.split('/')[-3]
        mix(hp, args, audio, num, s1_name, s1_target, s2, train=True)

    def test_wrapper(num):
        spk1 = random.sample(test_spk, 1)[0]
        spk2 = random.sample(background_spk, 1)[0]
        s1_target = random.choice(spk1)
        s2 = random.choice(spk2)
        s1_name = s1_target.split('/')[-3]
        mix(hp, args, audio, num, s1_name, s1_target, s2, train=False)

    arr = list(range(args.train_size))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(train_wrapper, arr), total=len(arr)))
#
    arr = list(range(args.test_size))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(test_wrapper, arr), total=len(arr)))
#
