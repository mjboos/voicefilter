import os
import json
import glob
import torch
import librosa
import numpy as np
import argparse

from .utils.audio import Audio
from .utils.hparams import HParam
from .model.model import VoiceFilterTrainable



def main(args, hp):
    with torch.no_grad():
        model = VoiceFilterTrainable(hp).cuda()
        chkpt_model = torch.load(args.checkpoint_path)['model']
        model.load_state_dict(chkpt_model)
        model.eval()

        audio = Audio(hp)

        # getting a mapping from name to dict
        with open(args.subj_dict, 'r') as fl:
            subj_dict = json.load(fl)

        with open(args.reference_speaker, 'r') as f:
            speaker_code = f.readline().strip()
        dvec_idx = torch.from_numpy(np.array(subj_dict[speaker_code]))
        dvec_idx = dvec_idx.unsqueeze(0).cuda()

        mixed_wav, _ = librosa.load(args.mixed_file, sr=16000)
        mag, phase = audio.wav2spec(mixed_wav)
        mag = torch.from_numpy(mag).float().cuda()

        mag = mag.unsqueeze(0)
        mask = model(mag, dvec_idx)
        est_mag = mag * mask

        est_mag = est_mag[0].cpu().detach().numpy()
        est_wav = audio.spec2wav(est_mag, phase)

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, 'result.wav')
        librosa.output.write_wav(out_path, est_wav, sr=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-s', '--subj_dict', type=str, required=True,
                        help="Path of the dictionary that specifies the subject name -> index embedding as a dict in a JSON file.")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")
    parser.add_argument('-m', '--mixed_file', type=str, required=True,
                        help='path of mixed wav file')
    parser.add_argument('-r', '--reference_speaker', type=str, required=True,
                        help='path of speaker code file')
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help='directory of output')

    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)
