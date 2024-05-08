import torch
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
from datasets import load_dataset

import os
import sys
current = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current))

from visual_methods.heatmap import Heatmap
from visual_methods.heatmaps import Heatmaps
from visualization import Visualization

def main():
    # loading our model weights
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    # loading the corresponding preprocessor config
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)

    # load demo audio and set processor
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    resample_rate = processor.sampling_rate
    # make sure the sample_rate aligned
    if resample_rate != sampling_rate:
        print(f'setting rate from {sampling_rate} to {resample_rate}')
        resampler = T.Resample(sampling_rate, resample_rate)
    else:
        resampler = None

    # audio file is decoded on the fly
    if resampler is None:
        input_audio = dataset[0]["audio"]["array"]
    else:
        input_audio = resampler(torch.from_numpy(dataset[0]["audio"]["array"]))
    
    inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model(**inputs, output_hidden_states=True)

    info = {
        'name' : "librispeech_asr_demo",
        'data' : inputs
    }

    visual = Heatmaps()
    visualization = Visualization(model, [info], visual)
    visualization.visualize()

if __name__ == '__main__':
    main()