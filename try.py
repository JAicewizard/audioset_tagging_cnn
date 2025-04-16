# Used in the pytorch modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))

import librosa
import soundfile as sf
import numpy as np
import inference
import json
import argparse as ap
from models import *
import config
import torch
import io
import soundfile
from pytorch_utils import move_data_to_device

BASE = "/home/jaap/Downloads/birdclef-2025/train_audio/"  

# configuration
frame_len = 320*32
hop_len = 320*4
# Fire alarm is not an animal, however pitch can be similar to targets. Little false-positives with, but many false negatives without
animal_things = {"Insect", "Animal", "Squeal", "Domestic animals, pets", "Cat", "Meow", "Horse", "Fire alarm", "Smoke detector, smoke alarm", "Owl", "Whistle", "Bird", "Mouse", "Cricket", "Bird vocalization, bird call, bird song", "Pigeon, dove", "Crowing, cock-a-doodle-doo", "Rodents, rats, mice", "Car alarm", "Crow", "Frog", "Chirp, tweet", "Duck", "Turkey", "Chirp tone", "Goat", "Sheep", "Livestock, farm animals, working animals"}
# Dog is an animal, but very different from what we are looking for.
# Same for Bee etc, they dont make enough sound and are usually just background
# Roaring cats (lions, tigers) because they don't occur that often, and it was manually checked to be non-target label
non_animal_things = {"dog", "Bee, wasp, etc.", "Roaring cats (lions, tigers)", "Speech", "White noise", "Vehicle", "Music", "Snort"}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model init
torch.set_num_threads(16)

labels = config.labels
model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=config.classes_num)
checkpoint = torch.load("Cnn14_mAP=0.431.pth", map_location=device)
model.load_state_dict(checkpoint['model'])
if 'cuda' in str(device):
    model.to(device)
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
else:
    print('Using CPU.')

with torch.no_grad():
    model.eval()

no_animal_set = set()

def process_file(species: str, sound_name: str):
    species_base = BASE+"/"+species+"/"
    sound, sr = librosa.load(species_base + sound_name + ".ogg")
    frames = librosa.util.frame(sound, frame_length=frame_len, hop_length=hop_len, axis=0)
    
    frames = move_data_to_device(frames.copy(), device)
    
    batch_output_dict = {}
    with torch.no_grad():
        batch_output_dict = model(frames, None)
    
    clipwise_outputs = batch_output_dict['clipwise_output'].data.cpu().numpy()
    
    final_stretches = []
    last_animal = 0.0
    for i, clipwise in enumerate(clipwise_outputs):
        t = i*hop_len/sr
        nt = (i*hop_len+frame_len)/sr
        
        print(f"##########{species}/{sound_name}: {t}-{nt}", file=sys.stderr)

        sorted_indexes = np.argsort(clipwise)[::-1]

        for k in range(2):
            print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
                clipwise[sorted_indexes[k]]), file=sys.stderr)
       
        if labels[sorted_indexes[0]] not in animal_things:
            print("no_animal:", labels[sorted_indexes[0]], file=sys.stderr)
            if labels[sorted_indexes[0]] not in non_animal_things:
                no_animal_set.add(labels[sorted_indexes[0]])
            if last_animal < t:
                final_stretches.append((last_animal, t))
            last_animal = nt 

    print(final_stretches, file=sys.stderr)
    result = {
        "onset": [],
        "offset": [],
        "cluster": [],
        "species": species,
        "sr": sr,
        "min_frequency": 0,
        "spec_time_step": 0.0025,
        "min_segment_length": 0.01,
        "tolerance": 0.01,
        "time_per_frame_for_scoring": 0.001,
        "eps": 0.02,
    }
    for low, high in final_stretches:
        result["onset"].append(low)
        result["offset"].append(high)
        result["cluster"].append(species)

    with open(species_base+"/" + sound_name + '.json', 'w') as fp:
        json.dump(result, fp, indent=4, ensure_ascii=False)
        fp.close()

total_size = 0
n_species = len(os.listdir(BASE))
for i, file in enumerate(os.listdir(BASE)):
    print(f"{i}/{n_species}")
    species = os.fsdecode(file)
    n_files = len(os.listdir(BASE+species))
    for j, file in enumerate(os.listdir(BASE+species)):
        filename = os.fsdecode(file)
        if filename.endswith(".ogg"):
            total_size += os.path.getsize(f"{BASE}/{species}/{filename}")
            print(f"\t{j}/{n_files} ({total_size/(1024*1024)}MiB)")
            process_file(species, filename[:-4])
    print(no_animal_set)
