import pandas as pd
import torch
import os
from nemo.collections.asr.models import EncDecCTCModel

# If you encountered any problem installing NeMo using the script from requirements.txt,
# I suggest you to use their docker image, or you can use google colab environment as it worked with us there.

# #################################################################
# To Load the model
# #################################################################

model_path = "best_model_checkpoint/quartznet_15x5_final.nemo"

quartznet = EncDecCTCModel.restore_from(model_path)

if torch.cuda.is_available():
    print('Cuda')
    quartznet.to('cuda')

# #################################################################
# Transcribing a full of audio folder into a CSV file
# #################################################################

def transcribe_audio_files(folder_path, output_csv):
    # List to store file paths
    file_paths = []

    # Loop through all files in the specified directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):  # Check if the file is a WAV file
            file_path = os.path.join(folder_path, filename)
            file_paths.append(file_path)

    # Transcribe all audio files at once
    with torch.no_grad():
        transcriptions = quartznet.transcribe(file_paths)

    # List to store transcription results
    results = []

    # Loop through transcriptions and file paths
    for filename, transcription in zip(os.listdir(folder_path), transcriptions):
        if filename.endswith(".wav"):  # Check if the file is a WAV file
            results.append({
                'audio': filename,
                'transcript': transcription
            })

    # Convert results to a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
# Example usage
folder_path = "path/to/audio_folder"
output_csv = "transcriptions.csv"
transcribe_audio_files(folder_path, output_csv)

# #################################################################
# Transcribing a single .wav sample
# #################################################################

path_to_sample = "path/to/sample.wave"

transcription = quartznet.transcribe([path_to_sample])