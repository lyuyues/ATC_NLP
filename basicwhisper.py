import pandas as pd
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load the training dataset from multiple local Parquet files
train_files = [
    'data/train-00000-of-00004-c1d7fb31dcbf644a.parquet', 
    'data/train-00001-of-00004-f165730df6bf7253.parquet', 
    'data/train-00002-of-00004-67e682f17e32b703.parquet', 
    'data/train-00003-of-00004-b0b05d4b243c95c6.parquet'
]
train_dfs = [pd.read_parquet(file) for file in train_files]
train_df = pd.concat(train_dfs, ignore_index=True)

# Load the test dataset from a local Parquet file
test_df = pd.read_parquet('data/test-00000-of-00001-01544bdf54b4ccf3.parquet')

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Define a function to transcribe audio
def transcribe_audio(audio_array, sampling_rate):
    # Create a tensor from the audio array
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    # Resample if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio_tensor = resampler(audio_tensor)
    # Process the audio to the expected input format
    input_features = processor(audio_tensor.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
    # Generate transcription
    generated_ids = model.generate(input_features)
    # Decode the generated ids to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription[0]

# Transcribe and add transcriptions to the dataset
def transcribe_dataset(dataset):
    transcriptions = []
    for i, row in dataset.iterrows():
        audio_data = row['audio']
        sampling_rate = row['sampling_rate']
        transcription = transcribe_audio(audio_data, sampling_rate)
        transcriptions.append(transcription)
    dataset['transcription'] = transcriptions
    return dataset

# Transcribe the audio files in the train and test datasets
train_df = transcribe_dataset(train_df)
test_df = transcribe_dataset(test_df)

# Save the transcriptions to new Parquet files
train_df.to_parquet('transcribed_train.parquet')
test_df.to_parquet('transcribed_test.parquet')
