import torch
import whisper

# Load the model and move it to GPU
model = whisper.load_model("base")
if torch.cuda.is_available():
    model = model.to("cuda")
else:
    print("CUDA is not available. Running on CPU.")

# Load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("data/intro.mp3")
audio = whisper.pad_or_trim(audio)

# Make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# Decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# Print the recognized text
print(result.text)
