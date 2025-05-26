import requests
import os
import json
from pydub import AudioSegment
import parselmouth
import pronouncing
import assemblyai as aai

# Set AssemblyAI API key
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Audio file (local or URL)
audio_file = "https://assembly.ai/wildfires.mp3"

# Configure transcription
config = aai.TranscriptionConfig(
    speech_model=aai.SpeechModel.best,
    word_boost=["uh", "um", "like", "you know"]  # Boost filler word detection
)

# Transcribe audio
transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe(audio_file)

# Check for transcription errors
if transcript.status == "error":
    raise RuntimeError(f"Transcription failed: {transcript.error}")

# Extract transcription text and word-level details
transcription = transcript.text
words = transcript.words  # AssemblyAI provides word-level timestamps

# Step 1: Pronunciation accuracy using confidence scores
# Calculate average confidence score as a proxy for pronunciation accuracy
confidence_scores = [word.confidence for word in words]
average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
pronunciation_accuracy = average_confidence * 100  # Scale to 0-100

# Step 2: Prosody and speech rate with parselmouth

snd = parselmouth.Sound("./wildfires.mp3")  # Load the audio file
pitch = snd.to_pitch()
intonation = pitch.selected_array["frequency"].mean()  # Mean pitch for intonation
speech_rate = len(transcription.split()) / snd.duration * 60  # Words per minute
    

# Step 3: Filler word detection with timestamps
filler_words = ["uh", "um", "like", "you know"]
filler_details = []
for word_info in words:
    word = word_info.text.lower()
    if word in filler_words:
        filler_details.append({
            "word": word,
            "start_time": word_info.start / 1000,  # Convert ms to seconds
            "end_time": word_info.end / 1000  # Convert ms to seconds
        })

filler_count = len(filler_details)

# Print results
print(f"Transcription: {transcription}")
print(f"Pronunciation Accuracy (based on confidence): {pronunciation_accuracy:.2f}%")
print(f"Intonation (mean pitch): {intonation:.2f} Hz")
print(f"Speech Rate: {speech_rate:.2f} WPM")
print(f"Filler Words Count: {filler_count}")
if filler_count > 0:
    print("Filler Words Details:")
    for filler in filler_details:
        print(f"  - Word: '{filler['word']}', Start: {filler['start_time']:.2f}s, End: {filler['end_time']:.2f}s")