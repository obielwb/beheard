import speech_recognition as sr
from pydub import AudioSegment
import parselmouth
import pronouncing
import Levenshtein

# Transcribe audio
recognizer = sr.Recognizer()
audio_file = "audio_test_beheard.wav"
with sr.AudioFile(audio_file) as source:
    audio = recognizer.record(source)
    transcription = recognizer.recognize_sphinx(audio)

# Pronunciation accuracy
reference_text = "This is an audio test to see if the program works."
accuracy = 1 - Levenshtein.distance(transcription, reference_text) / len(reference_text)

# Prosody analysis with parselmouth
snd = parselmouth.Sound(audio_file)
pitch = snd.to_pitch()
intonation = pitch.selected_array["frequency"].mean()  # Mean pitch for intonation
speech_rate = len(transcription.split()) / (snd.duration)  # Words per second

# Filler word detection
filler_words = ["uh", "um", "like", "you know"]
filler_count = sum(transcription.lower().count(filler) for filler in filler_words)

print(f"Transcription: {transcription}")
print(f"Accuracy: {accuracy}")
print(f"Intonation (mean pitch): {intonation} Hz")
print(f"Speech Rate: {speech_rate} words/second")
print(f"Filler Words: {filler_count}")