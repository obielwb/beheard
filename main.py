import speech_recognition as sr
from pydub import AudioSegment
import librosa
import numpy as np
from g2p_en import G2p
import os

import nltk
nltk.download('averaged_perceptron_tagger_eng')

# Initialize phoneme converter
g2p = G2p()

def load_audio(file_path):
    """Load audio file and convert to WAV if necessary."""
    audio = AudioSegment.from_file(file_path)
    wav_path = "temp.wav"
    audio = audio.set_channels(1).set_frame_rate(16000)  # Standardize for ASR
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(file_path):
    """Transcribe audio using CMU Sphinx."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_sphinx(audio)
        return transcription
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None

def text_to_phonemes(text):
    """Convert text to phonemes using g2p-en."""
    phonemes = g2p(text)
    return phonemes

def compute_phoneme_accuracy(spoken_phonemes, expected_phonemes):
    """Compute phoneme accuracy using Levenshtein distance."""
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    distance = levenshtein_distance(spoken_phonemes, expected_phonemes)
    accuracy = 1 - (distance / max(len(spoken_phonemes), len(expected_phonemes)))
    return max(0, accuracy) * 100  # Return as percentage

def analyze_prosody(file_path):
    """Analyze pitch and tempo for prosody scoring."""
    y, sr = librosa.load(file_path)
    # Extract pitch using librosa
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    # Simple heuristic: penalize if pitch variation is too low (monotone)
    pitch_std = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    prosody_score = min(100, (pitch_std / 50) * 100)  # Normalize to 0-100
    return prosody_score

def compute_clarity_score(audio_file, expected_text=None):
    """Compute pronunciation clarity score."""
    # Step 1: Load and preprocess audio
    wav_file = load_audio(audio_file)
    
    # Step 2: Transcribe audio
    transcription = transcribe_audio(wav_file)
    if transcription is None:
        print("Could not transcribe audio.")
        return 0
    
    print(f"Transcribed text: {transcription}")
    
    # Step 3: Convert to phonemes
    spoken_phonemes = text_to_phonemes(transcription)
    if expected_text:
        expected_phonemes = text_to_phonemes(expected_text)
    else:
        expected_phonemes = spoken_phonemes  # Assume transcription is correct
    
    # Step 4: Compute phoneme accuracy
    phoneme_accuracy = compute_phoneme_accuracy(spoken_phonemes, expected_phonemes)
    
    # Step 5: Analyze prosody
    prosody_score = analyze_prosody(wav_file)
    
    print(f"Phoneme accuracy: {phoneme_accuracy:.2f}%")
    print(f"Prosody score: {prosody_score:.2f}/100")
    
    # Step 6: Combine scores (weighted average)
    clarity_score = 0.7 * phoneme_accuracy + 0.3 * prosody_score
    
    # Clean up temporary file
    os.remove(wav_file)
    
    return clarity_score

# Example usage
audio_file = "audio_test_beheard.wav"  # Replace with your audio file
expected_text = "This is an audio text to see if the program works"  # Optional: provide the expected text
clarity_score = compute_clarity_score(audio_file, expected_text)
print(f"Pronunciation Clarity Score: {clarity_score:.2f}/100")