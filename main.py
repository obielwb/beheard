import os
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
import json


# Load environment variables from .env
load_dotenv()

speech_key = os.getenv("SPEECH_KEY")
service_region = os.getenv("SERVICE_REGION")



average_intonation = 0


speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

filename = 'audio_test_beheard.wav'
audio_config = speechsdk.audio.AudioConfig(filename=filename)
language = 'en-US'
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language=language, audio_config=audio_config)


enable_miscue, enable_prosody = True, True
config_json = {
    "GradingSystem": "HundredMark",
    "Granularity": "Word",
    "Dimension": "Comprehensive",
    "ScenarioId": "",  # "" is the default scenario or ask product team for a customized one
    "EnableMiscue": enable_miscue,
    "EnableProsodyAssessment": enable_prosody,
    "NBestPhonemeCount": 0,  # > 0 to enable "spoken phoneme" mode, 0 to disable
}
pronunciation_config = speechsdk.PronunciationAssessmentConfig(json_string=json.dumps(config_json))


# First pass: Transcribe audio
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language=language, audio_config=audio_config)
result = speech_recognizer.recognize_once_async().get()
reference_text = result.text

# Second pass: Pronunciation assessment
pronunciation_config.reference_text = reference_text
pronunciation_config.apply_to(speech_recognizer)
pronunciation_result = speech_recognizer.recognize_once_async().get()

print("Recognized: {}".format(pronunciation_result.text))

if pronunciation_result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("Recognized: {}".format(pronunciation_result.text))
    # ✅ GET THE JSON RESULT
    json_result = pronunciation_result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
    result_dict = json.loads(json_result)
    print(json.dumps(result_dict, indent=2))  # Inspect structure

    pronunciation_result_parsed_01 = result_dict['NBest'][0]
    print(json.dumps(pronunciation_result_parsed_01, indent=2))  # Inspect structure


    per_word_pronounciation_assessment_result = []
    final_pronounciation_assessment_result = []

    pronounciation_assessment_result = pronunciation_result_parsed_01['PronunciationAssessment']

    # print(pronounciation_assessment_result)
    final_pronounciation_assessment_result.append(pronounciation_assessment_result)
    length = len(pronunciation_result_parsed_01['Words'])
    for i in range(len(pronunciation_result_parsed_01['Words'])):

        # del pronunciation_result_parsed_01['Words'][i]['Syllables']
        del pronunciation_result_parsed_01['Words'][i]['Phonemes']

        # print(pronunciation_result_parsed_01['Words'][i])
        intonation_results = pronunciation_result_parsed_01['Words'][i]['PronunciationAssessment']['Feedback']['Prosody']['Intonation']
        # print(intonation_results)
        intonation = pronunciation_result_parsed_01['Words'][i]['PronunciationAssessment']['Feedback']['Prosody']['Intonation']['Monotone']['SyllablePitchDeltaConfidence']
        average_intonation = (average_intonation + intonation) 
        # print(intonation)
        per_word_pronounciation_assessment_result.append(pronunciation_result_parsed_01['Words'][i])
        per_word_pronounciation_assessment_result.append(intonation_results)
        # per_word_pronounciation_assessment_result.append(pitch_per_word[i])

        del pronunciation_result_parsed_01['Words'][i]['PronunciationAssessment']['Feedback']
    final_pronounciation_assessment_result.append(per_word_pronounciation_assessment_result)

    # print(per_word_pronounciation_assessment_result)
    # print(f'\n\n')

    average_intonation = average_intonation / length
    accuracy = pronounciation_assessment_result['AccuracyScore']/100
    fluency = pronounciation_assessment_result['FluencyScore']/100
    prosody_score = pronounciation_assessment_result['ProsodyScore']/100
    completeness = pronounciation_assessment_result['CompletenessScore']/100
    avg_pro_score = pronounciation_assessment_result['PronScore']/100

    print(f'Accuracy: {accuracy}')
    print(f'Fluency: {fluency}')
    print(f'ProsodyScore: {prosody_score}')
    print(f'Completeness: {completeness}')
    print(f'AveragePronounciationScore: {avg_pro_score}')
    print(f'Intonation: {average_intonation}')
    print(f'PerWordPronounciationAssessmentResult: {per_word_pronounciation_assessment_result}')
    print(f'FinalPronounciationAssessmentResult: {final_pronounciation_assessment_result}')

else:
    print(f"Recognition failed: {pronunciation_result.reason}")