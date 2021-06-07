from ibm_watson import TextToSpeechV1
from ibm_watson import SpeechToTextV1
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from keys import Keys
import pyaudio
import pydub
import pydub.playback
import wave


class LanguageTranslator:

    @staticmethod
    def run_translator():
        input('Press enter and start recording a sentence')
        LanguageTranslator.record_audio('english.wav')
        speech_to_text_english = LanguageTranslator.speech_to_text(file_name='english.wav',
                                                                   model_id='en-US_BroadbandModel')
        print('English:', speech_to_text_english)
        italian_text = LanguageTranslator.translate(speech_to_text_english, model='en-it')
        print('Italian:', italian_text)
        LanguageTranslator.text_to_speech(text_to_speak=italian_text, voice_to_use='it-IT_FrancescaV3Voice',
                                          file_name='italian.wav')
        LanguageTranslator.play_audio(file_name='italian.wav')

    @staticmethod
    def speech_to_text(file_name, model_id):
        authenticator = IAMAuthenticator(Keys.SPEECH_TO_TEXT_IAM_APIKEY)
        stt = SpeechToTextV1(authenticator=authenticator)
        stt.set_service_url(Keys.SPEECH_TO_TEXT_URL)
        with open(file_name, 'rb') as audio_file:
            result = stt.recognize(audio=audio_file,
                                   content_type='audio/wav', model=model_id).get_result()

        results_list = result['results']
        speech_recognition_result = results_list[0]
        alternatives_list = speech_recognition_result['alternatives']
        first_alternative = alternatives_list[0]
        transcript = first_alternative['transcript']
        return transcript

    @staticmethod
    def translate(text_to_translate, model):
        authenticator = IAMAuthenticator(Keys.LANGUAGE_TRANSLATOR_IAM_APIKEY)
        language_translator = LanguageTranslatorV3(version='2018-05-01',authenticator=authenticator)
        language_translator.set_service_url(Keys.LANGUAGE_TRANSLATOR_URL)

        translated_text = language_translator.translate(
            text=text_to_translate, model_id=model).get_result()
        translations_list = translated_text['translations']
        first_translation = translations_list[0]
        translation = first_translation['translation']
        return translation

    @staticmethod
    def text_to_speech(text_to_speak, voice_to_use, file_name):
        authenticator = IAMAuthenticator(Keys.TEXT_TO_SPEECH_APIKEY)
        tts = TextToSpeechV1(authenticator=authenticator)
        tts.set_service_url(Keys.TEXT_TO_SPEECH_URL)
        with open(file_name, 'wb') as audio_file:
            audio_file.write(tts.synthesize(text_to_speak,
                                            accept='audio/wav', voice=voice_to_use).get_result().content)

    @staticmethod
    def record_audio(file_name):
        frame_rate = 44100
        chunk = 1024
        frame_format = pyaudio.paInt16
        frame_channels = 2
        seconds = 5

        recorder = pyaudio.PyAudio()
        audio_stream = recorder.open(format=frame_format, channels=frame_channels,
                                     rate=frame_rate, input=True, frames_per_buffer=chunk)
        audio_frames = []
        print('Recording 5 seconds of audio')

        for i in range(0, int(frame_rate * seconds / chunk)):
            audio_frames.append(audio_stream.read(chunk))

        print('Recording complete')
        audio_stream.stop_stream()
        audio_stream.close()
        recorder.terminate()

        with wave.open(file_name, 'wb') as output_file:
            output_file.setnchannels(frame_channels)
            output_file.setsampwidth(recorder.get_sample_size(frame_format))
            output_file.setframerate(frame_rate)
            output_file.writeframes(b''.join(audio_frames))

    @staticmethod
    def play_audio(file_name):
        sound = pydub.AudioSegment.from_wav(file_name)
        pydub.playback.play(sound)






