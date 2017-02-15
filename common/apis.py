import os
import speech_recognition as sr


r = sr.Recognizer()

def recognize_from_api(audio, api, name='API', safe=True, **kwargs):
    if not isinstance(audio, sr.AudioData):
        with sr.AudioFile(audio) as source:
            audio = r.record(source)
    try:
        return api(audio, **kwargs)
    except sr.UnknownValueError as e:
        if not safe:
            raise e
        return "\t%s could not understand audio" % name
    except sr.RequestError as e:
        if not safe:
            raise e
        return "\tCould not request results from %s \
    service; {0}" % (name, e)


def recognize_google(audio,
                     credentials=os.environ['GOOGLE_CLOUD_API'],
                     **kwargs):

    return recognize_from_api(audio, r.recognize_google_cloud,
                              name='Google Cloud Speech',
                              credentials_json=credentials,
                              **kwargs)


def recognize_bing(audio, key=os.environ['BING_API'], **kwargs):
    return recognize_from_api(audio, r.recognize_bing,
                              name='Microsoft Bing Voice',
                              key=key, **kwargs)


def recognize_ibm(audio,
                  username=os.environ['IBM_USERNAME'],
                  password=os.environ['IBM_PASSWORD'], **kwargs):
    return recognize_from_api(audio, r.recognize_ibm,
                              name='IBM Speech to Text',
                              username=username, password=password,
                              **kwargs)
