import speech_recognition as sr




def get_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
    try:
        text = recognizer.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        return ""
    


import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main():
    while True:
        text = get_audio()
        if "hey echo" in text:
            # response = handle_command(text)  # Perform NLU and generate response
            # speak(response)  # Convert response to speech and speak it aloud
            return ""
            
        
        