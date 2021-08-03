import sys

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from scipy.io.wavfile import write
import os
import io
sys.path.insert(0, '..')

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/mnt/tmp/fluent-grin-113303-64d6b3493f7c.json"

from google.cloud import texttospeech

# Instantiates a client
tts_client = texttospeech.TextToSpeechClient()

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

from inference import TTS
tts = TTS()
tts.load_model()

#@app.route('/tts', methods=['GET', 'POST'])
#def texttospeech():
#  if request.method == 'POST':
#    result = request.form
#    lang = result['input_language']
#    text = result['input_text']
#    if lang == t2s.language:
#        audio = t2s.tts(text)
#    else:
#        audio = t2s.update_model(lang).tts(text)
#
#    def generate():
#
#    #return Response(generate(), mimetype="audio/mp3")
#    return render_template('simple.html', voice=audio, sample_text=text, opt_lang=t2s.language)

@app.get("/google_voice")
async def voice(txt: str):
    #list_voices()
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=txt)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", 
        name="en-US-Wavenet-F",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    async def generate():
        with io.BytesIO(response.audio_content) as buf:
            data = buf.read(1024)
            while data:
                yield data
                data = buf.read(1024)

    return StreamingResponse(generate(), media_type="audio/wav")

def list_voices():
    """Lists the available voices."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    # Performs the list voices request
    voices = client.list_voices()

    for voice in voices.voices:
        # Display the voice's name. Example: tpc-vocoded
        print(f"Name: {voice.name}")

        # Display the supported language codes for this voice. Example: "en-US"
        for language_code in voice.language_codes:
            print(f"Supported language: {language_code}")

        ssml_gender = texttospeech.SsmlVoiceGender(voice.ssml_gender)

        # Display the SSML Voice Gender
        print(f"SSML Voice Gender: {ssml_gender.name}")

        # Display the natural sample rate hertz for this voice. Example: 24000
        print(f"Natural Sample Rate Hertz: {voice.natural_sample_rate_hertz}\n")


@app.get("/voice")
async def voice(txt: str):
  txts = [txt]
  audios, alignments = tts.forward(txts)
  print(len(audios[0]))
  print(len(alignments[0]))

  async def generate():
    with io.BytesIO() as buf:
      write(buf, tts.args.sampling_rate, audios[0].cpu().numpy())
      buf.seek(0)

      data = buf.read(1024)
      while data:
        yield data
        data = buf.read(1024)

  return StreamingResponse(generate(), media_type="audio/wav")

#if __name__ == "__main__":
#  print("here")
#  import os
#  print(os.environ)
#  uvicorn.run("main:app", port=8000, reload=True)

