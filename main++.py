from collections import defaultdict, Counter
from flask import Flask, request, jsonify
import os
import torch
import whisper
from transformers import pipeline
from pyannote.audio import Pipeline
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {device}")

# Load Models
try:
    whisper_model = whisper.load_model("large", device=device)
    emotion_analyzer = pipeline(
        "text-classification",
        model="michellejieli/emotion_text_classifier",
        device=0 if torch.cuda.is_available() else -1,
    )
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="YOUR_HUGGINGFACE_TOKEN_HERE"
    )
    diarization_pipeline.to(torch.device(device))
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

ALLOWED_EXTENSIONS = {'mp3', 'm4a', 'mp4', 'wav'}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_audio_to_wav(file_path):
    """Converts non-WAV audio files to WAV format."""
    base, ext = os.path.splitext(file_path)
    if ext.lower() == '.wav':
        return file_path
    wav_file = f"{base}.wav"
    try:
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_file, format="wav")
        print("Audio converted to WAV.")
        return wav_file
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def merge_speakers(diarization, max_speakers=2):
    """Merges speakers based on speech duration."""
    speaker_durations = defaultdict(float)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_durations[speaker] += turn.end - turn.start

    sorted_speakers = sorted(speaker_durations, key=speaker_durations.get, reverse=True)[:max_speakers]
    return {speaker: f"Speaker {i+1}" for i, speaker in enumerate(sorted_speakers)}

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, transcription, diarization, and emotion analysis."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    print("Processing audio...")
    wav_file_path = convert_audio_to_wav(file_path)
    if not wav_file_path:
        return jsonify({"error": "Audio conversion failed"}), 500

    try:
        print("Running diarization...")
        diarization = diarization_pipeline(wav_file_path)
        print("Diarization complete.")
        speaker_mapping = merge_speakers(diarization)
    except Exception as e:
        print(f"Diarization error: {e}")
        return jsonify({"error": "Diarization failed"}), 500

    try:
        print("Transcribing...")
        result = whisper_model.transcribe(wav_file_path)
        segments = []
        emotion_counts = Counter()
        
        print("Analyzing emotions...")
        for segment in result.get('segments', []):
            start_time, end_time, text = segment['start'], segment['end'], segment['text']
            best_speaker, max_overlap = None, 0

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start, overlap_end = max(turn.start, start_time), min(turn.end, end_time)
                overlap_duration = max(0, overlap_end - overlap_start)
                if overlap_duration > max_overlap:
                    max_overlap, best_speaker = overlap_duration, speaker

            speaker_label = speaker_mapping.get(best_speaker, "Unknown")
            emotions = emotion_analyzer(text)
            detected_emotion = emotions[0]['label'] if emotions else "Neutral"
            emotion_counts[detected_emotion] += 1

            segments.append({
                "speaker": speaker_label,
                "start": start_time,
                "end": end_time,
                "text": text,
                "emotion": detected_emotion
            })

        overall_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Neutral"
        return jsonify({"segments": segments, "overall_emotion": overall_emotion})
    
    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({"error": "Processing failed"}), 500

@app.route('/')
def index():
    """Simple upload page."""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Speech Emotion Recognition</title>
        <style>
            body { font-family: 'Poppins', sans-serif;
                background-color: #0B0C10;
                color: #C5C6C7;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                text-align: center; }
            .container { background: #1F2833;
                padding: 50px;
                border-radius: 12px;
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.5);
                width: 100%;
                max-width: 700px; }
            h1 { font-size: 4em; color: #66FCF1; margin-bottom: 20px; }
            input[type="file"] {
                display: block;
                margin: 20px auto;
                padding: 12px;
                font-size: 1.1em;
                font-weight: bold;
                color: #1F2833;
                background: #66FCF1;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            button { background-color: #66FCF1;
                color: #1F2833;
                font-size: 1.2em;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease; }
            #loading { display: none;
                margin-top: 20px;
                font-size: 1.5em;
                color: #66FCF1;}
            #result { margin-top: 30px;
                padding: 20px;
                background: #0B0C10;
                border: 2px solid #66FCF1;
                border-radius: 10px;
                text-align: left;
                max-height: 400px;
                overflow-y: auto; }
        </style>
        <script>
            async function uploadFile(event) {
                event.preventDefault();
                const formData = new FormData();
                formData.append('file', document.querySelector('#fileInput').files[0]);

                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').innerHTML = '';

                const response = await fetch('/upload', { method: 'POST', body: formData });
                const result = await response.json();
                document.getElementById('loading').style.display = 'none';

                let resultDiv = document.querySelector('#result');
                resultDiv.innerHTML = `<h2 style="color:#66FCF1;">Overall Call Satisfaction: ${result.overall_emotion}</h2>`;

                result.segments.forEach(segment => {
                    resultDiv.innerHTML += `<p>
                        <strong>Speaker:</strong> ${segment.speaker} <br> 
                        <strong>Emotion:</strong> ${segment.emotion} <br> 
                        ${segment.text}
                    </p>`;
                });
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Speech Emotion Recognition</h1>
            <p>Upload an audio file, and we'll analyze the emotions behind the words.</p>
            <form id="uploadForm" onsubmit="uploadFile(event)">
                <input type="file" id="fileInput" accept=".mp3, .m4a, .mp4, .wav" required>
                <button type="submit">Upload & Analyze</button>
            </form>
            <div id="loading">Analyzing, please wait...</div>
            <div id="result"></div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=False)
