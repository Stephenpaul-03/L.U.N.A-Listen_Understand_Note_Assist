import os
import whisper

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
FILES_DIR = os.path.join(BASE_DIR, "Files")

MODEL_DIR = os.path.join(BASE_DIR, "Modules", "Models", "whisper_asr")
os.makedirs(MODEL_DIR, exist_ok=True)

RECORDINGS_DIR = os.path.join(FILES_DIR, "Recordings")
TRANSCRIPT_DIR = os.path.join(FILES_DIR, "Transcripts")
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

MODEL_NAME = "small"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Load Whisper Model
def load():
    print(f"Loading Whisper model: {MODEL_NAME} (stored in {MODEL_DIR})")
    if not os.path.exists(MODEL_PATH):
        print("Model not found, downloading...")
        model = whisper.load_model(MODEL_NAME, download_root=MODEL_DIR)
        print("Download complete.")
    else:
        print("Model found, loading from storage.")
        model = whisper.load_model(MODEL_NAME, download_root=MODEL_DIR)
    
    return model

# Transcription
def transcribe(folder, audio_filename):
    audio_file = os.path.join(RECORDINGS_DIR, folder, audio_filename)
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Error: The audio file '{audio_file}' does not exist.")    
    print(f"Transcribing '{audio_file}'...")
    model = load()
    output = model.transcribe(audio_file)
    transcript_text = output["text"]
    base_name = os.path.splitext(audio_filename)[0]
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    print(f"Transcription saved to: {transcript_path}")
    return transcript_text

if __name__ == "__main__":
    folder_options = ["system", "microphone", "both"]
    folder = input(f"Select a folder from {folder_options}: ").strip().lower()
    
    while folder not in folder_options:
        print("Invalid selection. Please choose from the given options.")
        folder = input(f"Select a folder from {folder_options}: ").strip().lower()
    
    audio_filename = input("Enter the name of the audio file (including extension): ").strip()
    
    try:
        transcription = transcribe(folder, audio_filename)
        print("\nSuccessfully Saved")
    except Exception as e:
        print("\nError:", str(e))
