# L.U.N.A - Listen_Understand_Note_Assist

L.U.N.A is a powerful AI-driven transcription, translation, and summarization tool designed to process audio recordings of meetings, seminars, and discussions. It leverages cutting-edge models from OpenAI and Hugging Face to provide accurate and efficient documentation of spoken content.

## Features

- **Audio Recording**: Captures system audio, microphone input, or both using `sounddevice`.
- **Transcription**: Uses OpenAI's Whisper model to convert speech to text with high accuracy.
- **Translation**: Utilizes Facebook's M2M100 model to translate transcribed text into multiple languages.
- **Summarization**: Employs DistilBERT to generate concise summaries of transcribed and translated content. [In Development]
- **Supports multiple languages**: Automatically detects and translates speech from various languages.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- ffmpeg (for handling audio files)

### Setup
```bash
# Clone the repository
git clone https://github.com/Stephenpaul-03/L.U.N.A-Listen_Understand_Note_Assist.git
cd L.U.N.A-Listen_Understand_Note_Assist
```

## Usage

### 1. Record an Audio File
```bash
python recorder.py  
# Follow prompts to select recording mode and filename
```

### 2. Transcribe an Audio File
```bash
python transcriber.py  
# Follow prompts to select the folder (system/microphone/both) and filename
```

### 3. Translate Transcribed Text
```bash
python translator.py  
# Follow prompts to enter the transcript filename and target language
```

## Dependencies
The application relies on the following libraries:
- `sounddevice`
- `numpy`
- `wave`
- `whisper`
- `transformers` (for M2M100 and DistilBERT)
- `torch`
- `langdetect`
- `ffmpeg-python`

Install them using:
```bash
pip install sounddevice numpy wave whisper transformers torch langdetect ffmpeg-python
```

## Future Development
- Working on the summarization module
- Creating a UI using PyQt 6
- Fixing kinks and dents in existing functionality

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Acknowledgments
- OpenAI for Whisper
- Facebook AI for M2M100
- Hugging Face for DistilBERT

