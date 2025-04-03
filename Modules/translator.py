import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import os
from langdetect import detect
import re

# Model Lirectory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "m2m100_base")
MODEL_NAME = "facebook/m2m100_418M"

# Offline Download
if not os.path.exists(MODEL_DIR):
    print("Downloading model...")
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
else:
    print("Loading model from offline storage...")
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_DIR)
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_DIR)
Supported = tokenizer.lang_code_to_id.keys()

# Split text
def split(text):
    return re.split(r'(?<=[.!?]) +', text)

# Load text
def load(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# Save text
def save(text, name, target):
    translations = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Files", "Translations")
    os.makedirs(translations, exist_ok=True)

    translation_filename = f"{name}_translated_{target}.txt"
    translation_file_path = os.path.join(translations, translation_filename)

    with open(translation_file_path, 'w', encoding='utf-8') as file:
        file.write(text)

    return translation_file_path

# Translate text
def translate(text, target="en", source="auto"):
    if not text.strip():
        return "No input text provided."

    translated = []

    if source != "auto":  
        if source not in Supported:
            return f"Error: The provided language '{source}' is not supported."
        
        print(f"Using provided language: {source} → Translating to {target}")
        tokenizer.src_lang = source
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target))
        
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    # autodetect ? works so DO NOT CHANGE ANYTHING
    sentences = split(text)
    
    for sentence in sentences:
        if not sentence.strip():
            continue

        detected = detect(sentence)
        print(f"Detected Language: {detected} → Translating to {target}")

        if detected not in Supported:
            translated.append(f"[Unsupported: {sentence}]")
            continue

        tokenizer.src_lang = detected
        inputs = tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target))
        
        translated.append(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])

    return " ".join(translated)

def main():
    transcripts_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Files", "Transcripts")
    
    filename = input("Enter the name of the transcript file (without .txt extension): ").strip()
    file_path = os.path.join(transcripts_folder, f"{filename}.txt")

    if not os.path.exists(file_path):
        print(f"Error: The file '{filename}.txt' does not exist in Transcripts folder.")
        return

    target = input("Enter target language code (e.g., 'en' for English, 'es' for Spanish): ").strip()
    source = input("Enter source language code or type 'auto' for automatic detection: ").strip()

    print("\nLoading and translating text...")
    text = load(file_path)

    if not text:
        print("Error: The file is empty.")
        return

    translated_text = translate(text, target, source)

    print("\nSaving translation to file...")
    translation_file_path = save(translated_text, filename, target)
    
    print(f"\nTranslation saved: {translation_file_path}")

if __name__ == "__main__":
    main()