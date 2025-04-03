import os
import sounddevice as sd
import numpy as np
import wave
import threading
from contextlib import ExitStack

SAMPLE_RATE = 16000  
CHANNELS = 1         
audio_data_system = []
audio_data_mic = []
recording = False 

def device(device_name):
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device_name in device['name'] and device['max_input_channels'] > 0:
            return idx
    return None

def start_recording(mode, filename, print=print):
    global audio_data_system, audio_data_mic, recording
    audio_data_system, audio_data_mic = [], []
    recording = True

    dev_system = device("Stereo Mix") or device("Virtual Cable")
    dev_mic = device("Microphone")

    if mode == "system" and dev_system is None:
        print("No system audio input device found.")
        return
    elif mode == "microphone" and dev_mic is None:
        print("No microphone input device found.")
        return
    elif mode == "both" and (dev_system is None or dev_mic is None):
        print("Both devices not available.")
        return

    print(f"Recording started (Mode: {mode})")

    def system_callback(indata, frames, time, status):
        if status:
            print(f"Error in system audio stream: {status}")
        audio_data_system.append(indata.copy())

    def mic_callback(indata, frames, time, status):
        if status:
            print(f"Error in microphone stream: {status}")
        audio_data_mic.append(indata.copy())

    def thread():
        streams = []

        if mode in ["system", "both"]:
            streams.append(sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', device=dev_system, callback=system_callback))
        if mode in ["microphone", "both"]:
            streams.append(sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', device=dev_mic, callback=mic_callback))
        
        try:
            with ExitStack() as stack:
                active_streams = [stack.enter_context(stream) for stream in streams]
                while recording:
                    sd.sleep(100)
        except Exception as e:
            print(f"Error in recording thread: {e}")

    thread = threading.Thread(target=thread, daemon=True)
    thread.start()
    return thread

def stop_recording(mode, filename, print=print):
    global recording
    recording = False 
    print("Stopping recording...")

    base_path = os.path.join(os.path.dirname(__file__), "..", "Files", "Recordings")

    if mode == "system" and audio_data_system:
        save(os.path.join(base_path, "System", f"{filename}.wav"), audio_data_system)
    elif mode == "microphone" and audio_data_mic:
        save(os.path.join(base_path, "Microphone", f"{filename}.wav"), audio_data_mic)
    elif mode == "both" and (audio_data_system or audio_data_mic):
        mixed_audio = mix(audio_data_system, audio_data_mic)
        save(os.path.join(base_path, "Mixed", f"{filename}.wav"), mixed_audio)

    print("Recording saved.")

def mix(system_audio, mic_audio):
    if not system_audio and not mic_audio:
        return []

    # Convert lists to NumPy arrays
    system_audio = np.concatenate(system_audio).astype(np.int16) if system_audio else np.array([], dtype=np.int16)
    mic_audio = np.concatenate(mic_audio).astype(np.int16) if mic_audio else np.array([], dtype=np.int16)

    # Ensure both arrays have the same length by padding the shorter one
    max_length = max(len(system_audio), len(mic_audio))
    system_audio = np.pad(system_audio, (0, max_length - len(system_audio)), mode='constant')
    mic_audio = np.pad(mic_audio, (0, max_length - len(mic_audio)), mode='constant')

    # Mix both audio streams (averaging to prevent clipping)
    mixed_audio = ((system_audio.astype(np.int32) + mic_audio.astype(np.int32)) // 2).astype(np.int16)

    return [mixed_audio]

def save(filename, audio_data):
    if not audio_data:  # Check if the list is empty
        print(f"Warning: No audio data captured for {filename}.")
        return  # Avoid writing empty files

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.concatenate(audio_data).astype(np.int16).tobytes())

if __name__ == "__main__":
    filename = input("Enter filename: ")
    mode = input("Record (system/microphone/both): ").strip().lower()
    thread = start_recording(mode, filename)

    input("Press Enter to stop recording...\n")
    stop_recording(mode, filename)
