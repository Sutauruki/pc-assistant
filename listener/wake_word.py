from openwakeword.model import Model


def load_wake_word_model(wake_word="hey_jarvis"):
    print("Loading wake word model...")
    return Model(wakeword_models=[wake_word])


def check_wake_word(model, audio_chunk, threshold=0.5):
    """audio_chunk should be int16 mono numpy array.
    Returns the detected wakeword name if triggered, else None."""
    prediction = model.predict(audio_chunk)
    for wakeword, score in prediction.items():
        if score > threshold:
            print(f"Wake word detected: {wakeword} (score: {score:.2f})")
            return wakeword
    return None