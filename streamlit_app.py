import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
filename = 'modelForPrediction1_best_one.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Function to extract features from audio data
def extract_feature(audio_data, sample_rate, mfcc=True, chroma=True, mel=True):
    try:
        if chroma:
            stft = np.abs(librosa.stft(audio_data))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
            mfccs = StandardScaler().fit_transform(mfccs.reshape(-1, 1)).reshape(-1)  # Normalize MFCCs
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result
    except Exception as e:
        st.exception(e)
        return None

# Streamlit UI
st.title("EmoAudio: Decode Emotions from Spoken Words")

st.markdown("""
EmoAudio is a powerful tool that deciphers emotions from spoken words. Whether you want to understand the sentiment behind a recorded conversation or explore the emotional depth of your own voice, EmoAudio has you covered. This intelligent tool utilizes advanced machine learning algorithms to analyze audio clips, recognizing emotions such as 'calm,' 'happy,' 'fearful,' and 'disgust.'
""")
st.markdown("****Created by Yugant Gotmare****")


# Choice for user to upload a file or record audio
option = st.radio("Choose an option:", ("Upload Audio File", "Record Audio"))

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an audio file...", type=["wav"])
    if uploaded_file:
        try:
            # Extract features from the uploaded audio file
            audio_data, sample_rate = sf.read(uploaded_file, dtype='float32')
            feature = extract_feature(audio_data, sample_rate, mfcc=True, chroma=True, mel=True)
            if feature is not None:
                feature = feature.reshape(1, -1)

                # Make prediction using the loaded model
                prediction = loaded_model.predict(feature)[0]

                # Display the predicted emotion
                st.success(f"Predicted Emotion: {prediction}")

                # Play the uploaded audio
                st.audio(audio_data, format='audio/wav', sample_rate=sample_rate)

        except Exception as e:
            st.error("Error processing the audio file. Please try a different file.")
            st.exception(e)

elif option == "Record Audio":
    if st.button("Record Audio"):
        st.write("Recording... Please speak into the microphone.")
        audio_data = sd.rec(int(3 * 44100), samplerate=44100, channels=1, dtype='float32')
        sd.wait()
        st.write("Recording completed.")

        # Extract features from recorded audio
        feature = extract_feature(audio_data.flatten(), 44100, mfcc=True, chroma=True, mel=True)
        if feature is not None:
            feature = feature.reshape(1, -1)

            # Make prediction using the loaded model
            prediction = loaded_model.predict(feature)[0]

            # Display the predicted emotion
            st.success(f"Predicted Emotion: {prediction}")

            # Play the recorded audio
            st.audio(audio_data, format='audio/wav', sample_rate=44100)



