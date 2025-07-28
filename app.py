import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io
import time

st.set_page_config(page_title="EEG BCI Simulator", layout="wide")

# —————————————————————————————————————————
# Helper: Bandpower (μ & β)
# —————————————————————————————————————————
def bandpower(epoch, fs, fmin, fmax):
    f, Pxx = welch(epoch, fs=fs, nperseg=fs*2)
    mask = (f >= fmin) & (f <= fmax)
    return np.log(np.trapz(Pxx[mask], f[mask]) + 1e-8)

# —————————————————————————————————————————
# Load EEG Data from .mat
# —————————————————————————————————————————
def load_eeg(mat_file):
    mat = loadmat(mat_file)
    eeg = mat.get('cnt', None)
    nfo = mat.get('nfo', None)
    if eeg is None:
        st.error("No EEG data found under key 'cnt'.")
        st.stop()

    # Clean extraction of channel names as strings
    channels = []
    if nfo is not None:
        try:
            raw_clab = nfo[0][0]['clab'][0]         # this is often an array of 1‑element arrays
            for ch in raw_clab:
                # each ch might be a 1‑element array or bytes, so coerce to str
                if isinstance(ch, (np.ndarray, list, tuple)):
                    channels.append(str(ch[0]))
                else:
                    channels.append(str(ch))
        except Exception:
            channels = []
    if not channels:
        channels = [f"Ch{i}" for i in range(eeg.shape[1])]

    # Sampling freq (fallback to 100 Hz)
    try:
        fs = float(nfo[0][0]['fs'][0][0])
    except Exception:
        fs = 100.0

    return eeg, channels, fs


# —————————————————————————————————————————
# Sidebar: Upload & Instructions
# —————————————————————————————————————————
st.sidebar.title("🔧 Controls")
uploaded = st.sidebar.file_uploader("Upload EEG .mat file", type="mat")
st.sidebar.markdown(
    """
    1. Upload a `.mat` file  
    2. Step through tabs:  
      • Data Overview  
      • Visualization  
      • Classification  
      • Live Demo  
    """
)

if not uploaded:
    st.title("🧠 EEG BCI Simulator")
    st.info("Please upload a `.mat` EEG file to begin.")
    st.stop()

# Load data once
eeg, channels, fs = load_eeg(uploaded)
n_samples, n_channels = eeg.shape

# —————————————————————————————————————————
# Tabs Setup
# —————————————————————————————————————————
tab1, tab2, tab3, tab4 = st.tabs([
    "1 – Data Overview", 
    "2 – Visualization", 
    "3 – Classification", 
    "4 – Live Demo"
])

# —————————————————————————————————————————
# Tab 1: Data Overview
# —————————————————————————————————————————
with tab1:
    st.title("1 – Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples", n_samples)
    col2.metric("Channels", n_channels)
    col3.metric("Sampling Freq (Hz)", fs)

    # — Range selector instead of fixed 5 rows —
    st.subheader("Raw Data Preview (select range)")
    start_idx, end_idx = st.slider(
        "Select sample range",
        min_value=0,
        max_value=n_samples-1,
        value=(0, min(100, n_samples-1)),
        step=1
    )
    # Show the table for that slice
    df_preview = pd.DataFrame(
        eeg[start_idx:end_idx+1, :],
        columns=channels
    )
    st.write(f"Showing rows {start_idx} to {end_idx} (inclusive):")
    st.dataframe(df_preview, width=800)



# —————————————————————————————————————————
# Tab 2: Visualization
# —————————————————————————————————————————
with tab2:
    st.title("2 – Visualization")
    channel_idx = st.slider("Select Channel", 0, n_channels-1, 0)
    show_fft = st.checkbox("Show Frequency Spectrum (FFT)")
    
    # Time Series Plot
    fig_ts, ax_ts = plt.subplots(figsize=(10, 3))
    ax_ts.plot(eeg[:, channel_idx])
    ax_ts.set_title(f"Time Series – {channels[channel_idx]}")
    ax_ts.set_xlabel("Sample")
    ax_ts.set_ylabel("Amplitude")
    st.pyplot(fig_ts)
    
    # FFT Plot
    if show_fft:
        fig_fft, ax_fft = plt.subplots(figsize=(10, 3))
        N = len(eeg[:, channel_idx])
        fft_vals = np.fft.fft(eeg[:, channel_idx])
        freqs = np.fft.fftfreq(N, d=1/fs)
        mask = freqs >= 0
        ax_fft.plot(freqs[mask], np.abs(fft_vals)[mask])
        ax_fft.set_title(f"Frequency Spectrum – {channels[channel_idx]}")
        ax_fft.set_xlabel("Frequency (Hz)")
        ax_fft.set_ylabel("Magnitude")
        st.pyplot(fig_fft)
    
    # Downloads
    st.markdown("**Download Results**")
    csv = pd.DataFrame(eeg[:, channel_idx], columns=[channels[channel_idx]]).to_csv(index=False)
    st.download_button("Download Channel Data (CSV)", csv, f"{channels[channel_idx]}_data.csv")
    buf_ts = io.BytesIO()
    fig_ts.savefig(buf_ts, format="png", bbox_inches="tight")
    st.download_button("Download Time Series Plot", buf_ts.getvalue(), f"{channels[channel_idx]}_timeseries.png")
    if show_fft:
        buf_fft = io.BytesIO()
        fig_fft.savefig(buf_fft, format="png", bbox_inches="tight")
        st.download_button("Download FFT Plot", buf_fft.getvalue(), f"{channels[channel_idx]}_fft.png")

# —————————————————————————————————————————
# Tab 3: Classification
# —————————————————————————————————————————
with tab3:
    st.title("3 – Classification & Decision")
    if st.button("Run Classification"):
        # Epoching: 2 s non-overlapping windows on selected channel
        window = int(2 * fs)
        epochs = [eeg[i:i+window, channel_idx] for i in range(0, n_samples - window + 1, window)]
        epochs = np.array(epochs)
        
        # Feature matrix
        X = np.array([[bandpower(ep, fs, 8, 12),
                       bandpower(ep, fs, 12, 30)] for ep in epochs])
        # Dummy labels
        y = np.random.randint(0, 2, size=X.shape[0])
        
        # Train/test
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        
        # Metrics
        acc = accuracy_score(yte, ypred)
        st.metric("Accuracy", f"{acc*100:.1f}%")
        
        cm = confusion_matrix(yte, ypred)
        fig_cm, ax_cm = plt.subplots(figsize=(4,4))
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
        ax_cm.set_title("Confusion Matrix")
        for (i,j), v in np.ndenumerate(cm):
            ax_cm.text(j, i, v, ha="center", va="center", color="white")
        st.pyplot(fig_cm)
        
        st.text("Classification Report")
        st.text(classification_report(yte, ypred))
        
        # Simulated Decision
        decision = "YES" if np.mean(ypred) > 0.5 else "NO"
        if decision == "YES":
            st.success(f"🧠 Simulated Decision: {decision}")
        else:
            st.warning(f"🧠 Simulated Decision: {decision}")

# —————————————————————————————————————————
# Tab 4: Live Demo
# —————————————————————————————————————————
with tab4:
    st.title("4 – Live Simulation")
    if st.button("Start Live Demo"):
        placeholder = st.empty()
        # Train model (same as above)
        window = int(2 * fs)
        epochs = [eeg[i:i+window, channel_idx] for i in range(0, n_samples - window + 1, int(fs*0.5))]
        X_live = np.array([[bandpower(ep, fs, 8, 12),
                            bandpower(ep, fs, 12, 30)] for ep in epochs])
        y_dummy = np.random.randint(0, 2, size=X_live.shape[0])
        clf_live = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_live.fit(X_live, y_dummy)
        
        # Streaming simulation
        for idx, start in enumerate(range(0, n_samples - window + 1, int(fs*0.5))):
            ep = eeg[start:start+window, channel_idx]
            feat = [bandpower(ep, fs, 8, 12), bandpower(ep, fs, 12, 30)]
            pred = clf_live.predict([feat])[0]
            t_sec = start / fs
            label = "YES" if pred == 1 else "NO"
            placeholder.metric(label="Time (s)", value=f"{t_sec:.1f}", delta=f"Decision: {label}")
            time.sleep(0.5)

st.success("🎉 Your EEG BCI Simulator is ready to use!")
