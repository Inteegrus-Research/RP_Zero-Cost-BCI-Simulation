# 🧠 Zero-Cost BCI: Simulated EEG Signal Classification using Streamlit

This project demonstrates a **Brain-Computer Interface (BCI)** simulation using real EEG signals to classify binary decisions (Yes/No). It is designed as an accessible prototype for researchers and students to understand and explore cognitive signal processing—**without needing expensive hardware**.

---

## 🛠️ Tech Stack
- **Python 3.x**
- **Streamlit** for interactive UI
- **MNE-Python** for EEG signal handling
- **Scikit-learn** for machine learning (Logistic Regression)
- **MATLAB `.mat` EEG Dataset** for simulated input

---

## 🚀 Features
- 📁 Upload EEG `.mat` data
- 📊 Visualize EEG signals per channel
- 📈 Real-time FFT analysis
- 🧠 Classify EEG patterns into Yes/No decisions
- 🔄 Simulated speller (Yes/No toggle) demo
- 💬 Output performance metrics (accuracy, confusion matrix, report)

---

## 🧪 How It Works
1. **Upload**: Load a `.mat` EEG dataset containing signals.
2. **Visualization**: Explore channels, zoom into ranges, and view FFT spectra.
3. **Classification**: EEG segments are classified using a logistic regression model trained on input data.
4. **Live Demo**: A mock interface shows how real-time EEG response might trigger "Yes" or "No" decisions.

---

<pre>
## 📂 Folder Structure
├── app.py            # Streamlit UI
├── sample.mat        # EEG dataset (example)
├── screenshots/      # UI images
└── README.md         # This file
</pre>
---

## 📌 Applications
- **Educational BCI Prototyping**
- **Neuroadaptive System Simulations**
- **Assistive Communication Research**
- **Cognitive Load/Attention Detection (Basic)**

---

## 📄 License
This project is licensed under the **MIT License**. Feel free to adapt or build upon it.

---

## 🙋 Contact
Project by: *Keerthi Kumar K J*  
For queries: inteegrus.research@gmail.com

---
