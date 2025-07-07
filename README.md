https://www.kaggle.com/code/swift27/bearing-health-classi-16june
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

This paper proposes a two-path bearing fault diagnosis system using deep learning and language models and introduces a novel approach to data augmentation using RMS-scaled Gaussian noise. Vibration raw signals are split and transformed to 2D spectrograms by Short-Time Fourier Transform (STFT) in order to enable spatial learning in features. A CNN model is learned with augmented data with frequency masking, scaling in amplification, and our proposed RMS-scaled noise injection — and this fortifies fault classification in robusteness within signal variation. The model can classify bearing states in four states: Normal, Inner Race (IR), Outer Race (OR), and Ball Faults with accuracy.

In tandem, we conduct a retrieval-augmented generation (RAG) pipeline with a locally deployed LLaMA 3.2 model. Manually designed statistical features (RMS, kurtosis, skewness, crest factor, dominant frequency) are inserted through Sentence-BERT and compared to a domain knowledge base via FAISS. Retrieved cases are submitted to the LLM for explainable, feature-level diagnosis. Experimental outcomes demonstrate both models to have strong accuracy, providing a trade-off in predictive performance and explainability.



5. Importance of Bearings in Industry
Role of bearings in machinery

Impact of failure: downtime, safety, cost

Need for condition monitoring and predictive maintenance

🔹 6. Introduction
Problem statement

Motivation for using AI

Brief on your proposed approach (spectrogram + CNN + LLM)

Goals and objectives

🔹 7. Literature Review
Overview of traditional techniques (FFT, envelope detection)

ML/DL-based techniques for fault detection

Summary of previous research using the CWRU dataset

Comparison of methods (handcrafted vs deep features)

Identified research gap or scope for improvement

🔹 8. Dataset Description (CWRU)
Source and structure of CWRU dataset

Sampling rate, sensor location, types of faults (IR, OR, Ball)

File format (.mat), RPM levels

Class distribution

Data acquisition test rig (can include schematic)

🔹 9. Data Preprocessing
Signal segmentation: window size, overlap

Noise addition: RMS-scaled Gaussian noise

Spectrogram generation: method, shape (128×128)

Normalization

Data augmentation: frequency masking, amplitude scaling

Class labeling logic

Balanced dataset approach (if any)

🔹 10. Feature Extraction (Optional Section)
RMS, Kurtosis, Skewness, Crest Factor, Peak-to-Peak

Frequency domain feature: Dominant frequency

Periodogram

Table of extracted features for few sample segments

🔹 11. Model Architecture
Choice of CNN: Why CNN?

Full layer-by-layer description with diagram

Number of trainable parameters

Loss function: Sparse Categorical Crossentropy

Optimizer: Adam

Regularization: Dropout, BatchNorm

Activation functions used

🔹 12. Training Setup
Train-validation-test split (stratified, 80-20)

Number of epochs, batch size, early stopping

Class weights to handle imbalance

Logging mechanism (CSV logging per epoch)

Environment: Colab/Kaggle, GPU used

🔹 13. Evaluation & Results
Accuracy, Precision, Recall, F1-score

Confusion matrix (well-labeled)

Training curves: Accuracy & Loss

Comparison of training vs validation performance

Augmented vs non-augmented impact

Table summarizing results of different experiments

🔹 14. RAG + LLM-based Fault Explanation
How features were structured into a knowledge base

Use of SentenceTransformer + FAISS for similarity search

Integration of LLaMA 3 via Groq API

Example of LLM query + explanation

Comparison of LLM vs CNN predictions

Use case of explainability for industrial decision-making

🔹 15. Deployment Considerations
Model size, inference time

Suitability for real-time monitoring

Scope for edge deployment (e.g., Raspberry Pi)

UI / Interactive Demo (if applicable)

🔹 16. Challenges Faced
Large .mat file handling

Model convergence

Class imbalance

Noise sensitivity

API integration hurdles (e.g., Groq rate limits)

🔹 17. Learnings & Skills Gained
Vibration signal processing

Deep learning and CNNs

Data augmentation strategies

Spectrogram engineering

RAG pipelines and LLM usage

API usage and integration

🔹 18. Future Scope
Real-time system using embedded devices

Multi-sensor fusion (temperature + vibration)

Self-supervised fault diagnosis

Larger datasets (IMS, XJTU)

Explainable AI with attention maps or Grad-CAM

🔹 19. Conclusion
Recap of what was done and achieved

Summary of performance

Real-world relevance and potential

🔹 20. References
Research papers

Official dataset links

Library and framework documentation

LLM/embedding models used (e.g., SentenceTransformers)

🔹 21. Appendix
Code snippets (model definition, spectrogram generation)

Additional plots

Sample spectrograms for each fault class

Training logs

LLM responses

