# SER

# Speech Emotion Recognition Using Real-Time Deep Learning and Diarization Techniques

In this project, we developed a real-time Speech Emotion Recognition (SER) system aimed at analyzing human emotions in spoken conversations using advanced deep learning techniques. The system takes an audio recording—such as a customer care call or in-person interaction—as input and outputs speaker-wise transcriptions along with detected emotional states. This has applications in customer service quality monitoring, mental health assessment, and intelligent human-computer interaction.

Our pipeline integrates several key components:

Speech-to-Text Transcription: Implemented using Whisper, which convert audio input into text with high accuracy.

Speaker Diarization: To identify and separate individual speakers in multi-speaker conversations, we employed PyAnnote's diarization model, enabling speaker-specific emotion detection.

Emotion Classification: We used a pre-trained emotion classifier fine-tuned on labeled emotional speech datasets to categorize each segment into emotion classes such as angry, happy, neutral, sad, etc.

Visualization: The final output includes a styled table displaying speaker IDs (e.g., Speaker 1, Speaker 2), their spoken segments, and the corresponding emotional labels.

The model pipeline was optimized for local GPU execution using PyTorch, with support for LoRA-based fine-tuning on custom emotion datasets, making it adaptable to various domain-specific use cases. This solution effectively bridges the gap between academic SER research and real-world, deployable applications.
