# CLIP ExecuTorch Android Application

An on-device Android application for generating and comparing text and image embeddings using the OpenAI CLIP model. This project leverages PyTorch's ExecuTorch framework to run the CLIP Vision and Text encoders entirely locally, without requiring internet connectivity.

## Core Features

- **On-Device Inference**: Generates robust 512-dimensional embeddings directly on the Android device.
- **CPU Fallback**: Uses the ExecuTorch backend to ensure stable, mathematically deterministic inference. This avoids crashes and operator mismatches frequently encountered on mobile NPUs.
- **Custom BPE Tokenizer**: Integrates a fully native Kotlin Byte-Pair Encoding (BPE) tokenizer to process textual queries accurately against the vocabulary before sending them to the runtime.
- **Dimension Matching**: Patched Python export logic that forcefully guarantees standard 512 projection dimensions, preventing hidden layer sequence leakage when lowering the model.

## Build Instructions

### Prerequisites
- Android Studio (Koala or newer recommended)
- Java 11+
- ExecuTorch Android AAR libraries natively situated in the `/libs` directory.

### 1. Generating the Models
To run this application, you must generate the `clip_vision.pte` and `clip_text.pte` ExecuTorch binaries. These are excluded from version control due to their large size.

Execute the provided Python export script to lower the HuggingFace models:
```bash
python3 export_clip.py
```
Move the resulting `.pte` files into: `app/src/main/assets/models/`

### 2. Building the SDK
1. Open the project in Android Studio.
2. Synchronize the Gradle environment.
3. Build the Release or Debug APK directly to your physical Android device.

## Architecture & Limitations

The inference relies on `attn_implementation="eager"` during the export cycle because Flash Attention primitives are currently incompatible with mobile CPU lowering pipelines. As a result, inference takes slightly longer but guarantees absolute stability and crash-free execution.
