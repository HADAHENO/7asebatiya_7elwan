# QuartzNet-based ASR System for Arabic Speech Recognition

### ✨ Check out "inference.py" to use our best model checkpoint for inference. Have fun experimenting with our model! 🚀

## Table of Contents
1. [📘 Introduction](##Introduction)
2. [🏗️ System Architecture](##System Architecture)
3. [🧠 Methodology](##Methodology)
4. [🔧 Technical Details](##Technical Details)
5. [📊 Results](##Results)
6. [🔁 Reproducibility](##Reproducibility)
7. [🚀 Performance Assessment](##Performance Assessment)

   

## Introduction 📘

This project implements an Automatic Speech Recognition (ASR) system for Arabic using the QuartzNet architecture. QuartzNet is an end-to-end neural acoustic model that achieves near state-of-the-art accuracy while using fewer parameters than competing models.

The QuartzNet architecture, as described in the original paper, is composed of multiple blocks with residual connections. Each block consists of one or more modules with 1D time-channel separable convolutional layers, batch normalization, and ReLU layers. The model is trained using Connectionist Temporal Classification (CTC) loss.

## System Architecture 🏗️

Our implementation uses the QuartzNet15x5 configuration, which consists of:

- An input layer
- 15 QuartzNet blocks
- An output layer

Each QuartzNet block contains:
- 1D time-channel separable convolutional layers
- Batch normalization
- ReLU activation
- Residual connections

The model uses 256 to 512 filters in different layers, with kernel sizes ranging from 33 to 87.

## Methodology 🧠

1. **Data Preparation**: We used the Arabic speech dataset provided by the MTC-AIC2 organizers. The data is processed and converted into manifest files for training and validation.

2. **Model Configuration**: We used the NeMo toolkit to configure and train the QuartzNet model. The configuration includes:
   - Audio preprocessing parameters
   - Model architecture details
   - Training hyperparameters

3. **Training**: The model was trained using PyTorch Lightning with the following settings:
   - Mixed precision (16-bit)
   - GPU acceleration
   - Maximum of 20 epochs
   - Novograd optimizer with Cosine Annealing learning rate schedule

4. **Evaluation**: We evaluated the model's performance using Word Error Rate (WER) on a validation set.

## Technical Details 🔧

### Model Configuration

- **Preprocessor**: AudioToMelSpectrogramPreprocessor
  - Window size: 0.02
  - Window stride: 0.01
  - Features: 64
  - n_fft: 512

- **Encoder**: ConvASREncoder
  - 15 blocks with varying configurations
  - Separable convolutions
  - Residual connections

- **Decoder**: ConvASRDecoder
  - 1024 input features
  - 41 output classes (Arabic characters)

### Training Configuration

- **Optimizer**: Novograd
  - Learning rate: 0.01
  - Betas: [0.8, 0.5]
  - Weight decay: 0.001

- **Learning Rate Schedule**: CosineAnnealing

- **Data Augmentation**: SpecAugment

## Results 📊

**We trained the model for various numbers of epochs (20, 40, 60, 80, 100, 120) and plotted the results:**

<table>
  <tr>
    <td>
      <strong>20_epochs:</strong>
      <br>
      <img src="https://github.com/HADAHENO/7asebatiya_7elwan/assets/93373983/c6f0e183-3cf3-4547-b45c-c32b014d190c" width="500" alt="20_epochs">
    </td>
    <td>
      <strong>40_epochs:</strong>
      <br>
      <img src="https://github.com/HADAHENO/7asebatiya_7elwan/assets/93373983/cfd80ab7-22da-4a4c-b8e1-7a05f635f416" width="500" alt="40_epochs">
    </td>
  </tr>
  <tr>
    <td>
      <strong>60_epochs:</strong>
      <br>
      <img src="https://github.com/HADAHENO/7asebatiya_7elwan/assets/93373983/8a8fd98e-c971-4176-a57b-2ddc0779d162" width="500" alt="60_epochs">
    </td>
    <td>
      <strong>80_epochs:</strong>
      <br>
      <img src="https://github.com/HADAHENO/7asebatiya_7elwan/assets/93373983/a4fbca71-5ca4-4384-9393-eb347d882b76" width="500" alt="80_epochs">
    </td>
  </tr>
  <tr>
    <td>
      <strong>100_epochs:</strong>
      <br>
      <img src="https://github.com/HADAHENO/7asebatiya_7elwan/assets/93373983/c153ef2a-a156-4cee-a79d-75a7c50225fa" width="500" alt="100_epochs">
    </td>
    <td>
      <strong>120_epochs:</strong>
      <br>
      <img src="https://github.com/HADAHENO/7asebatiya_7elwan/assets/93373983/19c565cf-8d46-4f19-bfa4-f73e8207df09" width="500" alt="120_epochs">
    </td>
  </tr>
</table>

## Reproducibility 🔁

To reproduce our results:

1. Install the required dependencies:

   - apt-get update && apt-get install -y libsndfile1 ffmpeg
   - pip -q install nemo_toolkit['asr'] Cython packaging
   - pip install torch
   - pip install pytorch-lightning
   - pip install omegaconf
   - pip install pandas
   - pip install matplotlib

3. Prepare your data and update the manifest file paths in the configuration.
   - To create the manifest file required for training the ASR model, you can use the following Python code. This code assumes you have prepared a CSV file with the following structure:
     
      | audio                           | transcript                    |
      |---------------------------------|-------------------------------|
      | path/to/audio_folder/file1.wav  | Transcript of the first audio |
      | path/to/audio_folder/file2.wav  | Transcript of the second audio|
   
    - Use the following Python script to generate the manifest file:
      ```python
      # Function to build a manifest
      def build_manifest(csv_path, manifest_path, wav_dir):
          
          df = pr.read_csv(csv_path)
      
          with open(manifest_path, 'w', encoding='utf-8') as fout:
              for index, row in df.iterrows():
                  audio_path = row['audio']
                  transcript = row['transcript']
      
                  duration = librosa.core.get_duration(filename=audio_path)
      
                  # Write the metadata to the manifest
                  metadata = {
                      "audio_filepath": audio_path,
                      "duration": duration,
                      "text": transcript
                  }
                  json.dump(metadata, fout)
                  fout.write('\n')
         ```

5. Use the provided Python script from training_notebook.ipynb to train the model. Please make sure you have the necessary computational resources (GPU recommended).

6. Adjust hyperparameters as needed for your specific use case.

## Performance Assessment 🚀

To assess the performance of the model:

1. Use the validation set WER as the primary metric.
2. Compare the model's performance against other state-of-the-art ASR systems for Arabic.
3. Evaluate the model's inference speed and resource usage.
4. Test the model on various Arabic dialects and accents to assess its generalization capabilities.

For a more comprehensive evaluation, consider using additional metrics such as Character Error Rate (CER).
