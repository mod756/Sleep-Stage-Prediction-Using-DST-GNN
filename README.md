💤 Sleep Stage Detection from Multi-Signal Fusion with Dynamic Spatial–Temporal Graph Neural Networks
=====================================================================================================

This repository contains Python scripts for **preprocessing**, **training**, and **demonstration** of a deep learning model for automated sleep stage classification based on **EEG, EOG, and EMG** signals from the **Sleep-EDF Expanded Dataset**.

The preprocessing script (preprocessing.py) prepares the data with **GPU acceleration** using CuPy, while the training script (protraining.py) implements an improved **Dyanmic Spatial–Temporal Graph Neural Network (DSTGNN)** model with custom loss functions and optimization strategies for improved detection of minority stages (e.g., N1, N3).

A separate demo script (demo.py) allows direct prediction of sleep stage from a pair of raw .edf and .Hypnogram.edf files using the trained model.

🧠 Core Idea
------------

The model follows the **ASTGSleep** framework:

*   Computes **Differential Entropy (DE)** features across 9 EEG frequency bands
    
*   Uses **context windows** of 9 epochs (4.5 minutes)
    
*   Models both **temporal and spatial dependencies** using a dynamic STGNN
    
*   Incorporates **attention**, **Chebyshev graph convolutions**, and **adaptive loss balancing**
    

⚙️ System Requirements
----------------------

### Software

*   **Python:** 3.10–3.12
    
*   ⚠️ On Windows, DataLoader uses num\_workers=0 to avoid multiprocessing issues.
    

#### Required Packages

`   pip install numpy scipy mne torch torchvision torchaudio scikit-learn matplotlib seaborn   `

#### For GPU acceleration (optional)

`   pip install cupy-cuda11x   # For CUDA 11.x  pip install cupy-cuda12x   # For CUDA 12.x   `

#### Verify Installation

`   python -c "import torch; print(torch.cuda.is_available())"  python -c "import cupy; print(cupy.cuda.Device().mem_info)"   `

### Hardware

| Component | Recommended |
|------------|-------------|
| **CPU** | Intel i5 or better |
| **GPU** | NVIDIA CUDA-enabled (≥ GTX 1060, 4GB VRAM) |
| **RAM** | ≥ 16 GB (32 GB recommended) |
| **Disk** | ≥ 10 GB free space |


🧩 Installation Steps
---------------------

### 1️⃣ Create a Virtual Environment

`   python -m venv sleep_env   `

**Activate:**
 `

### 2️⃣ Install Dependencies

`   pip install numpy scipy mne torch torchvision torchaudio scikit-learn matplotlib seaborn   `

### 3️⃣ Enable GPU (Optional)

`   pip install cupy-cuda11x  # For CUDA 11.x  # OR  pip install cupy-cuda12x  # For CUDA 12.x   `

🧱 Step-by-Step Workflow
------------------------

### Step 1: Download the Dataset

1.  Download **Sleep-EDF Expanded (v1.0.0)** from [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/).
    
2.  Extract the files. Folder structure should look like:


sleep-edf-database-expanded-1.0.0/sleep-cassette/  
├── SC4001E0-PSG.edf  
├── SC4001E0-Hypnogram.edf  
└── ...   


3.  Update the dataset path in preprocessing.py:
    

`   DATA_DIR = "sleep-edf-database-expanded-1.0.0/sleep-cassette"   `

### Step 2: Run Preprocessing (preprocessing.py)

This script:

*   Selects EEG, EOG, and EMG channels
    
*   Applies **Notch** and **Bandpass** filters
    
*   Resamples to 100 Hz
    
*   Computes **Differential Entropy (DE)** features
    
*   Creates **9-epoch context windows**
    
*   Saves features in compressed .npz format
    

**Output Files:**

*   Individual subjects → preprocessed\_subjects\_gpu/subject\_0000.npz
    
*   Combined dataset → preprocessed\_sleep\_edf\_gpu\_final.npz
    

**Default Parameters**

| Parameter         | Value       |
|-------------------|-------------|
| **Target Sampling Rate** | 100 Hz |
| **Epoch Length**  | 30 s   |
| **Context Epochs** | 9DE   |
| **Bands**  | 9   |


**Output Directory:** preprocessed\_subjects\_gpu/

### Step 3: Train the Model (protraining.py)

This script trains the **DSTGNN** model using the preprocessed .npz file.

**Features**

*   Adaptive loss weighting for minority classes
    
*   Mixed precision (AMP) training
    
*   Dynamic graph learning for spatial & temporal relationships
    
*   Real-time validation and early stopping
    

**Output Files**

*   best\_stgnn\_v2\_improved.pt — trained model checkpoint
    
*   training\_curves\_stgnn\_v2.png — loss and F1 curves
    
*   confusion\_matrix\_stgnn\_v2.png — test confusion matrix
    
*   edges\_visualizations\_v2/ — learned graph visualization
    

**Expected Results**

| Metric            | Typical Value |
|-------------------|-------------|
| **Accuracy**      | 88-90%      |
| **Macro-F1**      | 0.70–0.80   |
| **Cohen’s Kappa** | ~0.81       |




🧪 Step 4: Run the Demo (demo.py)
---------------------------------

Use demo.py to test the trained model on **any PSG + Hypnogram pair**.

**How It Works**

1.  Loads EEG/EOG/EMG signals and annotations
    
2.  Runs preprocessing pipeline
    
3.  Loads the trained STGNN model
    
4.  Runs inference on one window
    
5.  Displays predicted and true sleep stages
    

**Run Command**

`   python demo.py   `

**Example Output**

`⚙️  Using device: cuda` 

`📥 Loading PSG and Hypnogram files... ` 

`Loaded PSG: ST7011J0-PSG.edf `

`Sampling rate: 100 Hz` 

`Channels: ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']`  

`⚙️ Preprocessing signals... `

`✅ Extracted 224 windows from ST7011J0-PSG.edf  `

`📦 Loading trained STGNN model...`

`✅ Model and projector loaded successfully.  `

`🚀 Running inference on one sample...  `

`=== 💤 Single File Sleep Stage Prediction ===  `

`True stage label (from Hypnogram):N2  `

`Predicted stage (raw argmax):N2`

`Predicted stage (after calibration): N2  `

`Class probabilities: [[0.002 0.018 0.953 0.009 0.018]]  `

`✅ Prediction complete!`

🧩 Sleep Stage Labels
---------------------

| Label | Stage | Description |
|--------|--------|-------------|
| **0** | Wake | Awake stage |
| **1** | N1 | Light sleep |
| **2** | N2 | Stable sleep |
| **3** | N3 | Deep slow-wave sleep |
| **4** | REM | Rapid eye movement (dream) sleep |


🧾 Troubleshooting
------------------

| Issue | Fix |
|-------|-----|
| **_pickle.UnpicklingError during model loading** | PyTorch ≥ 2.6 requires `weights_only=False` (already fixed in `demo.py`). |
| **“No valid features extracted”** | Ensure correct PSG–Hypnogram pairing and valid EEG/EOG/EMG channels. |
| **CuPy not installed** | Install with `pip install cupy-cuda11x` or `pip install cupy-cuda12x`. |
| **CPU too slow** | Enable GPU by setting `use_gpu=True` in `SleepEDFPreprocessor`. |


👨‍💻 Contributors
------------------

## Contributors

This project was developed as part of the course **EC365 - AI for Biomedical Signal Interpretation**.

| Name              | Roll Number |
|-------------------|-------------|
| **KV Modak**      | 23bcs067    |
| **P Aneesh**  | 23bcs095    |
| **R Janaki Ram**      | 23bcs141   |
| **Amith Mathew**  | 23bec005    |


🏁 Quick Reference Commands
---------------------------
- Preprocess Sleep-EDF dataset  python preprocessing.py  
- Train the improved STGNN model  python protraining.py  
- Run a single-file demo prediction  python demo.py   `