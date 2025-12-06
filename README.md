# **AAI-521 Final Project – Vehicle Detection, Classification, and Traffic Analytics**  
**Shiley-Marcos School of Engineering – University of San Diego**  
**Master of Applied Artificial Intelligence – Spring 2025**  

## **Team Members**
- **Birendra Khimding**  
- **Victor Hugo Germano**
- **Matt Hashemi**  

---

# **1. Project Summary**

This project develops a full **computer vision pipeline** for detecting, classifying, and analyzing traffic patterns using deep learning and video processing techniques.  
The system takes images or videos as input and automatically:

1. **Detects vehicles**  
2. **Classifies** them into categories (Car, Van, Bus, Others)  
3. **Maps classes into LMV/HMV** (Light vs Heavy Motor Vehicles)  
4. **Counts vehicles over time**  
5. **Analyzes traffic density and composition**  
6. **Runs on real-life video** (deployment-style demo)  
7. (Optional) **Enhances degraded images** using restoration models  

The end-to-end pipeline demonstrates the practical application of **convolutional neural networks**, **video analytics**, and **computer-vision engineering**.

---

# **2. Project Objectives**

Our main objectives were:

- Build a reliable **vehicle classification model** using deep learning.
- Create a **data preprocessing pipeline** directly from the UA-DETRAC dataset.
- Produce detailed **traffic analytics** relevant for real-world applications (LMV/HMV ratios, densities, temporal patterns).
- Demonstrate **deployment feasibility** by applying the model to a real-life traffic video.
- Document a clear **research and engineering workflow** via notebooks and a written report.

---

# **3. Roadmap & Development Plan**

We organized the project around a series of Jupyter notebooks, each representing a clear phase of the pipeline:

### **Step 1 – Dataset Exploration (`01_dataset_exploration.ipynb`)**
- Load DETRAC image sequences and XML annotations.
- Visualize frames and bounding boxes.
- Compute statistics on:
  - Number of vehicles per frame,
  - Distribution of classes,
  - Bounding box sizes and aspect ratios.
- Identify dataset challenges: imbalance, occlusions, varying illumination, motion blur.

### **Step 2 – Data Preprocessing & Cropping (`02_preprocessing_and_cropping.ipynb`)**
- Parse XML annotations for all sequences.
- Crop individual vehicle images from full frames.
- Normalize and resize cropped patches to a fixed input size.
- Save a standardized dataset as `cropped_vehicle_dataset.npz`.
- Summarize class distribution (Car, Van, Bus, Others) and dataset size.

### **Step 3 – Build & Train Classifier (PyTorch) (`03_vehicle_classification_model.ipynb`)**
- Implement a custom **PyTorch CNN** for vehicle classification.
- Split the cropped dataset into training/validation sets.
- Train the model and monitor:
  - Training and validation loss,
  - Accuracy curves.
- Evaluate with:
  - Confusion matrix,
  - Classification report,
  - Example predictions on validation images.
- Save the trained model as `models/vehicle_classifier.pth`.

### **Step 4 – Vehicle Counting & Time-Series Analysis (`04_vehicle_counting_and_analysis.ipynb`)**
- Combine:
  - Frame-level detections (bounding boxes),
  - CNN predictions,
  with:
  - LMV/HMV mapping (Light vs Heavy Motor Vehicles).
- Compute:
  - Vehicles per frame,
  - LMV/HMV counts over time,
  - Traffic density measures.
- Visualize:
  - Time series of vehicle counts,
  - LMV vs HMV ratios,
  - Histograms of counts,
  - Frame-level summaries.

### **Step 5 – Image Restoration (Optional Extra Credit) (`05_image_restoration_optional.ipynb`)**
- Demonstrate use of **pretrained restoration models** (e.g., HuggingFace):
  - Super-resolution,
  - Denoising,
  - Colorization.
- Apply models to selected DETRAC frames.
- Compare original vs restored images to discuss possible benefits for detection.

### **Step 6 – Real-Life Deployment Demo (`06_real_life_traffic_video_demo.ipynb`)**
- Use **background subtraction** (MOG2) to detect moving vehicles in an unconstrained video:
  - Input video: `data/real_life/traffic_example.mov`.
- Crop detected regions and classify each using the trained PyTorch model.
- Compute for each frame:
  - Vehicle count,
  - LMV/HMV density level (Low, Medium, High).
- Generate three output videos:
  - **Analysis video** with boxes, labels, and traffic density overlay,
  - **Detections-only video** with bounding boxes,
  - **Foreground mask video** for debugging.
- Run analytics on the processed video:
  - Vehicle count over time,
  - Density distributions,
  - Summary statistics.

### **Step 7 – Experiments (TensorFlow & YOLO) – `experiments/`**
- Implement an experimental **TensorFlow/Keras** classifier.
- Explore **class-imbalance strategies** and LMV/HMV evaluation.
- Prototype a **YOLO + tracking** pipeline for vehicle detection and tracking in video.
- These experiments are *not* part of the final reported pipeline, but are kept for reference.

---

# **4. Methods & Techniques**

### **4.1 Data Extraction & Preprocessing**
- Parse UA-DETRAC XML annotations (bounding boxes, frame indices, class labels).
- Crop bounding boxes from full-size frames.
- Normalize image intensities and resize crops to a consistent size.
- Store all cropped vehicles, labels, and class mappings in a compact NumPy `.npz` file.
- Perform exploratory visualizations:
  - Sample frames with bounding boxes,
  - Histograms of box sizes and counts.

### **4.2 Deep Learning Model (PyTorch CNN)**
- Custom CNN architecture with:
  - Convolutional layers (feature extraction),
  - Max-pooling (downsampling),
  - Fully connected layers (classification).
- Training setup:
  - Loss: Cross-Entropy,
  - Optimizer: Adam,
  - Mini-batch training,
  - Learning rate scheduling (if applicable).
- Evaluation:
  - Accuracy on validation set,
  - Confusion matrix to identify class confusion,
  - Classification report showing precision/recall/F1 per class.

### **4.3 Video Processing & Detection**
- Use OpenCV’s **MOG2 background subtractor**:
  - Identify moving regions in each frame,
  - Apply morphological operations to clean up the foreground mask,
  - Extract bounding boxes around moving objects (candidate vehicles).
- For each bounding box:
  - Crop, resize, and normalize,
  - Classify with the PyTorch model,
  - Draw bounding boxes and labels on the frame.

### **4.4 Traffic Analytics**
- Per-frame metrics:
  - Estimated vehicle count,
  - LMV/HMV classification,
  - Density level (`Low`, `Medium`, `High`).
- Aggregate analytics:
  - Time-series plots of vehicle count,
  - Density over time (numeric encoding and scatter plots),
  - Histograms of vehicle counts per frame,
  - Frame-level density distribution.
- Summary metrics:
  - Average vehicles per frame,
  - Min/Max counts,
  - Duration of video (based on FPS and frame count),
  - Percentage of frames in each density level.

### **4.5 Image Restoration (Optional)**
- Use pretrained models (e.g., from HuggingFace) for:
  - Super-resolution,
  - Denoising,
  - Colorization.
- Evaluate qualitatively how restoration affects visual clarity and potential for better detection.

---

# **5. Tools & Technologies**

| Category                  | Tools / Libraries                                           |
|---------------------------|-------------------------------------------------------------|
| Programming Language      | Python 3.12                                                 |
| Deep Learning (Main)      | PyTorch, TorchVision                                        |
| Deep Learning (Experiment)| TensorFlow, Keras                                           |
| Computer Vision           | OpenCV                                                      |
| Data & Math               | NumPy, Pandas                                               |
| Visualization             | Matplotlib (and optionally Seaborn)                        |
| Dataset                   | UA-DETRAC (images + annotations)                           |
| Image Restoration         | HuggingFace / pretrained restoration models (optional)      |
| Environment               | Virtualenv (or Conda), Jupyter Notebooks                   |
| Experiment (Optional)     | YOLO / `ultralytics` for detection + tracking              |

---

# **6. Repository Structure**

```text
project/
│
├── data/
│   ├── DETRAC-Images/
│   ├── DETRAC-Train-Annotations/
│   ├── DETRAC-Test-Annotations/
│   ├── cropped_vehicle_dataset.npz
│   └── real_life/
│       └── traffic_example.mov
│
├── document/
│   ├── AAI-521-Final-Team-Project-Instructions.docx
│   ├── AAI-521-Final-Team-Project-Proposal.docx
│   └── AAI-521-Final-Team-Project-Final-Report.pdf
│
├── models/
│   └── vehicle_classifier.pth
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_and_cropping.ipynb
│   ├── 03_vehicle_classification_model.ipynb
│   ├── 04_vehicle_counting_and_analysis.ipynb
│   ├── 05_image_restoration_optional.ipynb
│   ├── 06_real_life_traffic_video_demo.ipynb
│   └── 07_experiment_tensorflow_model.ipynb
│
├── outputs/
│   ├── plots/
│   ├── videos/
│   │   ├── traffic_example_analysis.mp4
│   │   ├── traffic_example_detections.mp4
│   │   └── traffic_example_mask.mp4
│   └── annotated_frames/
│
├── src/
│   └── utils_detrac.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# **7. Notebook Summaries**

### **📘 01 – Dataset Exploration (`01_dataset_exploration.ipynb`)**
- Load raw DETRAC sequences and annotations.
- Visualize images with bounding boxes.
- Compute EDA statistics for:
  - Frames,
  - Vehicle counts per frame,
  - Bounding box shapes and sizes.
- Summarize observations for the project report.

---

### **📘 02 – Preprocessing & Cropping (`02_preprocessing_and_cropping.ipynb`)**
- Parse XML annotations and iterate over all frames.
- Crop each labeled vehicle into a separate image patch.
- Normalize and resize crops to the chosen input size (e.g., 64 × 64).
- Save the final dataset as `cropped_vehicle_dataset.npz` with:
  - `images`,
  - `labels`,
  - `class_to_idx` mapping.
- Provide class distribution statistics and sanity-check plots.

---

### **📘 03 – Vehicle Classification Model (PyTorch) (`03_vehicle_classification_model.ipynb`)**
- Load the cropped dataset from `data/cropped_vehicle_dataset.npz`.
- Split into training/validation sets.
- Define the **`VehicleClassifier`** CNN architecture.
- Train the model and track:
  - Training/validation loss over epochs,
  - Training/validation accuracy.
- Evaluate performance:
  - Confusion matrix,
  - Classification report,
  - Examples of correct and incorrect predictions.
- Save the trained model to `models/vehicle_classifier.pth`.

---

### **📘 04 – Vehicle Counting & Analysis (`04_vehicle_counting_and_analysis.ipynb`)**
- Use detections combined with model predictions to:
  - Count vehicles per frame,
  - Map classes into LMV vs HMV.
- Generate:
  - Time-series plots of vehicles per frame,
  - LMV/HMV ratios over time,
  - Histograms of traffic load.
- Provide interpretations relevant for traffic analytics and engineering.

---

### **📘 05 – Image Restoration (Optional) (`05_image_restoration_optional.ipynb`)**
- Apply restoration models (e.g., from HuggingFace) to DETRAC frames:
  - Super-resolution,
  - Denoising,
  - Colorization.
- Compare before/after images.
- Discuss potential impact on detection and classification if time permits.

---

### **📘 06 – Real-Life Traffic Video Demo (`06_real_life_traffic_video_demo.ipynb`)**
- Load a real-life traffic video from `data/real_life/traffic_example.mov`.
- Use background subtraction (MOG2) to detect moving vehicles per frame.
- Crop and classify each detection using the trained `VehicleClassifier`.
- For each frame:
  - Compute vehicle count,
  - Assign density level (Low / Medium / High).
- Save:
  - An analysis video with bounding boxes, labels, and overlays,
  - A detections-only video,
  - A foreground mask video.
- Generate analytics:
  - Vehicle count over time,
  - Density distributions,
  - Summary statistics (average vehicles/frame, min/max, etc.).

---

### **🧪 Experiments – TensorFlow & YOLO (`experiments/tensorflow_experiments.ipynb`)**
- Implement a TensorFlow/Keras version of the classifier (experimental).
- Explore:
  - Class balancing strategies,
  - LMV/HMV mapping in a 2-class confusion matrix.
- Prototype YOLO-based detection and simple tracking on video.
- **Not used** in the main reported pipeline; kept for reference and future work.

---

# **8. How to Run the Project**

1. **Clone or download the project repository**  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a virtual environment**  
   Create a virtual environment to manage dependencies:
   ```bash
   python3.12 -m venv env
   ```

   Activate the environment on macOS/Linux:
   ```bash
   source env/bin/activate
   ```

3. **Install the dependencies**  
   Use the `requirements.txt` file to install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Jupyter notebooks**
   ```bash
   jupyter notebook
   ```

    Then open the notebooks under `notebooks/` in order (01 → 06).

5. **To deactivate the virtual environment, run:**

   ```bash
   deactivate
   ```

### **Real-Life Video Demo**
1. Place a video file named `traffic_example.mov` under:
   ```text
   data/real_life/
   ```
2. Open and run:
   ```text
   notebooks/06_real_life_traffic_video_demo.ipynb
   ```
3. Generated videos will appear under:
   ```text
   outputs/videos/
   ```

---

# **9. Summary & Contributions**

This project demonstrates:

- An **end-to-end computer vision pipeline**:
  - From raw annotated dataset → cropped dataset → trained classifier → video analytics.
- Integration of:
  - Deep learning (PyTorch),
  - Classical computer vision (background subtraction, bounding boxes),
  - Video analytics and visualization.
- Traffic-relevant insights using LMV/HMV aggregation and density metrics.
- Team collaboration across:
  - Data processing,
  - Model implementation,
  - Visualization,
  - Real-world validation.

The work showcases how modern computer vision techniques can be applied to **intelligent transportation systems**, using real datasets and practical deployment scenarios.

---

# **10. References**

Dataset sources:

- UA-DETRAC Project Page  
  https://sites.google.com/view/daweidu/projects/ua-detrac  

- UA-DETRAC on Kaggle  
  https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset  

- [Google Drive Folder](https://drive.google.com/drive/u/1/folders/1XGP1b8UBR3VErb7FYzzjA5CTBIFrq_WD)

Additional references (models, techniques, and libraries) are cited in the final written report and within the code comments where appropriate.