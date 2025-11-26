# Final Project - AAI521 - Computer Vision - Group 3


Shiley-Marcos School of Engineering, University of San Diego AAI-521: Computer Vision

## Group members

- Birendra Khimding
- Matt Hashemi
- Victor Hugo Germano


## Files

[Google Drive Folder](https://drive.google.com/drive/u/1/folders/1XGP1b8UBR3VErb7FYzzjA5CTBIFrq_WD)


## Objective: Vehicle Counting and Classification Using Computer Vision

This project focuses on developing a computer vision system capable of detecting, classifying, and counting different types of vehicles on the road. Using deep learning–based object detection, the system identifies vehicles in video footage or image sequences and classifies them into categories such as:

- LMV (Light Motor Vehicle): cars, sedans, hatchbacks, SUVs, vans
- HMV (Heavy Motor Vehicle): buses, trucks, lorries, construction vehicles
 

### Project Requirements

For this project, you will use Python in Google Colab, write a technical report, and prepare a recorded video presentation, including visuals based on your report.

#### Project Deliverables:
- Technical Report and Appendix  
- Code Base (GitHub repo)
- Video Presentation

#### Project Timeline:
By Day 7 (end of the module week): 

- Module 1, you will complete a Teammate Survey to inform your instructor of your group preferences. 
- Module 2, you will be assigned a group of two to three members. Connect with your team members via Canvas, USD Email, or Slack. Research a dataset and project idea to introduce to your teammates. 
- Module 4, your team will complete the Team Project Status Update Form and describe the dataset(s) your team has chosen. 
- Module 7, your team will finalize and submit the three deliverables for the final project. 
- Module 7, you will complete a Peer Review for each team member and submit in an individual assignment in Canvas.   


---

## References

download dataset:
- https://sites.google.com/view/daweidu/projects/ua-detrac?authuser=0
- https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset?resource=download

---

## Technical Details

### Setup Instructions

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

4. **To deactivate the virtual environment, run:**

   ```bash
   deactivate
   ```

---
### Final Project Structure

```
project/
│
├── data/
│   ├── DETRAC-Images/
│   ├── DETRAC-Train-Annotations/
│   └── DETRAC-Test-Annotations/
│
├── document/
│   ├── AAI-521-Final-Team-Project-Instructions.docx
│   ├── AAI-521-Final-Team-Project-Proposal.docx
│   └── AAI-521-Final-Team-Project-Final-Report.pdf
│
├── env/
│
├── models/
│   └── vehicle_classifier.pth
|
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_and_cropping.ipynb
│   ├── 03_vehicle_classification_model.ipynb
│   ├── 04_vehicle_counting_and_analysis.ipynb
│   └── 05_image_restoration_optional.ipynb
│
├── outputs/
│   ├── plots/
│   ├── videos/
|   ├── cropped_vehicle_dataset.npz
│   ├── annotated_frames/
│   └── annotated_videos/
│
├── src/
│   └── utils_detrac.py
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

### Project Notebook Structure

#### Notebook 1 — `01_dataset_exploration.ipynb`
**Input:** Raw DETRAC images + XML
**Output:**
- EDA stats
- Plots
- Sample frames
- Insight summary for report

#### Notebook 2 — `02_preprocessing_and_cropping.ipynb`
**Input:** Raw data
**Output:**
- Cropped vehicle dataset (numpy or torch tensors)
- Saved file: `cropped_dataset.npz` or `.pt`
- Dataset statistics

#### Notebook 3 — `03_vehicle_classification_model.ipynb`
**Input:** Cropped dataset
**Output:**
- Model training
- Saved model: `vehicle_classifier.pth`

**Metrics:**
- Accuracy
- Loss curves
- Confusion matrix
- Classification report

**Visualizations:**
- Sample predictions
- Annotated frames
- Annotated video (optional)

#### Notebook 4 — `04_vehicle_counting_and_analysis.ipynb`
**Performs:**
- Per-frame vehicle counting
- LMV vs HMV time plot
- Useful visualizations for traffic analysis

#### Notebook 5 — `05_image_restoration_optional.ipynb` *(Optional - Bonus for Extra Credit)*
**Use HuggingFace pretrained models to show:**
- Super-resolution
- Denoising
- Colorization
