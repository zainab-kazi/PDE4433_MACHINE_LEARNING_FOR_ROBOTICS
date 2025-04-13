# 📘 README  
## Real-Time Emotion Detection with Face Tracking Using Machine Learning

### 👩‍🎓 Student Information  
- **Name:** Zainab Mohammed Akil Kazi  
- **MISIS:** M01044738  
- **Course:** MSc, Robotics  
- **Module:** PDE4433 – Machine Learning For Robotics  
- **Professor:** Ms. Maha Saadeh

---

## 📌 Project Overview  
This project presents a real-time emotion detection system integrated with face tracking. It is designed for human-robot interaction (HRI), enabling robots to recognize human facial expressions and respond with appropriate actions. The system uses machine learning models (CNNs) and computer vision techniques for real-time analysis.

---

## Prerequisites  
- Python 3.11+  
- Git  
- pip (Python package manager)  

---

## 🛠 Installation Steps  

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/emotion-detection.git
   cd emotion-detection
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Dataset Preparation  
- **Dataset Used:** FER2013 ([Download Here](https://www.kaggle.com/datasets/msambare/fer2013))  
- **Preprocessing Steps:**  
  - Convert images to grayscale  
  - Align and crop face regions  
  - Apply data augmentation techniques such as rotation, flipping, and brightness adjustment  

---

## Running the Notebook  

1. Open the Jupyter Notebook  
   ```bash
   jupyter notebook emotion_detection.ipynb
   ```
2. The notebook will perform the following tasks:  
   - Load and preprocess the dataset  
   - Split the data into training and testing sets  
   - Train a Convolutional Neural Network (CNN) model  
   - Evaluate the model's performance  
   - Generate predictions and visualizations  

---

## 🧠 Model Architecture  

- **Base Model:** Convolutional Neural Network (CNN)  
- **Layers:**  
  - Convolution + Pooling  
  - Fully Connected (Dense)  
  - Softmax activation for emotion categories  

---

## 🎥 Face Tracking  

- **Tools Used:**  
  - Haar Cascades (OpenCV)  
  - Dlib Face Detector  
  - Kalman Filter for tracking stability  
  - Multi-object Tracking (MOT)  

---

## 🎯 Emotion Categories  

- Neutral  
- Happy  
- Surprise  
- Fear  
- Sad  

---

## ⚠️ Challenges & Solutions  

| Challenge | Solution |
|----------|----------|
| Varying lighting conditions | Adaptive histogram equalization |
| Multiple face detection | Kalman Filter with MOT support |
| Embedded performance | Lightweight models and gray-scale input |

---

## 📈 Results  

- Successfully achieved real-time emotion detection and face tracking.  
- Verified facial expression categories through a live demo.  
- Robust to lighting and multi-user interaction.  

---

## 🔮 Future Improvements  

- Deploy on Nao / Pepper robots  
- Accelerate inference using Jetson Nano / Google Coral  
- Integrate NLP for voice + facial emotion understanding  
- Enable robotic gestures/tone changes based on emotion  

---

## 🎬 Demo Video  

**[Insert Link to Your Demo Video Here]**

---

## 📚 References  

1. Goodfellow, Bengio, Courville – *Deep Learning*. [Link](https://www.deeplearningbook.org/)  
2. Viola & Jones – Rapid Object Detection. [Link](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)  
3. M. Sambare – FER2013 Dataset. [Link](https://www.kaggle.com/datasets/msambare/fer2013)  
4. Dlib – Face Detector. [Link](http://dlib.net/face_detector.py.html)  
5. Serengil & Ozpinar – DeepFace. [Link](https://github.com/serengil/deepface)  
6. OpenCV – Meanshift & Camshift. [Link](https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html)  
7. Tariq et al. – Real-time Facial Expression Recognition. [IEEE Link](https://ieeexplore.ieee.org/document/8470105)  
8. Li & Deng – Deep Facial Expression Recognition. [Link](https://www.sciencedirect.com/science/article/pii/S1877050920313390)  
9. Soleymani et al. – Multimodal Database for Affect Recognition. [Springer Link](https://link.springer.com/article/10.1007/s00138-021-01208-3)  
