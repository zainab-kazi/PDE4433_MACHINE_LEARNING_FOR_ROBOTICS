
# 📘 README  
## Real-Time Emotion Detection with Face Tracking Using Machine Learning

### 👩‍🎓 Student Information  
- **Name:** Zainab Mohammed Akil Kazi  
- **MISIS:** M01044738  
- **Course:** MSc, Robotics  
- **Module:** PDE4433 – ML Robotic System Design  
- **Professor:** Ms. Maha Saadeh

---

## 📌 Project Overview  
This project presents a real-time emotion detection system integrated with face tracking. It is designed for human-robot interaction (HRI), enabling robots to recognize human facial expressions and respond with appropriate actions. The system uses machine learning models (CNNs) and computer vision techniques for real-time analysis.

---

## 🤖 Applications in Robotics  
- **Retail & Hospitality:** Adapts responses based on customer emotions.  
- **Healthcare:** Monitors patients’ mental well-being.  
- **Education:** Adjusts lesson delivery based on student engagement.  
- **Security:** Detects suspicious or aggressive behavior.  
- **Elderly Care:** Provides emotional support and monitoring.

---

## 📂 Dataset Information  
### Datasets Used:
- **FER2013:** [Facial Expression Recognition 2013](https://www.kaggle.com/datasets/msambare/fer2013)  
- **KDEF:** [Karolinska Directed Emotional Faces](https://www.kdef.se/)

### Preprocessing:
- Grayscale conversion  
- Face alignment & cropping  
- Data augmentation (rotation, flipping, brightness adjustment)

---

## 🧠 Model Architecture  
- **Base Model:** Convolutional Neural Network (CNN)  
- **Layers:**  
  - Convolution + Pooling  
  - Fully Connected (Dense)  
  - Softmax activation for emotion categories  

---

## 🎯 Emotion Categories  
- Neutral  
- Happy  
- Surprise  
- Fear  
- Sad

---

## 🎥 Face Tracking  
- **Tools:**  
  - Haar Cascades (OpenCV)  
  - Dlib Face Detector  
  - Kalman Filter for tracking stability  
  - Multi-object Tracking (MOT)

---

## 🛠 Hardware & Software  
### Hardware:  
- **Logitech C270 HD Webcam**

### Software & Libraries:  
- Python  
- OpenCV  
- TensorFlow / Keras  
- DeepFace  
- Scikit-learn / Scikit-image  
- Graphviz, Pydot, Matplotlib, Scipy

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
4. Karolinska Institute – KDEF Dataset. [Link](https://www.kdef.se/)  
5. Dlib – Face Detector. [Link](http://dlib.net/face_detector.py.html)  
6. Serengil & Ozpinar – DeepFace. [Link](https://github.com/serengil/deepface)  
7. OpenCV – Meanshift & Camshift. [Link](https://docs.opencv.org/4.x/d7/d00/tutorial_meanshift.html)  
8. Tariq et al. – Real-time Facial Expression Recognition. [IEEE Link](https://ieeexplore.ieee.org/document/8470105)  
9. Li & Deng – Deep Facial Expression Recognition. [Link](https://www.sciencedirect.com/science/article/pii/S1877050920313390)  
10. Soleymani et al. – Multimodal Database for Affect Recognition. [Springer Link](https://link.springer.com/article/10.1007/s00138-021-01208-3)
