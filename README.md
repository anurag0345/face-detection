# Real-Time Face Detection using OpenCV

This project demonstrates **real-time face detection** using the **Haar Cascade Classifier** provided by OpenCV. It captures video from your webcam and draws rectangles around detected faces in each frame.

---

## 🧠 Features

- Detects human faces in real time using the system’s webcam.  
- Uses OpenCV’s pre-trained `haarcascade_frontalface_default.xml` model.  
- Displays a live video stream with bounding boxes around detected faces.  
- Press **`q`** to exit the video window.

---

## 📁 Project Structure

```
.
├── face_detection.py     # Main Python script
├── README.md             # Project documentation
```

---

## ⚙️ Requirements

Before running the project, ensure you have the following installed:

- Python 3.7 or higher  
- OpenCV library (`cv2`)  

Install the dependencies using pip:

```bash
pip install opencv-python
```

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/face-detection-opencv.git
   cd face-detection-opencv
   ```

2. **Run the script**
   ```bash
   python face_detection.py
   ```

   > The webcam will start automatically. Press `q` to close the window.

---

## 🧩 How It Works

1. Loads the **Haar Cascade XML** model for face detection:
   ```python
   cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
   ```
2. Captures live video using:
   ```python
   video_capture = cv2.VideoCapture(0)
   ```
3. Converts each frame to grayscale (for better accuracy) and detects faces:
   ```python
   gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
   ```
4. Draws green rectangles around detected faces and displays them in a window.

---

## 🧰 Optional: Command Line Arguments

The script includes an argument parser structure (currently not fully implemented) that can be extended to add:
- Input image path (`--image`)
- Video path or model directory (`--video`)

Example:
```bash
python face_detection.py --image path/to/image.jpg
```

---

## 💡 Future Improvements

- Add support for DNN-based face detection models (e.g., OpenCV DNN or Mediapipe).  
- Enable saving detected face snapshots.  
- Allow custom video input instead of webcam.  
- Integrate performance optimization for low-end systems.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
