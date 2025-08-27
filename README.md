# 🖼️ Object Detection

This project demonstrates object detection using YOLOv8.
It supports detecting objects from images, videos, and live webcam feeds.

## ✍️ Note Running with CPU first it's very slow

## 📸 Screenshots

![Screenshots](https://github.com/user-attachments/assets/f0475078-2123-4ccd-8238-d7e2eb601a7d)

## 🛠️ Installation :

1️⃣ Clone the repository:

```bash
git clone https://github.com/SA1946/Object_Detection.git
cd Object_Detection
```

2️⃣ Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

```

3️⃣ Install dependencies:

```bash
pip install -r requirements.txt

```

4️⃣ 🚀 Usage

```bash
python yolo_test.py    #test image first
-> python yolo_webcam.py

```


## ✍️ Note Running with GPU
- install CUDA and CUDNN -> [CUDA](https://developer.nvidia.com/cuda-toolkit) , [CUDNN](https://developer.nvidia.com/cudnn-downloads) if u don't have
- copy all *bin*, *include*, *lib(x64)* from CUDNN into CUDA
- pip uninstall **torch** , **torchvision**
- use **pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126** from [pytorch](https://pytorch.org/get-started/locally/)
- the size to install over 2GB
- after **python yolo_webcam.py**



## 💻 Tech Stack

- **Python**


## 📄 License

<div align='center'>

Licensed under the © [Synsaa](https://github.com/SA1946/Object_Detection/blob/main/LICENSE)

![Screenshot](https://camo.githubusercontent.com/ff1d4eb768b74fa335491dd8a7e87d95017665c1570e5a8828fddfdb728da450/68747470733a2f2f63617073756c652d72656e6465722e76657263656c2e6170702f6170693f747970653d776176696e6726636f6c6f723d6772616469656e74266865696768743d3130302673656374696f6e3d666f6f746572)

</div>