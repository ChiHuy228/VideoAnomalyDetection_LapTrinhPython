## THÔNG TIN THÀNH VIÊN NHÓM 1_DH23TIN05
* Lê Hoàng Danh - 235087 - EVTMIL
* Phan Quốc Duy - 235019 - MIL
* Trần Anh Đoàn - 234818 - VadCLIP
* Tô Chí Huy - 234662 - RTFM
* Phạm Trí Khang - 235187 - URDMU
* Đặng Thị Yến Nhi - 235187 - VadCLIP

** Đường dẫn dataset: https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=2&dl=0


## 🚀 Hướng dẫn Cài đặt Trọng số Mô hình (Model Weights)

Vì giới hạn dung lượng của GitHub, thư mục `model` không được đẩy lên kho lưu trữ này. Để hệ thống hoạt động, bạn vui lòng làm theo các bước sau:


#1. Tải trọng số từ Google Drive:**
* Truy cập link Google Drive sau:https://drive.google.com/file/d/1UDR8skSQ9ut4KP8AYnMBRNJR4pzghAFE/view?usp=sharing
* Tải toàn bộ thư mục `model` về máy.

#2. Đặt vào đúng vị trí:**
Giải nén và copy thư mục `model` vừa tải đặt vào bên trong thư mục `streamlit_demo/`. Cấu trúc cây thư mục phải trông như sau:

```text
VideoAnomalyDetection_LapTrinhPython/
│
├── streamlit_demo/
│   ├── app.py
│   ├── model/                  <-- (Đặt thư mục model tại đây)
│   │   ├── evtmil/
│   │   ├── mil_sultani/
│   │   ├── rtfm/
│   │   ├── urdmu/
│   │   └── vadclip/
│   ├── temp_features/
│   ├── temp_scores/
│   └── temp_videos/
...
# 1. Tạo và kích hoạt môi trường ảo conda với Python 3.10
conda create -n video_anomaly python=3.10 -y
conda activate video_anomaly

# 2. Cài đặt PyTorch với CUDA 11.8 (Tối ưu cho GPU NVIDIA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Cài đặt các thư viện phụ thuộc từ file requirements.txt
pip install -r requirements.txt

#Khởi chạy trang web
streamlit run streamlit_demo/app.py
