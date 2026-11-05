## 🚀 Hướng dẫn Cài đặt Trọng số Mô hình (Model Weights)

Vì giới hạn dung lượng của GitHub, thư mục `model` không được đẩy lên kho lưu trữ này. Để hệ thống hoạt động, bạn vui lòng làm theo các bước sau:

**1. Tải trọng số từ Google Drive:**
* Truy cập link Google Drive sau: `[https://drive.google.com/file/d/1UDR8skSQ9ut4KP8AYnMBRNJR4pzghAFE/view?usp=sharing]`
* Tải toàn bộ thư mục `model` về máy.

**2. Đặt vào đúng vị trí:**
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