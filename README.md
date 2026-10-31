# 🔥 WildfireSpreadTS - AI Wildfire Spread Prediction System

Hệ thống AI dự báo mức độ lan truyền của cháy rừng theo chuỗi thời gian dựa trên mô hình học sâu **Res18-UTAE** (Spatio-Temporal Attention).

---

## 🚀 Các tính năng chính
- **Mô hình AI**: Res18-UTAE kết hợp LTAE (Lightweight Temporal Attention Encoder) phân tích dữ liệu viễn thám và chuỗi thời gian cháy rừng.
- **Inference Service**: FastAPI & Gradio Web Server cung cấp API dự đoán và bảng điều khiển bản đồ GIS tương tác trực quan.
- **Containerization**: Đóng gói hoàn chỉnh với **Docker**.
- **Tự động hóa CI/CD**: Sử dụng **GitHub Actions** để tự động kiểm thử (Unit test) và tự động build/push Docker Image lên **Docker Hub**.

---

## 🛠️ Triển khai nhanh với Docker

### 1. Kéo Image từ Docker Hub
```bash
docker pull <DOCKERHUB_USERNAME>/wildfire-cicd:latest
```

### 2. Khởi chạy Container
```bash
docker run -d -p 7860:7860 --name wildfire-app <DOCKERHUB_USERNAME>/wildfire-cicd:latest
```

Truy cập hệ thống tại: `http://localhost:7860`

---

## 🧪 Chạy Kiểm thử Cục bộ (Unit Test)
```bash
pip install -r requirements.txt pytest
pytest test_model.py -v
```

---

## 🔄 Cấu hình CI/CD Pipeline (GitHub Actions)
Pipeline được định nghĩa trong `.github/workflows/ci-cd.yml` gồm 2 giai đoạn:
1. **CI (Continuous Integration)**: Chạy trên Ubuntu runner, cài đặt dependencies và kiểm tra tính toàn vẹn của mô hình qua `test_model.py`.
2. **CD (Continuous Delivery)**: Sau khi CI thành công, tự động build multi-tag image (`latest`, `${{ github.sha }}`) và đẩy lên **Docker Hub**.
