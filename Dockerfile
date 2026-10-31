# Sử dụng Python 3.10 slim để image nhẹ và tối ưu dung lượng
FROM python:3.10-slim

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Cài đặt các thư viện hệ thống cần thiết cho xử lý ảnh và HDF5
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt PyTorch bản CPU trước để tối ưu dung lượng Image (giảm từ 5GB xuống còn ~1.5GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements và cài đặt các phụ thuộc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn, model weight và dữ liệu test vào container
COPY . .

# Mở cổng 7860
EXPOSE 7860

# Health check kiểm tra trạng thái ứng dụng
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Khởi chạy ứng dụng
CMD ["python", "app.py"]

