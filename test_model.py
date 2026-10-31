import os
import pytest
import torch

def test_model_weight_file_exists():
    """Kiểm tra file trọng số mô hình đã huấn luyện có tồn tại không"""
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    assert os.path.exists(model_path), f"Không tìm thấy file trọng số mô hình tại {model_path}"
    assert os.path.getsize(model_path) > 10 * 1024 * 1024, "File model quá nhỏ hoặc bị lỗi"

def test_predictor_model_load():
    """Kiểm tra module predictor và mô hình Res18UTAEModel nạp thành công"""
    import predictor
    assert predictor.model is not None, "Mô hình chưa được khởi tạo"
    assert hasattr(predictor, "predict_fire"), "Module predictor thiếu hàm predict_fire"

def test_model_inference_sample():
    """Kiểm tra hàm suy luận (inference) của mô hình với tệp dữ liệu test có sẵn"""
    import predictor
    sample_file = "Smith_River_Complex_CA_2023.hdf5"
    result = predictor.predict_fire(fire_key=sample_file, days=1)
    assert result is not None, "Kết quả trả về rỗng"
    assert result.get("success") is True, f"Inference thất bại với lỗi: {result.get('error')}"
    assert "metrics" in result, "Thiếu metrics trong kết quả trả về"
    assert "pred_image" in result, "Thiếu ảnh dự đoán trong kết quả trả về"
