# --- 1. IMPORT SPACES ĐẦU TIÊN BẮT BUỘC CHO HUGGINGFACE ZEROGPU ---
try:
    import spaces
except Exception:
    class spaces:
        @staticmethod
        def GPU(fn=None, *args, **kwargs):
            if fn is None:
                return lambda f: f
            return fn

import os
import sys
import asyncio

# Tắt telemetry và analytics để tránh crash mạng/lifespan
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Đảm bảo event loop tồn tại trên Python 3.12+
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# UTF-8 console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import traceback

# --- TƯƠNG THÍCH HUGGINGFACE HUB VỚI GRADIO ---
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @classmethod
            def get_token(cls):
                return os.environ.get("HF_TOKEN") or None
            @classmethod
            def save_token(cls, token): pass
            @classmethod
            def delete_token(cls): pass
        huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

# --- TƯƠNG THÍCH GRADIO VỚI PYDANTIC V2 SCHEMA ---
try:
    import gradio_client.utils
    _orig_get_type = gradio_client.utils.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)
    gradio_client.utils.get_type = _safe_get_type

    _orig_json_schema = gradio_client.utils._json_schema_to_python_type
    def _safe_json_schema(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_json_schema(schema, defs)
    gradio_client.utils._json_schema_to_python_type = _safe_json_schema
except Exception:
    pass

import gradio as gr
from gradio.routes import App
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")

# ========================================================
# 1. HÀM DỰ BÁO ZERO-GPU ĐƯỢC ĐĂNG KÝ VỚI GRADIO
# ========================================================
@spaces.GPU(duration=60)
def run_prediction(fire_name: str) -> str:
    """Hàm suy luận ZeroGPU chính thức phục vụ quét đăng ký ZeroGPU."""
    try:
        result = predictor.predict_fire(fire_key=fire_name, days=1)
        area = result.get('area_ha', 0)
        risk = result.get('risk_level', 'Nguy cơ cao')
        name = result.get('fire_name', fire_name)
        return f"✅ ZeroGPU hoàn tất phân tích! Vụ cháy: {name} | Diện tích: {area:.1f} ha | Rủi ro: {risk}"
    except Exception as e:
        return f"❌ Lỗi suy luận: {str(e)}"

FIRE_LIST = [
    "Smith_River_Complex_CA_2023.hdf5",
    "Mosquito_Fire_CA_2022.hdf5",
    "Rabbit_Fire_CA_2023.hdf5",
    "Black_Fire_NM_2022.hdf5",
    "Wildcat_Fire_AZ_2023.hdf5",
    "Bedrock_Fire_OR_2023.hdf5",
    "Calf_Canyon_Fire_NM_2022.hdf5",
    "Hermits_Peak_Fire_NM_2022.hdf5",
    "Lookout_Fire_OR_2023.hdf5",
    "Yosemite_Complex_CA.hdf5",
    "Six_Rivers_Fire_CA.hdf5",
    "Happy_Camp_Complex_CA.hdf5",
    "Colt_Fire_MT.hdf5",
    "Pine_Pond_Fire_TX.hdf5"
]

# ========================================================
# 2. GRADIO INTERFACE (BẮT BUỘC ĐỂ ZEROGPU HOẠT ĐỘNG)
# ========================================================
custom_css = """
body, html {
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    height: 100% !important;
    width: 100% !important;
    background-color: #0b0f19 !important;
}
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    height: 100vh !important;
    width: 100vw !important;
}
footer {
    display: none !important;
}
"""

with gr.Blocks(title="🔥 Wildfire AI - Dự báo Cháy Rừng", fill_width=True, css=custom_css) as demo:
    # Ẩn toàn bộ widget mặc định này để không làm rối mắt
    with gr.Column(visible=False):
        fire_dropdown = gr.Dropdown(choices=FIRE_LIST, value="Smith_River_Complex_CA_2023.hdf5")
        status_out = gr.Textbox()
        run_btn = gr.Button("GPU Hook")
        run_btn.click(fn=run_prediction, inputs=fire_dropdown, outputs=status_out)

    # Toàn bộ màn hình hiển thị trực tiếp Web GIS Leaflet Command Center
    gr.HTML('<iframe src="/map" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; border: none; z-index: 999999; margin: 0; padding: 0;"></iframe>')

# ========================================================
# 3. GẮN CÁC ROUTE FASTAPI & STATIC FILES VÀO GRADIO APP
# ========================================================
_orig_create_app = App.create_app

def _custom_create_app(*args, **kwargs):
    app = _orig_create_app(*args, **kwargs)
    
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    @app.head("/health")
    async def health_check():
        return JSONResponse(content={"status": "healthy"})

    @app.get("/map", response_class=HTMLResponse)
    @app.get("/app", response_class=HTMLResponse)
    async def serve_index():
        if os.path.exists(INDEX_HTML_PATH):
            with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>Wildfire AI Command Center</h1>")

    @app.get("/download/sample_tif")
    async def download_sample_tif():
        p = os.path.join(STATIC_DIR, "sample_fire_smith_river.tif")
        if os.path.exists(p):
            return FileResponse(p, filename="sample_fire_smith_river.tif", media_type="image/tiff")
        return JSONResponse({"error": "Sample file not found"}, status_code=404)

    @app.get("/download/sample_h5")
    async def download_sample_h5():
        p = os.path.join(BASE_DIR, "test_data", "Smith_River_Complex_CA_2023.hdf5")
        if os.path.exists(p):
            return FileResponse(p, filename="Smith_River_Complex_CA_2023.hdf5", media_type="application/x-hdf5")
        return JSONResponse({"error": "Sample file not found"}, status_code=404)

    @app.get("/wf_api/initial_data")
    @app.get("/api/initial_data")
    async def api_initial_data():
        try:
            result = predictor.predict_fire("Smith_River_Complex_CA_2023.hdf5", days=1)
            return JSONResponse(content=result)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.post("/wf_api/predict")
    async def api_wf_predict(request: Request):
        try:
            body = await request.json()
            fire_key = body.get("fire_key", "Smith_River_Complex_CA_2023.hdf5")
            days = int(body.get("days", 5))
            date_str = body.get("date", None)
            result = predictor.predict_fire(fire_key=fire_key, days=days, date_str=date_str)
            return JSONResponse(content=result)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(content={"success": False, "error": str(e)})

    @app.post("/wf_api/upload")
    @app.post("/wf_api/upload_and_predict")
    async def api_wf_upload(
        file: UploadFile = File(...),
        days: int = Form(5),
        date: str = Form(None)
    ):
        try:
            contents = await file.read()
            safe_name = os.path.basename(file.filename)
            temp_p = os.path.join(STATIC_DIR, f"upload_{safe_name}")
            with open(temp_p, "wb") as f:
                f.write(contents)
            
            result = predictor.predict_fire(fire_key=temp_p, days=days, date_str=date)
            disp_name = safe_name
            for ext in [".hdf5", ".h5", ".tiff", ".tif"]:
                disp_name = disp_name.replace(ext, "")
            result["name"] = disp_name.replace("_", " ")
            result["file"] = safe_name
            return JSONResponse(content=result)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(content={"success": False, "error": f"Lỗi đọc tệp tải lên: {str(e)}"})

    return app

App.create_app = _custom_create_app

# ========================================================
# 4. KHỞI CHẠY BẰNG DEMO.LAUNCH (CHUẨN ZEROGPU HOOK)
# ========================================================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
