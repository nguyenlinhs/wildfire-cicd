/**
 * ========================================================
 * WILDFIRE AI - STREAMLINED LOGIC
 * Tối giản, tập trung vào bản đồ và dự báo
 * ========================================================
 */

// Global State
let selectedFire = "Smith_River_Complex_CA_2023.hdf5";
let uploadedFile = null;
let currentData = null;
let isPredicted = false;

// Map & Layer References
let map = null;
let inputLayer = null;
let predLayer = null;
let epicenterMarker = null;
let currentBounds = null;

// High-Res Esri Satellite Layer
const SATELLITE_TILE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const SATELLITE_ATTR = "Tiles &copy; Esri &mdash; VIIRS / MODIS Fire Response";

// Preset Fire Historical Dates
const FIRE_DATES = {
    "Smith_River_Complex_CA_2023.hdf5": "20-08",
    "Mosquito_Fire_CA_2022.hdf5": "10-09",
    "Rabbit_Fire_CA_2023.hdf5": "15-07",
    "Black_Fire_NM_2022.hdf5": "18-05",
    "Wildcat_Fire_AZ_2023.hdf5": "02-07",
    "Bedrock_Fire_OR_2023.hdf5": "28-07",
    "Calf_Canyon_Fire_NM_2022.hdf5": "16-04",
    "Hermits_Peak_Fire_NM_2022.hdf5": "20-04",
    "Lookout_Fire_OR_2023.hdf5": "15-08",
    "Yosemite_Complex_CA.hdf5": "06-10",
    "Six_Rivers_Fire_CA.hdf5": "13-09",
    "Happy_Camp_Complex_CA.hdf5": "12-09",
    "Colt_Fire_MT.hdf5": "17-08",
    "Pine_Pond_Fire_TX.hdf5": "28-09"
};

// ========================================================
// 1. INITIALIZATION ON DOM READY
// ========================================================
document.addEventListener("DOMContentLoaded", async () => {
    initMap();
    bindEvents();
    await loadInitialData();
});

function initMap() {
    const defaultCenter = [41.788, -123.788];
    const defaultZoom = 12;

    map = L.map('map', {
        center: defaultCenter,
        zoom: defaultZoom,
        zoomControl: true,
        attributionControl: false
    });

    L.tileLayer(SATELLITE_TILE, {
        maxZoom: 18,
        attribution: SATELLITE_ATTR
    }).addTo(map);

    const pulseIcon = L.divIcon({
        className: 'hotspot-pulse-marker',
        iconSize: [18, 18],
        iconAnchor: [9, 9]
    });

    epicenterMarker = L.marker(defaultCenter, { icon: pulseIcon }).addTo(map);
}

// ========================================================
// 2. DATA LOADING & UI SYNC
// ========================================================
async function loadInitialData() {
    try {
        const res = await fetch('/wf_api/initial_data');
        if (res.ok) {
            const data = await res.json();
            if (data && data.success) {
                updateUI(data, false);
                return;
            }
        }
    } catch (e) {
        console.warn("Could not fetch initial data:", e);
    }
    await fetchFireFootprint(selectedFire, false);
}

async function fetchFireFootprint(fireKey, andPredict = false) {
    try {
        const targetDate = FIRE_DATES[fireKey] || "20-08";
        const res = await fetch('/wf_api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                fire_key: fireKey,
                days: 5,
                date: targetDate
            })
        });
        const data = await res.json();
        updateUI(data, andPredict);
    } catch (err) {
        console.error("Lỗi khi tải dữ liệu đám cháy:", err);
    }
}

function updateUI(data, isPredictedTriggered = false) {
    if (!data || !data.success) {
        const errMsg = data && data.error ? data.error : (data && data.detail ? JSON.stringify(data.detail) : "Không có phản hồi từ máy chủ");
        alert("Lỗi khi tải dữ liệu dự báo: " + errMsg);
        return;
    }

    currentData = data;
    isPredicted = isPredictedTriggered;
    const lat = data.lat || 41.788;
    const lng = data.lng || -123.788;
    const delta = 0.075;
    currentBounds = data.bounds || [[lat - delta, lng - delta], [lat + delta, lng + delta]];

    const initHa = (data.area_init || 0).toLocaleString();
    const predHa = (data.area_pred || 0).toLocaleString();

    // 1. Header Summary Stats
    const headerInitEl = document.getElementById('header-init-area');
    const headerPredEl = document.getElementById('header-pred-area');
    const headerPctEl = document.getElementById('header-pct-change');
    const headerBadgeEl = document.getElementById('header-risk-badge');

    if (headerInitEl) headerInitEl.innerText = `${initHa} ha`;

    if (isPredictedTriggered) {
        if (headerPredEl) {
            headerPredEl.innerText = `${predHa} ha`;
            headerPredEl.style.color = "#ff6b00";
            headerPredEl.style.fontWeight = "700";
        }
        if (headerPctEl) headerPctEl.innerText = `(${data.pct_change || "+0%"})`;
        if (headerBadgeEl) {
            headerBadgeEl.innerText = data.risk_badge || "KHẨN CẤP";
            headerBadgeEl.style.backgroundColor = data.risk_color || "#ef4444";
        }
    } else {
        if (headerPredEl) {
            headerPredEl.innerText = "Chờ dự báo...";
            headerPredEl.style.color = "#94a3b8";
            headerPredEl.style.fontWeight = "500";
        }
        if (headerPctEl) headerPctEl.innerText = "";
        if (headerBadgeEl) {
            headerBadgeEl.innerText = "CHỜ DỰ BÁO";
            headerBadgeEl.style.backgroundColor = "#475569";
        }
    }

    // 2. Map Floating Card
    const nameEl = document.getElementById('info-fire-name');
    const subEl = document.getElementById('info-coords-date');
    const riskEl = document.getElementById('info-risk-level');
    const statusEl = document.getElementById('info-status-text');

    if (nameEl) nameEl.innerText = data.name || "Đám cháy thực tế";
    if (subEl) subEl.innerText = `📍 ${data.coords || `${lat}° N, ${lng}° W`} • Quan sát vệ tinh gần nhất`;

    if (riskEl) {
        if (isPredictedTriggered) {
            riskEl.innerText = data.risk_badge || "Khẩn cấp";
            riskEl.style.color = data.risk_color || "#ef4444";
        } else {
            riskEl.innerText = "Chờ dự báo";
            riskEl.style.color = "#94a3b8";
        }
    }
    if (statusEl) {
        statusEl.innerText = isPredictedTriggered ? "Đã dự báo lan truyền" : "Sẵn sàng dự báo";
    }

    // 3. Map View & Marker
    map.setView([lat, lng], 12);
    epicenterMarker.setLatLng([lat, lng]);

    const opacitySlider = document.getElementById('slider-opacity');
    const currentOpacity = opacitySlider ? parseFloat(opacitySlider.value) : 0.85;

    // Layer 1: Vết cháy hiện tại
    if (inputLayer) map.removeLayer(inputLayer);
    const inputSrc = data.leaflet_input || data.input_img || data.img_t5;
    if (inputSrc) {
        inputLayer = L.imageOverlay(inputSrc, currentBounds, { opacity: currentOpacity });
        if (document.getElementById('chk-layer-init')?.checked) {
            inputLayer.addTo(map);
        }
    }

    // Layer 2: AI Dự báo lan truyền (CHỈ HIỂN THỊ KHI BẮT ĐẦU CHẠY DỰ BÁO)
    if (predLayer) map.removeLayer(predLayer);
    const predSrc = data.leaflet_pred || data.pred_img || data.img_pred;
    const chkPred = document.getElementById('chk-layer-pred');
    if (predSrc) {
        predLayer = L.imageOverlay(predSrc, currentBounds, { opacity: currentOpacity });
        if (isPredictedTriggered) {
            if (chkPred) chkPred.checked = true;
            predLayer.addTo(map);
        } else {
            if (chkPred) chkPred.checked = false;
        }
    }
}

// ========================================================
// 3. EVENT LISTENERS
// ========================================================
function bindEvents() {
    // 1. Fire Selection Dropdown
    const select = document.getElementById('select-fire');
    if (select) {
        select.addEventListener('change', async (e) => {
            selectedFire = e.target.value;
            uploadedFile = null;
            currentData = null;
            const uploadText = document.getElementById('upload-status-text');
            if (uploadText) uploadText.innerText = "";
            if (fileInput) fileInput.value = "";
            // Hiển thị vết cháy hiện tại của đám cháy mới, CHƯA hiện dự báo lan truyền
            await fetchFireFootprint(selectedFire, false);
        });
    }

    // 2. File Upload Button
    const uploadTrigger = document.getElementById('btn-upload-trigger');
    const fileInput = document.getElementById('file-input');
    if (uploadTrigger && fileInput) {
        uploadTrigger.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', async (e) => {
            if (e.target.files && e.target.files.length > 0) {
                uploadedFile = e.target.files[0];
                const uploadText = document.getElementById('upload-status-text');
                if (uploadText) uploadText.innerText = `⏳ Đang đọc tệp ${uploadedFile.name}...`;

                try {
                    const formData = new FormData();
                    formData.append('file', uploadedFile);
                    formData.append('days', 5);

                    const res = await fetch('/wf_api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    if (data && data.success) {
                        currentData = data;
                        if (uploadText) uploadText.innerText = `📄 ${uploadedFile.name}`;
                        // CHỈ HIỂN THỊ VẾT CHÁY HIỆN TẠI, CHƯA HIỂN THỊ LAN TRUYỀN DỰ BÁO
                        updateUI(data, false);
                    } else {
                        if (uploadText) uploadText.innerText = "❌ Lỗi đọc tệp";
                        alert("Lỗi khi đọc tệp: " + (data ? data.error : "Không thể xử lý dữ liệu"));
                    }
                } catch (err) {
                    if (uploadText) uploadText.innerText = "❌ Lỗi tải lên";
                    alert("Lỗi kết nối khi tải tệp: " + err);
                }
            }
        });
    }

    // 3. Main Predict Button
    const btnPredict = document.getElementById('btn-predict');
    if (btnPredict) {
        btnPredict.addEventListener('click', executePredict);
    }

    // 4. Layer Checkboxes
    const chkInit = document.getElementById('chk-layer-init');
    const chkPred = document.getElementById('chk-layer-pred');

    if (chkInit) {
        chkInit.addEventListener('change', (e) => {
            if (!inputLayer) return;
            if (e.target.checked) inputLayer.addTo(map);
            else map.removeLayer(inputLayer);
        });
    }

    if (chkPred) {
        chkPred.addEventListener('change', (e) => {
            if (!predLayer) return;
            if (e.target.checked) predLayer.addTo(map);
            else map.removeLayer(predLayer);
        });
    }

    // 5. Opacity Slider
    const opacitySlider = document.getElementById('slider-opacity');
    const opacityVal = document.getElementById('opacity-val');
    if (opacitySlider) {
        opacitySlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            if (opacityVal) opacityVal.innerText = `${Math.round(val * 100)}%`;
            if (inputLayer) inputLayer.setOpacity(val);
            if (predLayer) predLayer.setOpacity(val);
        });
    }
}

// ========================================================
// 4. API PREDICTION EXECUTION
// ========================================================
async function executePredict() {
    const btn = document.getElementById('btn-predict');
    if (!btn) return;

    const originalText = btn.innerHTML;
    btn.innerHTML = `<span>⏳ Đang tính toán lan truyền...</span>`;
    btn.disabled = true;

    try {
        if (currentData) {
            // Đã có dữ liệu đám cháy (từ tải tệp hoặc chọn sẵn)
            // Mô phỏng AI tính toán và KÍCH HOẠT HIỂN THỊ DỰ BÁO LAN TRUYỀN (true)
            await new Promise(r => setTimeout(r, 600));
            updateUI(currentData, true);
        } else {
            const targetDate = FIRE_DATES[selectedFire] || "20-08";
            const res = await fetch('/wf_api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fire_key: selectedFire,
                    days: 5,
                    date: targetDate
                })
            });
            const data = await res.json();
            updateUI(data, true);
        }
    } catch (err) {
        console.error("Lỗi khi chạy dự báo:", err);
        alert("Lỗi kết nối máy chủ: " + err);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}
