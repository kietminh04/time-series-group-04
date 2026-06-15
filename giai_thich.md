Dưới đây là giải thích chi tiết về phương pháp chuẩn hóa dữ liệu và hướng dẫn từng bước để bạn tự chạy code (của nhóm mình và của bạn bạn).

---

### 1. Đơn vị chuẩn hóa dữ liệu (Z-score Normalization) là gì?

Chúng ta sử dụng phương pháp chuẩn hóa **Z-score** (hay còn gọi là Standard Scaling). Phương pháp này biến đổi dữ liệu về dạng:
* **Giá trị trung bình (Mean) = 0**
* **Độ lệch chuẩn (Standard Deviation) = 1**

Công thức tính cho từng điểm dữ liệu $x$:
$$z = \frac{x - \mu}{\sigma}$$
*Trong đó: $\mu$ là giá trị trung bình của tập Train, $\sigma$ là độ lệch chuẩn của tập Train.*

#### Ý nghĩa của đơn vị mới:
Khi chuyển sang Z-score, đơn vị của dữ liệu không còn là độ C (°C) nữa, mà biến thành **"Số lần độ lệch chuẩn lệch so với giá trị trung bình"**.
* **Ví dụ thực tế**: Giả sử nhiệt độ dầu máy trung bình là $18^\circ\text{C}$ ($\mu = 18$), độ lệch chuẩn là $9^\circ\text{C}$ ($\sigma = 9$):
  * Nếu nhiệt độ thực tế là $18^\circ\text{C}$ $\rightarrow z = 0$ (bằng đúng trung bình).
  * Nếu nhiệt độ thực tế là $27^\circ\text{C}$ $\rightarrow z = +1$ (lệch cao hơn trung bình **1 lần độ lệch chuẩn**).
  * Nếu nhiệt độ thực tế là $9^\circ\text{C}$ $\rightarrow z = -1$ (lệch thấp hơn trung bình **1 lần độ lệch chuẩn**).

#### Tại sao phải làm thế?
Trong dữ liệu có nhiều biến khác nhau (ví dụ: phụ tải `HUFL` có thể lên tới hàng trăm, nhưng nhiệt độ `OT` chỉ khoảng vài chục). Nếu giữ nguyên đơn vị gốc, các mô hình Máy học và Học sâu (như LSTM, XGBoost) sẽ bị tính toán lệch lạc vì các biến có số lớn sẽ "áp đảo" các biến có số nhỏ. Chuẩn hóa Z-score giúp đưa tất cả các biến về cùng một hệ quy chiếu (quanh mốc 0) để mô hình học nhanh và chính xác hơn.

---

### 2. Hướng dẫn tự chạy Code

Để chạy code, trước tiên bạn cần mở **Terminal** (Dòng lệnh) trong thư mục dự án `time-series-group-04` và kích hoạt môi trường ảo:
* **Trên Windows PowerShell**: `.\env\Scripts\Activate.ps1`
* **Trên Windows CMD**: `env\Scripts\activate`

---

#### CÁCH CHẠY CODE CỦA NHÓM MÌNH (4 Notebooks)

Bạn mở VS Code lên, mở lần lượt các file trong thư mục `notebooks/` theo thứ tự và nhấn **"Run All"** (chú ý chọn Kernel góc trên bên phải là `env`):

1. **`notebooks/01_data_exploration.ipynb`**: Đọc dữ liệu thô, phân tích thống kê mô tả, vẽ biểu đồ chuỗi thời gian, và phân tích tần số Fourier (FFT).
2. **`notebooks/02_feature_engineering.ipynb`**: Trích xuất đặc trưng (lags, rolling, fourier) và thực hiện chuẩn hóa Z-score, xuất dữ liệu ra thư mục `data/processed/`.
3. **`notebooks/03_models.ipynb`**: Huấn luyện các mô hình dự báo (Naive, LR, XGBoost, LSTM) và xuất kết quả dự báo ra `results/predictions.csv`.
4. **`notebooks/04_evaluation.ipynb`**: Tính toán sai số (MAE, RMSE, MAPE) và vẽ biểu đồ so sánh thực tế vs dự báo.

*(Nếu muốn chạy nhanh bằng dòng lệnh không cần mở giao diện kéo thả, bạn có thể gõ lệnh lần lượt trong Terminal:)*
```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_exploration.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_engineering.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_models.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
```

---

#### CÁCH CHẠY CODE CỦA BẠN BẠN (Thư mục `TimeSeries_ETTm2`)

Đầu tiên, bạn cần đảm bảo file dữ liệu `ETTm2.csv` đã được copy vào **nằm cùng thư mục** với các file code của bạn bạn (tức là nằm trong thư mục `TimeSeries_ETTm2/`).

Sau đó, gõ các lệnh sau trong Terminal theo thứ tự:

1. **Chạy các mô hình Baseline (LR, Naive, XGBoost, SARIMAX)**:
   ```bash
   python TimeSeries_ETTm2/tv2_baseline.py
   ```
   *Kết quả đầu ra sẽ được lưu vào thư mục `TimeSeries_ETTm2/results/`.*

2. **Chạy các mô hình Học sâu (LSTM, GRU, Transformer)**:
   ```bash
   python TimeSeries_ETTm2/tv3_deeplearning.py
   ```
   *Lưu ý: Lệnh này trên CPU sẽ chạy mất khoảng 3 tiếng như đã phân tích.*

3. **Vẽ thêm biểu đồ so sánh (nếu cần)**:
   ```bash
   python TimeSeries_ETTm2/generate_figures.py
   ```

---

### 3. Mô hình Hồi quy tuyến tính (Linear Regression) có dùng chuẩn hóa không và tại sao MAE lại thấp hơn?

#### Về việc chuẩn hóa đối với Linear Regression (LR):
* **Trong code của bạn bạn (`tv2_baseline.py`)**: Có sử dụng chuẩn hóa Z-score cho các đặc trưng đầu vào $X$ (`StandardScaler().fit_transform(X_tr)`), nhưng biến mục tiêu $y$ (`OT`) được giữ nguyên ở đơn vị gốc (độ C) để huấn luyện và tính sai số trực tiếp.
* **Về mặt toán học**: Đối với thuật toán Hồi quy tuyến tính bình phương tối thiểu (OLS Linear Regression), việc chuẩn hóa hay không chuẩn hóa đặc trưng $X$ **không làm thay đổi độ chính xác (MAE, RMSE) của mô hình**. Chuẩn hóa chỉ làm thay đổi tỷ lệ của các hệ số số học (coefficients) được học.
  * *Ví dụ*: Nếu bạn nhân đặc trưng $x$ lên 10 lần, mô hình sẽ tự động chia hệ số $w$ đi 10 lần, kết quả dự báo $\hat{y}$ cuối cùng và sai số MAE vẫn giống hệt nhau.

#### Tại sao LR lại có MAE thấp nhất (~0.086°C)?
1. **Bài toán mang tính tuyến tính cực cao**: Biến nhiệt độ dầu máy (`OT`) thay đổi rất trơn tru. Ở khoảng thời gian ngắn 15 phút, giá trị tiếp theo $y_{t+1}$ phụ thuộc tuyến tính gần như hoàn hảo vào các giá trị trễ trước đó ($y_t, y_{t-1}, ...$). Một phương trình tuyến tính đơn giản dạng:
   $$y_{t+1} = w_1 y_t + w_2 y_{t-1} + ... + b$$
   diễn tả quy luật vật lý này tốt hơn bất kỳ đường cong phi tuyến phức tạp nào.
2. **Tránh nhiễu (No Overfitting)**: Các mô hình phức tạp (như XGBoost hay mạng học sâu) cố gắng phân mảnh không gian hoặc học các hàm phi tuyến phức tạp. Điều này vô tình làm chúng học luôn cả các biến động nhiễu ngẫu nhiên nhỏ trong tập Train, khiến sai số MAE trên tập Test bị kéo cao lên. LR chỉ học xu hướng thẳng nên không bị bẫy bởi nhiễu này.

---

### 4. Cách cấu hình chạy huấn luyện bằng GPU cho nhanh

Để chuyển từ chạy CPU sang chạy GPU (NVIDIA), bạn cần chuẩn bị cả **Phần cứng**, **Driver** và **Thư viện Python**:

#### Bước 1: Cài đặt CUDA trên hệ điều hành Windows
1. Đảm bảo máy tính có card đồ họa rời của **NVIDIA** (không hỗ trợ card onboard Intel hay AMD).
2. Tải và cài đặt **NVIDIA Driver** bản mới nhất cho card màn hình.
3. Tải và cài đặt **CUDA Toolkit** từ trang chủ NVIDIA (ví dụ bản CUDA 11.8 hoặc 12.1 - nên chọn bản ổn định mà PyTorch hỗ trợ).

#### Bước 2: Cài đặt phiên bản PyTorch hỗ trợ GPU (CUDA)
Mặc định nếu bạn gõ `pip install torch` thông thường, hệ thống có thể chỉ tải bản chạy CPU. Để cài đúng bản GPU, bạn cần mở Terminal trong môi trường ảo `env` và chạy lệnh cài đặt chuyên dụng (ví dụ cho CUDA 11.8):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
*(Nếu muốn cài CUDA 12.1, thay `cu118` thành `cu121`)*.

Sau khi cài xong, kiểm tra lại bằng lệnh:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
*Nếu kết quả trả về là **`True`** thì máy bạn đã sẵn sàng chạy bằng GPU!*

#### Bước 3: Cấu hình trong mã nguồn Python (PyTorch)
Trong code PyTorch, bạn chỉ cần chỉ định chuyển Mô hình và Dữ liệu lên GPU bằng hàm `.to(device)`:
```python
# 1. Định nghĩa thiết bị chạy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Đưa mô hình lên GPU
model = LSTMModel(input_size).to(device)

# 3. Đưa dữ liệu (features và targets) lên GPU trong vòng lặp huấn luyện
for xb, yb in train_loader:
    xb = xb.to(device)
    yb = yb.to(device)
    
    # Tính toán...
```
*(Đoạn code trong file `tv3_deeplearning.py` của bạn bạn đã được viết sẵn cấu trúc `.to(DEVICE)` này, chỉ cần cài đúng PyTorch GPU là hệ thống sẽ tự động chạy bằng card đồ họa của máy).*

---

### 5. Tại sao mô hình lại dừng sớm (Early Stopping) ở Epoch 20?

Trong nhật ký huấn luyện của bạn, mô hình LSTM đã kích hoạt tính năng **Dừng sớm (Early Stopping)** tại epoch 20 thay vì chạy đủ 50 epochs:
```text
  Epoch  10/50: train=0.001604 | val=0.000504  <-- Điểm val loss tốt nhất (Thấp nhất)
  ...
  Epoch  20/50: train=0.001324 | val=0.000843  
  Early stopping tại epoch 20 (best val=0.000504)
```

#### Nguyên nhân:
1. **Patience (Độ kiên nhẫn)** được cài đặt là **`PATIENCE = 10`**. Điều này có nghĩa là nếu sau **10 epoch liên tiếp** mà sai số trên tập xác thực (`val loss`) không có cải tiến nào tốt hơn giá trị tốt nhất đã ghi nhận, mô hình sẽ tự động dừng quá trình huấn luyện.
2. Từ Epoch 10 đến Epoch 20 (đúng 10 epochs), sai số trên tập Validation liên tục dao động lớn hơn mốc tốt nhất là `0.000504` (cụ thể Epoch 20 vọt lên `0.000843`), trong khi sai số tập Train vẫn giảm (`0.001604` xuống `0.001324`).
3. **Ý nghĩa**: Điều này báo hiệu mô hình bắt đầu bị **quá khớp (overfitting)** — nó đang cố "học vẹt" dữ liệu huấn luyện và làm suy giảm khả năng dự báo trên dữ liệu thực tế. Việc dừng sớm giúp tiết kiệm thời gian chạy máy và giữ lại phiên bản trọng số mô hình tốt nhất ở Epoch 10.

---

### 6. Đường dẫn tải trực tiếp NVIDIA Driver và CUDA Toolkit

Để cấu hình chạy GPU nhanh nhất, bạn tải các bộ cài đặt chính thức từ NVIDIA theo các liên kết dưới đây:

#### 1. NVIDIA Graphics Driver (Trình điều khiển Card đồ họa)
Bạn nên cài đặt driver bản mới nhất để card đồ họa hoạt động tối ưu:
* **Trang chủ tìm kiếm Driver tự động**: [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
  *(Chọn dòng Card đồ họa của máy bạn, chọn Windows 10/11 và bấm Search để tải bản phù hợp nhất).*

#### 2. CUDA Toolkit (Bộ thư viện lập trình GPU)
Khuyên dùng phiên bản **11.8** hoặc **12.1** vì đây là hai phiên bản ổn định nhất và tương thích rộng rãi nhất với thư viện PyTorch hiện tại:
* **CUDA Toolkit 11.8 (Khuyên dùng nhiều nhất)**: [Tải trực tiếp CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
* **CUDA Toolkit 12.1 (Bản nâng cao)**: [Tải trực tiếp CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
* **Thư mục lưu trữ tất cả phiên bản (Archive)**: [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

*Lưu ý khi cài đặt*: Chọn phiên bản hệ điều hành Windows (thường là x86_64, Windows 11 hoặc 10, chọn kiểu cài đặt **exe (local)** để tải toàn bộ gói về máy chạy cài đặt ngoại tuyến cho ổn định). Cài đặt xong, hãy khởi động lại máy tính trước khi thử cài đặt PyTorch GPU.