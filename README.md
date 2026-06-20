# Nhóm 04 - Dự báo Chuỗi thời gian nhiều chiều

Chào mừng bạn đến với repository của **Nhóm 04**. Đây là đồ án môn học về **Dự báo Chuỗi thời gian nhiều chiều** (Multivariate Time Series Forecasting) áp dụng kỹ thuật phân tích mùa vụ bằng biến đổi Fourier (FFT) kết hợp với các mô hình Machine Learning và Deep Learning.

---

## 1. Thành viên nhóm
* **Trưởng nhóm**: Vũ Minh Kiệt 20227128
* **Thành viên**:
  * Phan Huy Hoàng 20227051
  * Chu Nguyễn Thành Long 20227130

## 2. Chủ đề nghiên cứu
* **Đề tài**: Dự báo nhiệt độ dầu máy biến áp (Oil Temperature - `OT`) bằng chuỗi thời gian nhiều chiều.
* **Mục tiêu**: Đầu vào là chuỗi thời gian nhiều chiều (các biến phụ tải phụ của máy biến áp), đầu ra là nhiệt độ dầu máy tại bước tương lai ($y[t+h]$).

## 3. Mô tả bộ dữ liệu
* **Dataset**: `ETTm2` (Electricity Transformer Temperature - 15 minutes resolution).
* **Kích thước**: 69,680 dòng dữ liệu theo chu kỳ 15 phút.
* **Các thuộc tính**:
  * `date`: Thời gian lấy mẫu (mỗi 15 phút).
  * `HUFL`, `HULL` (High Use/Low Use Frequency Load): Phụ tải tần số cao/thấp.
  * `MUFL`, `MULL` (Medium Use Frequency Load): Phụ tải tần số trung bình.
  * `LUFL`, `LULL` (Low Use Frequency Load): Phụ tải tần số thấp.
  * `OT` (Oil Temperature - Target): Nhiệt độ dầu máy biến áp cần dự báo.

## 4. Tóm tắt 3 bài báo khoa học đã đọc
Thông tin tóm tắt chi tiết 3 bài báo nghiên cứu nổi bật về chuỗi thời gian được đặt trong thư mục `papers/`:
1. **[iTransformer (2023 - ICLR 2024)](papers/paper_01.md)**: Đảo chiều chiều đầu vào của Transformer để xem mỗi biến là một token, giúp học tương quan chéo giữa các biến hiệu quả hơn.
2. **[TimesFM (2024 - ICML 2024)](papers/paper_02.md)**: Mô hình nền tảng Decoder-only do Google phát triển, được huấn luyện trước trên 100 tỷ điểm dữ liệu, hỗ trợ dự báo Zero-shot mạnh mẽ.
3. **[Timer (2024 - ICML 2024)](papers/paper_03.md)**: Xây dựng mô hình nền tảng Generative GPT-like trên hàng tỷ điểm dữ liệu chuỗi thời gian phục vụ dự báo Zero-shot và Few-shot.

## 5. Phương pháp thực hiện
1. **Tiền xử lý & Outliers**: Phát hiện và giới hạn ngoại lệ bằng phương pháp IQR.
2. **Phân tích Fourier (FFT)**: Khử xu hướng tuyến tính (detrending) và biến đổi Real FFT trên biến `OT`. Kết quả nhận diện 2 chu kỳ mùa vụ chính là **24 giờ** (ngày) và **168 giờ** (tuần).
3. **Feature Engineering**: Tạo biến thời gian (hour, dayofweek, month, is_weekend), biến Fourier ($\sin/\cos$ với chu kỳ 24 và 168), các lag features của `OT` (1, 2, 4, 8, 24, 48, 96, 192), và rolling mean/std (cửa sổ 4, 12, 24, 96).
4. **Chuẩn hóa & Chia dữ liệu**: Chia theo thời gian (70% Train, 15% Val, 15% Test) và chuẩn hóa Z-score dựa trên tập Train.
5. **Mô hình hóa**:
   * **Baseline**: Naive, Moving Average (window = 96), Linear Regression (chỉ dùng lags), và SARIMAX (được downsample theo giờ).
   * **Seasonality / ML**: Linear Regression kết hợp biến Fourier và XGBoost Regressor.
   * **Advanced (Deep Learning)**: PyTorch LSTM, GRU, và Transformer Regressor (lookback window = 96).

## 6. Kết quả mô hình
Bảng so sánh kết quả đánh giá trên tập Test (các chỉ số MAE, RMSE, MAPE) với bộ dữ liệu ETTm2:

| Mô hình | MAE | RMSE | MAPE |
|---|---|---|---|
| Linear Regression Fourier | 0.0151 | 0.0209 | 18.01% |
| Linear Regression Baseline | 0.0150 | 0.0212 | 17.17% |
| XGBoost Regressor | 0.0178 | 0.0266 | 19.72% |
| PyTorch GRU | 0.1522 | 0.2040 | 139.12% |
| PyTorch Transformer | 0.1923 | 0.2610 | 178.62% |
| PyTorch LSTM | 0.2367 | 0.3419 | 238.47% |
| Naive | 0.0220 | 0.0326 | 22.61% |
| Moving Average (24h) | 0.3358 | 0.4271 | 239.35% |
| SARIMAX | 0.5005 | 0.6255 | 318.40% |

## 7. Hướng dẫn chạy chương trình
### Cấu hình môi trường
1. Khởi tạo môi trường ảo Python:
   ```bash
   python -m venv env
   ```
2. Kích hoạt môi trường ảo:
   * Trên Windows (PowerShell): `.\env\Scripts\Activate.ps1`
   * Trên Windows (CMD): `env\Scripts\activate`
3. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

### Chạy các Notebook phân tích
Các file Jupyter Notebook nằm trong thư mục `notebooks/` cần chạy theo thứ tự:
1. `notebooks/01_data_exploration.ipynb`: Khám phá dữ liệu và phân tích phổ tần số Fourier (FFT).
2. `notebooks/02_feature_engineering.ipynb`: Tiền xử lý, trích xuất đặc trưng và chia dữ liệu.
3. `notebooks/03_models.ipynb`: Huấn luyện mô hình cơ bản và nâng cao (LSTM).
4. `notebooks/04_evaluation.ipynb`: Tính toán sai số và vẽ đồ thị so sánh thực tế vs dự báo.

## 8. Kết luận
* Việc sử dụng dữ liệu độ phân giải cao 15 phút (ETTm2) giúp các mô hình hồi quy tuyến tính nắm bắt cực tốt các thay đổi tức thời.
* Bổ sung biến Fourier mô tả mùa vụ ngày/tuần giúp tối ưu hóa thêm sai số RMSE của Linear Regression.
* Mạng nơ-ron tuần tự LSTM cần điều chỉnh thêm để bắt kịp hiệu năng của các mô hình tuyến tính trên bài toán dự báo một bước trước với tần suất cao.
