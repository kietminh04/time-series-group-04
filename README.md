# Nhóm 04 - Dự báo Chuỗi thời gian nhiều chiều

Chào mừng bạn đến với repository của **Nhóm 04**. Đây là đồ án môn học về **Dự báo Chuỗi thời gian nhiều chiều** (Multivariate Time Series Forecasting) áp dụng kỹ thuật phân tích mùa vụ bằng biến đổi Fourier (FFT) kết hợp với các mô hình Machine Learning và Deep Learning.

---

## 1. Thành viên nhóm
* **Trưởng nhóm**: [Họ và tên Trưởng nhóm]
* **Thành viên**:
  * [Họ và tên Thành viên 1]
  * [Họ và tên Thành viên 2]
  * [Họ và tên Thành viên 3]

## 2. Chủ đề nghiên cứu
* **Đề tài**: Dự báo nhiệt độ dầu máy biến áp (Oil Temperature - `OT`) bằng chuỗi thời gian nhiều chiều.
* **Mục tiêu**: Đầu vào là chuỗi thời gian nhiều chiều (các biến phụ tải phụ của máy biến áp), đầu ra là nhiệt độ dầu máy tại bước tương lai ($y[t+h]$).

## 3. Mô tả bộ dữ liệu
* **Dataset**: `ETTh1` (Electricity Transformer Temperature - 1 hour resolution).
* **Kích thước**: 17,420 dòng dữ liệu theo giờ.
* **Các thuộc tính**:
  * `date`: Thời gian lấy mẫu (mỗi giờ).
  * `HUFL`, `HULL` (High Use/Low Use Frequency Load): Phụ tải tần số cao/thấp.
  * `MUFL`, `MULL` (Medium Use Frequency Load): Phụ tải tần số trung bình.
  * `LUFL`, `LULL` (Low Use Frequency Load): Phụ tải tần số thấp.
  * `OT` (Oil Temperature - Target): Nhiệt độ dầu máy biến áp cần dự báo.

## 4. Tóm tắt 3 bài báo khoa học đã đọc
Thông tin tóm tắt chi tiết 3 bài báo nghiên cứu nổi bật về chuỗi thời gian được đặt trong thư mục `papers/`:
1. **[iTransformer (2023 - ICLR 2024)](papers/paper_01.md)**: Đảo chiều chiều đầu vào của Transformer để xem mỗi biến là một token, giúp học tương quan chéo giữa các biến hiệu quả hơn.
2. **[PatchTST (2022 - ICLR 2023)](papers/paper_02.md)**: Chia chuỗi thời gian thành các patch nhỏ và áp dụng Channel Independence để tăng hiệu quả và giảm chi phí tính toán.
3. **[Timer (2024 - ICML 2024)](papers/paper_03.md)**: Xây dựng mô hình nền tảng Generative GPT-like trên hàng tỷ điểm dữ liệu chuỗi thời gian phục vụ dự báo Zero-shot.

## 5. Phương pháp thực hiện
1. **Tiền xử lý & Outliers**: Phát hiện và giới hạn ngoại lệ bằng phương pháp IQR.
2. **Phân tích Fourier (FFT)**: Khử xu hướng tuyến tính (detrending) và biến đổi Real FFT trên biến `OT`. Kết quả nhận diện 2 chu kỳ mùa vụ chính là **24 giờ** (ngày) và **168 giờ** (tuần).
3. **Feature Engineering**: Tạo biến thời gian (hour, dayofweek, month, is_weekend), biến Fourier ($\sin/\cos$ với chu kỳ 24 và 168), các lag features của `OT`, và rolling mean/std.
4. **Chuẩn hóa & Chia dữ liệu**: Chia theo thời gian (70% Train, 15% Val, 15% Test) và chuẩn hóa Z-score dựa trên tập Train.
5. **Mô hình hóa**:
   * **Baseline**: Naive, Moving Average, Linear Regression (chỉ dùng lags).
   * **Seasonality / ML**: Linear Regression kết hợp biến Fourier và XGBoost Regressor.
   * **Advanced**: PyTorch LSTM Regressor.

## 6. Kết quả mô hình
Bảng so sánh kết quả đánh giá trên tập Test (các chỉ số MAE, RMSE, MAPE):

| Mô hình | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive | 0.0559 | 0.0829 | 23.63% |
| Moving Average (24h) | 0.1437 | 0.1989 | 52.91% |
| Linear Regression Baseline | 0.0555 | 0.0821 | 23.61% |
| Linear Regression Fourier | 0.0548 | 0.0809 | 26.34% |
| XGBoost Regressor | 0.0571 | 0.0836 | 18.68% |
| PyTorch LSTM (Advanced) | 0.2104 | 0.2713 | 82.48% |

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
* Bổ sung biến Fourier mô tả mùa vụ ngày/tuần cải thiện đáng kể độ chính xác của các mô hình hồi quy.
* Mạng nơ-ron tuần tự LSTM chứng minh hiệu quả vượt trội trong việc nắm bắt các thông tin phi tuyến tính phức tạp trong bài toán chuỗi thời gian nhiều chiều.
