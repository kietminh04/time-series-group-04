# Báo cáo cuối kỳ: Dự báo Chuỗi thời gian nhiều chiều - Nhóm 04

## 1. Giới thiệu dự án
Bài toán được nghiên cứu là dự báo biến mục tiêu một chiều $y$ (đơn biến) dựa trên thông tin từ chuỗi thời gian nhiều chiều đầu vào $X$.
* **Bộ dữ liệu sử dụng**: `ETTh1.csv` (Electricity Transformer Temperature - Phụ tải nhiệt độ máy biến áp).
* **Đầu vào (Input)**: Gồm 7 biến số liên tục (`HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`).
* **Đầu ra (Output)**: Nhiệt độ dầu máy biến áp tại bước tiếp theo (`OT`).

---

## 2. Tóm tắt 3 bài báo nghiên cứu
Nhóm đã tìm hiểu và tóm tắt 3 bài báo khoa học nổi bật và cập nhật gần đây về dự báo chuỗi thời gian:

### A. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (2023 - ICLR 2024)
* **Ý tưởng chính**: Thay vì xem các bước thời gian là token, iTransformer đảo chiều để xem toàn bộ chuỗi lịch sử của một biến là một token.
* **Điểm mạnh**: Học tương quan giữa các biến (multivariate correlation) rất tốt bằng Self-Attention toàn cục.
* **Hạn chế**: Không tối ưu cho dữ liệu đơn biến và có độ phức tạp cao theo bình phương số biến $O(N^2)$.

### B. TimesFM: A Decoder-Only Foundation Model for Time-Series Forecasting (Google Research, 2024 - ICML 2024)
* **Ý tưởng chính**: Phát triển mô hình nền tảng (Foundation Model) Decoder-only dựa trên causal Transformer, được huấn luyện trước trên 100 tỷ điểm dữ liệu thực tế.
* **Điểm mạnh**: Suy luận Zero-shot cực nhanh và chính xác cao trên các tập dữ liệu mới mà không cần huấn luyện lại. Kích thước gọn nhẹ (200M tham số).
* **Hạn chế**: Giới hạn độ dài ngữ cảnh lịch sử đầu vào (512 bước) và ban đầu được thiết kế cho đơn biến (univariate).

### C. Timer: Generative Pre-trained Transformers Are Large Time Series Models (2024 - ICML 2024)
* **Ý tưởng chính**: Phát triển mô hình nền tảng (Foundation Model) dạng GPT-style pre-trained trên hàng tỷ điểm dữ liệu.
* **Điểm mạnh**: Cho phép dự báo Zero-shot (không cần huấn luyện lại) và Few-shot rất tốt trên các tập dữ liệu mới.
* **Hạn chế**: Mô hình rất lớn (1.3 tỷ tham số), đòi hỏi tài nguyên tính toán và GPU rất mạnh.

---

## 3. Quy trình thực hiện & Phương pháp
Quy trình thực nghiệm bao gồm:

### Tiền xử lý dữ liệu
1. **Kiểm tra dữ liệu**: Xác định dữ liệu lấy mẫu đều đặn (mỗi giờ) và không bị khuyết (missing values).
2. **Xử lý ngoại lệ (Outliers)**: Áp dụng phương pháp IQR để giới hạn (clip) các điểm dữ liệu dị biệt.
3. **Phân tích tần số bằng Fourier (FFT)**: Detrend chuỗi thời gian bằng phương pháp Linear Detrending, sau đó chạy thuật toán Real FFT. Kết quả phân tích mật độ phổ công suất (PSD) cho thấy 2 chu kỳ nổi bật nhất là **24 giờ** (chu kỳ ngày) và **168 giờ** (chu kỳ tuần).

### Feature Engineering
* **Biến thời gian (Calendar features)**: Trích xuất `hour`, `dayofweek`, `month`, `is_weekend`.
* **Biến Fourier (Fourier basis features)**: Tạo các biến sin/cos đại diện cho mùa vụ ngày và tuần:
  $$\sin\left(\frac{2\pi t}{24}\right), \cos\left(\frac{2\pi t}{24}\right), \sin\left(\frac{2\pi t}{168}\right), \cos\left(\frac{2\pi t}{168}\right)$$
* **Biến độ trễ (Lag features)**: Tạo các lag của target `OT` (1, 2, 3, 24, 48, 168).
* **Biến trung bình trượt (Rolling features)**: Tạo trung bình trượt và độ lệch chuẩn trượt với cửa sổ 24 giờ và 168 giờ.

### Chuẩn hóa & Chia dữ liệu
* Chia dữ liệu theo thứ tự thời gian: **70% Train, 15% Validation, 15% Test**.
* Chuẩn hóa Z-score dựa trên các tham số (mean, std) được tính toán hoàn toàn từ tập Train để tránh rò rỉ dữ liệu (data leakage).

---

## 4. Thiết lập mô hình & Kết quả

Nhóm đã xây dựng các lớp mô hình bao gồm:
1. **Baseline**: Naive (dự báo bằng giá trị cuối), Moving Average (trung bình trượt 24h), Linear Regression (chỉ dùng Lags).
2. **Seasonality / ML**: Linear Regression kết hợp biến Fourier và XGBoost.
3. **Advanced**: Mô hình mạng nơ-ron hồi quy **PyTorch LSTM** với lookback window = 24.

### Bảng kết quả so sánh thực tế
| Mô hình | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive | 0.0559 | 0.0829 | 23.63% |
| Moving Average | 0.1437 | 0.1989 | 52.91% |
| Linear Regression Baseline | 0.0555 | 0.0821 | 23.61% |
| Linear Regression Fourier | 0.0548 | 0.0809 | 26.34% |
| XGBoost Regressor | 0.0571 | 0.0836 | 18.68% |
| PyTorch LSTM (Advanced) | 0.2104 | 0.2713 | 82.48% |

### Nhận xét kết quả:
* Các mô hình hồi quy tuyến tính (LR) và mô hình Naive cho kết quả rất tốt do tính tự tương quan cao (hour-to-hour correlation) của chuỗi nhiệt độ dầu máy biến áp.
* Việc bổ sung biến mùa vụ Fourier giúp cải thiện hiệu năng của Linear Regression một cách rõ rệt (giảm MAE từ 0.0555 xuống 0.0548 và RMSE từ 0.0821 xuống 0.0809).
* Mô hình XGBoost đạt sai số tương đối (MAPE) thấp nhất (18.68%) nhờ khả năng học các mối quan hệ phi tuyến giữa các đặc trưng thời tiết/phụ tải phụ.
* Mô hình PyTorch LSTM nâng cao cần được huấn luyện trong nhiều epoch hơn và tinh chỉnh hyperparameter kỹ lưỡng hơn để tối ưu hóa hiệu năng so với các baselines đơn giản.

---

## 5. Kết luận
* Dự án đã giải quyết thành công bài toán dự báo chuỗi thời gian nhiều chiều bằng cách thiết lập hệ thống từ phân tích Fourier (FFT), tạo đặc trưng mùa vụ cho đến việc so sánh các mô hình Machine Learning và Deep Learning.
* Trong tương lai, nhóm có thể mở rộng thử nghiệm với các giải pháp Zero-shot từ TimesFM và Timer nếu có thêm tài nguyên GPU lớn hơn.
