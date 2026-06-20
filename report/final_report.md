# BÁO CÁO ĐỒ ÁN CUỐI KỲ: DỰ BÁO CHUỖI THỜI GIAN NHIỀU CHIỀU
**Môn học:** Phân tích Chuỗi thời gian (Time Series Analysis)  
**Nhóm thực hiện:** Nhóm 04  

---

## I. GIỚI THIỆU ĐỀ TÀI & BỘ DỮ LIỆU

### 1. Đặt vấn đề
Nhiệt độ dầu máy biến áp (Oil Temperature - `OT`) là một chỉ số sống còn đối với sự vận hành an toàn và ổn định của lưới điện. Việc dự báo chính xác nhiệt độ dầu máy giúp đưa ra các cảnh báo sớm về nguy cơ quá tải, giảm thiểu rủi ro sự cố và tối ưu hóa kế hoạch bảo trì. Đồ án này tập trung nghiên cứu bài toán dự báo nhiệt độ dầu máy biến áp tại bước tiếp theo bằng cách sử dụng các mô hình chuỗi thời gian nhiều chiều (Multivariate Time Series Forecasting), kết hợp phân tích mùa vụ bằng biến đổi Fourier (FFT), mô hình máy học (XGBoost) và các kiến trúc học sâu nâng cao (LSTM, GRU, Transformer).

### 2. Bộ dữ liệu nghiên cứu
Bộ dữ liệu được sử dụng là **ETTm2** (Electricity Transformer Temperature) thu thập với chu kỳ 15 phút một lần, tổng cộng có **69,680 mẫu** dữ liệu. Các thuộc tính bao gồm:
*   `date`: Thời gian lấy mẫu (độ phân giải 15 phút).
*   `HUFL` & `HULL` (High Use Frequency Load): Phụ tải tần số cao (mức lớn/mức nhỏ).
*   `MUFL` & `MULL` (Medium Use Frequency Load): Phụ tải tần số trung bình (mức lớn/mức nhỏ).
*   `LUFL` & `LULL` (Low Use Frequency Load): Phụ tải tần số thấp (mức lớn/mức nhỏ).
*   `OT` (Oil Temperature - Target): Biến mục tiêu cần dự báo.

---

## II. QUY TRÌNH XỬ LÝ DỮ LIỆU & TRÍCH XUẤT ĐẶC TRƯNG

### 1. Tiền xử lý dữ liệu & Khử nhiễu ngoại lai
*   Dữ liệu được làm sạch, kiểm tra giá trị thiếu (không có giá trị thiếu trong ETTm2).
*   Áp dụng phương pháp IQR (Interquartile Range) để phát hiện và giới hạn các điểm ngoại lệ (outliers) trên biến mục tiêu `OT`, tránh hiện tượng mô hình học sâu bị nhiễu kéo lệch phân phối.

### 2. Phân tích mùa vụ bằng biến đổi Fourier (FFT)
*   Thực hiện khử xu hướng tuyến tính (detrending) trên chuỗi `OT`.
*   Áp dụng biến đổi Fourier thực (Real FFT) để chuyển dữ liệu từ miền thời gian sang miền tần số.
*   Kết quả phân tích phổ tần số cho thấy hai chu kỳ mùa vụ mạnh nhất là:
    *   **Mùa vụ ngày (24 giờ):** Tương đương 96 bước thời gian (mỗi bước 15 phút).
    *   **Mùa vụ tuần (168 giờ):** Tương đương 672 bước thời gian.

### 3. Trích xuất đặc trưng (Feature Engineering)
Dựa trên phân tích FFT, tập đặc trưng được xây dựng bao gồm:
*   **Đặc trưng lịch (Calendar Features):** `hour`, `dayofweek`, `month`, và biến phân loại cuối tuần `is_weekend`.
*   **Đặc trưng mùa vụ Fourier (Fourier Features):** Các hàm sóng $\sin$ và $\cos$ tương ứng với chu kỳ 24 giờ và 168 giờ.
*   **Đặc trưng trễ (Lag Features):** Các bước trễ lịch sử của biến `OT` bao gồm trễ $1, 2, 4, 8, 24, 48, 96, 192$ bước.
*   **Đặc trưng trượt (Rolling Features):** Giá trị trung bình trượt (Rolling Mean) và độ lệch chuẩn trượt (Rolling Std) của biến `OT` trên các cửa sổ $4, 12, 24, 96$ bước.

### 4. Chia tập dữ liệu & Chuẩn hóa
*   Tập dữ liệu được chia theo thời gian thành 3 tập riêng biệt nhằm đảm bảo tính khách quan: **70% Train** (48,305 dòng), **15% Val** (10,352 dòng) và **15% Test** (10,352 dòng).
*   Áp dụng chuẩn hóa **Z-score** trên tập đặc trưng, trong đó tham số trung bình và phương sai được tính *chỉ trên tập Train* để tránh hiện tượng rò rỉ thông tin (data leakage) sang tập Val và Test.

---

## III. KIẾN TRÚC CÁC MÔ HÌNH DỰ BÁO

Chúng ta xây dựng và huấn luyện 3 nhóm mô hình chính:

### 1. Nhóm mô hình cơ sở (Baselines)
*   **Naive Model:** Dự báo giá trị bước tiếp theo bằng chính giá trị hiện tại ($y_{t+1} = y_t$).
*   **Moving Average Model (MA):** Dự báo bằng giá trị trung bình của cửa sổ 96 bước (24 tiếng lịch sử).
*   **Linear Regression Baseline:** Hồi quy tuyến tính chỉ sử dụng các đặc trưng trễ (Lag Features).

### 2. Nhóm mô hình máy học & Mùa vụ (ML & SARIMAX)
*   **Linear Regression Fourier:** Hồi quy tuyến tính kết hợp cả đặc trưng trễ và đặc trưng mùa vụ Fourier.
*   **XGBoost Regressor:** Mô hình cây quyết định tăng cường gradient, huấn luyện trên toàn bộ tập đặc trưng (Lịch, Fourier, Trễ, Trượt).
*   **SARIMAX(1,1,1)(1,0,0,24):** Mô hình tự hồi quy tích hợp trung bình trượt mùa vụ có biến ngoại sinh (Fourier). Nhằm tối ưu tốc độ chạy, dữ liệu được downsample sang tần số giờ (hourly), sử dụng 30 ngày cuối tập Train để huấn luyện và kiểm tra trên 10 ngày đầu tập Test.

### 3. Nhóm mô hình học sâu nâng cao (Deep Learning - PyTorch)
Cả 3 mô hình học sâu đều sử dụng chuỗi đầu vào độ dài **96 bước** (lookback window = 96), kích thước batch 64, tối ưu bằng thuật toán Adam với learning rate 0.001. Sử dụng kỹ thuật dừng sớm (Early Stopping) với patience = 10 dựa trên sai số Validation Loss:
*   **LSTM (Long Short-Term Memory):** Mạng nơ-ron tuần tự 2 tầng ẩn, hidden dimension = 128, giải quyết vấn đề triệt tiêu gradient trên chuỗi dài.
*   **GRU (Gated Recurrent Unit):** Kiến trúc rút gọn của LSTM với ít tham số hơn, sử dụng các cổng cập nhật và cổng thiết lập lại giúp tăng tốc độ huấn luyện.
*   **Transformer:** Mô hình tự chú ý (Self-Attention) sử dụng 2 tầng mã hóa (Encoder layers), d_model = 64, nhead = 4, giúp nắm bắt mối quan hệ xa trong chuỗi mà không phụ thuộc vào thứ tự tuần tự của các bước.

---

## IV. KẾT QUẢ THỰC NGHIỆM & SO SÁNH HIỆU NĂNG

Các mô hình được đánh giá trên tập kiểm thử (Test Set) bằng 3 chỉ số chính: Sai số tuyệt đối trung bình (MAE), Căn phương sai sai số trung bình (RMSE), và Sai số phần trăm tuyệt đối trung bình (MAPE).

### Bảng kết quả tổng hợp:
| STT | Mô hình | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|
| 1 | **Linear Regression (Fourier)** | **0.0151** | **0.0209** | **18.01%** |
| 2 | **Linear Regression (Base)** | 0.0150 | 0.0212 | 17.17% |
| 3 | **XGBoost Regressor** | 0.0178 | 0.0266 | 19.72% |
| 4 | **PyTorch GRU (Deep)** | 0.1522 | 0.2040 | 139.12% |
| 5 | **PyTorch Transformer (Deep)** | 0.1923 | 0.2610 | 178.62% |
| 6 | **PyTorch LSTM (Deep)** | 0.2367 | 0.3419 | 238.47% |
| 7 | **Naive** | 0.0220 | 0.0326 | 22.61% |
| 8 | **SARIMAX (Hourly)** | 0.5005 | 0.6255 | 318.40% |
| 9 | **Moving Average (24h)** | 0.3358 | 0.4271 | 239.35% |

### Nhận xét & Thảo luận:
1.  **Sự vượt trội của mô hình tuyến tính (Linear Regression):**
    *   Mô hình `Linear Regression (Fourier)` đạt sai số RMSE thấp nhất (0.0209) nhờ việc bổ sung các hàm sóng sin/cos mùa vụ ngày và tuần từ phân tích FFT. 
    *   Hồi quy tuyến tính hoạt động cực tốt trong bài toán này do tính chất dự báo ngắn hạn (dự báo 1 bước kế tiếp - 15 phút tương lai). Biến nhiệt độ dầu `OT` biến thiên rất mượt mà và tuyến tính trong khoảng thời gian ngắn, giúp hồi quy tuyến tính nắm bắt nhanh chóng và không bị hiện tượng quá khớp (overfitting).
2.  **So sánh các mô hình Deep Learning:**
    *   Trong nhóm học sâu, **GRU** đạt kết quả tốt nhất (MAE = 0.1522, RMSE = 0.2040), vượt trội rõ rệt so với **Transformer** (RMSE = 0.2610) và **LSTM** (RMSE = 0.3419). GRU có cấu trúc cổng đơn giản hơn giúp mô hình học nhanh và ít bị overfitting hơn LSTM trên tập dữ liệu này.
    *   Nhìn chung, cả 3 mô hình Deep Learning đều có sai số lớn hơn các mô hình tuyến tính cơ bản. Điều này là do các mô hình học sâu có số lượng tham số lớn, dễ có xu hướng "quá khớp" trên dữ liệu nhiễu tần suất cao (15 phút) khi dự báo chỉ 1 bước trước. Chúng sẽ phát huy thế mạnh tốt hơn trong bài toán dự báo dài hạn (Multi-step ahead forecasting).
3.  **Mô hình SARIMAX:**
    *   Mô hình SARIMAX có sai số tương đối lớn (RMSE = 0.6255) vì được huấn luyện trên dữ liệu downsample theo giờ và dự báo trong khoảng thời gian xa hơn, đồng thời bị hạn chế về khả năng tự cập nhật khi không chạy rolling forecast liên tục.

---

## V. KẾT LUẬN & HƯỚNG PHÁT TRIỂN

### 1. Kết luận
*   Phân tích phổ Fourier (FFT) là công cụ tiền đề mạnh mẽ giúp xác định chu kỳ mùa vụ tự nhiên (24h và 168h) của nhiệt độ dầu máy biến áp, từ đó tạo ra các đặc trưng Fourier giúp tối ưu sai số cho mô hình hồi quy.
*   Với bài toán dự báo ngắn hạn 1 bước thời gian (15 phút kế tiếp), các mô hình tuyến tính đơn giản như Hồi quy tuyến tính (LR) và XGBoost mang lại hiệu năng tối ưu nhất về cả độ chính xác (RMSE cực nhỏ) lẫn chi phí tính toán (chạy tính bằng giây).
*   Các kiến trúc học sâu (Deep Learning) cần được kiểm soát overfitting chặt chẽ và chỉ nên ưu tiên áp dụng khi bài toán mở rộng ra dự báo dài hạn nhiều bước trong tương lai.

### 2. Hướng phát triển
*   Mở rộng bài toán dự báo dài hạn (Multi-step ahead forecasting) ví dụ dự báo liên tục 24 tiếng tương lai ($y_{t+1}$ đến $y_{t+96}$) để kiểm nghiệm khả năng nắm bắt xu hướng dài hạn của LSTM và Transformer.
*   Áp dụng các kỹ thuật tinh chỉnh hyperparameter nâng cao (như Optuna) để tối ưu số lớp ẩn, chiều ẩn và tỷ lệ dropout của các mô hình học sâu.
*   Thử nghiệm các kiến trúc lai (Hybrid Models) kết hợp giữa Hồi quy tuyến tính/FFT và Deep Learning để tận dụng ưu điểm của cả hai bên.
