# Paper 1: iTransformer

## Details
- **Title**: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
- **Authors**: Yuxuan Jia, et al.
- **Year**: 2023 (ICLR 2024)
- **Link/Reference**: [arXiv:2310.06625](https://arxiv.org/abs/2310.06625)
- **Keywords**: Inverted Transformer, time series forecasting, multivariate

## Summary
### Vấn đề nghiên cứu
Các mô hình Transformer truyền thống xử lý chuỗi thời gian bằng cách mã hóa các thời điểm (temporal steps) thành các token, điều này hạn chế khả năng học các mối tương quan chéo giữa nhiều biến số khác nhau (multivariate correlation).

### Ý tưởng chính & Mô hình đề xuất
Đảo ngược cấu trúc đầu vào của Transformer (Inverted Transformer): coi toàn bộ chuỗi thời gian trong cửa sổ quan sát (lookback window) của mỗi biến số là một "variate token". Cơ chế Self-Attention sau đó sẽ học mối quan hệ tương quan chéo trực tiếp giữa các biến số, thay vì học mối quan hệ thời gian.

### Kết quả chính
- Đạt kết quả State-of-the-art (SOTA) trên các bộ dữ liệu lớn như ETT, Weather, Electricity, Traffic.
- Giảm sai số MSE từ 15% - 21% so với các Transformer truyền thống.
- Vượt qua các mô hình mạnh như Autoformer, Informer, DLinear và PatchTST.
- Tốc độ huấn luyện nhanh và hiệu quả.

### Điểm mạnh & Hạn chế
- **Ưu điểm**: Không cần sửa đổi kiến trúc lõi của Transformer mà chỉ cần đổi chiều dữ liệu đầu vào. Có khả năng tổng quát hóa rất tốt sang số lượng biến chưa từng thấy trong quá trình huấn luyện.
- **Nhược điểm**: Nén toàn bộ cửa sổ lookback của một biến thành một vector duy nhất qua lớp Linear tuyến tính, có thể làm mất thông tin chi tiết về thứ tự thời gian. Độ phức tạp tính toán của Attention tăng theo bình phương số biến $O(N^2)$, gây tốn chi phí khi số biến $N$ quá lớn. Kém hiệu quả trên dữ liệu đơn biến (univariate).

## Khả năng áp dụng cho bài toán của nhóm
Rất khả thi và phù hợp vì bài toán của nhóm là dự báo chuỗi thời gian nhiều chiều (multivariate input). iTransformer có mã nguồn công khai và cấu trúc mạnh mẽ để khai thác tốt mối tương quan chéo giữa các biến phụ tải/nhiệt độ trong ETT.
