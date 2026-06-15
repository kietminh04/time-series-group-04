# Paper 2: TimesFM

## Details
- **Title**: A Decoder-Only Foundation Model for Time-Series Forecasting
- **Authors**: Google Research, et al.
- **Year**: 2024 (ICML 2024)
- **Link/Reference**: [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
- **Keywords**: Foundation model, decoder-only, zero-shot

## Summary
### Vấn đề nghiên cứu
Sự thành công của các mô hình nền tảng (Foundation Models) trong xử lý ngôn ngữ tự nhiên (NLP) chưa được tái hiện tương đương trong lĩnh vực chuỗi thời gian, nơi phần lớn các mô hình vẫn là dạng chuyên biệt (task-specific) và được huấn luyện từ đầu trên tập dữ liệu nhỏ.

### Ý tưởng chính & Mô hình đề xuất
Đề xuất **TimesFM** - một mô hình nền tảng Decoder-only sử dụng cấu trúc causal Transformer. Dữ liệu được chia thành các phân đoạn (patching) và mô hình học cách dự báo tự hồi quy (autoregressive) trên tập dữ liệu huấn luyện trước khổng lồ chứa hơn 100 tỷ điểm dữ liệu thực tế từ các nguồn công cộng lẫn dữ liệu tìm kiếm.

### Kết quả chính
- Đạt hiệu năng dự báo Zero-shot (dự báo ngay lập tức) xuất sắc trên nhiều bộ dữ liệu benchmark khác nhau (như Monash, M4, ETT).
- Vượt qua mô hình iTransformer ở chế độ Zero-shot.
- Mô hình có kích thước gọn nhẹ (chỉ 200 triệu tham số), giúp thời gian suy luận (inference) cực kỳ nhanh.

### Điểm mạnh & Hạn chế
- **Ưu điểm**: Nhẹ hơn nhiều so với các mô hình nền tảng khác như Timer (1.3 tỷ tham số) nên có thể chạy suy luận dễ dàng trên các máy cấu hình yếu. Checkpoint được Google công bố công khai, khả năng sử dụng thực tế rất cao.
- **Nhược điểm**: Giới hạn độ dài ngữ cảnh lịch sử đầu vào (context length) là 512 bước, gây hạn chế với các chuỗi cần lịch sử rất dài (như chu kỳ mùa vụ năm). Bản gốc chủ yếu hỗ trợ chuỗi đơn biến (univariate), chưa xử lý trực tiếp các biến phụ thuộc bổ sung (covariates). Kiến trúc Decoder-only không có Encoder nên kém ổn định hơn với dữ liệu có nhiều nhiễu và non-stationary.

## Khả năng áp dụng cho bài toán của nhóm
Rất cao. Vì Google đã công bố rộng rãi mã nguồn và checkpoint của TimesFM, nhóm có thể dễ dàng tải mô hình về chạy suy luận Zero-shot trên tập dữ liệu phụ tải điện của nhóm để so sánh hiệu quả dự báo của một mô hình nền tảng lớn so với các mô hình baseline truyền thống.
