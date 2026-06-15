# Paper 2: PatchTST

## Details
- **Title**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers
- **Authors**: Yuqi Nie, et al.
- **Year**: 2022 (ICLR 2023)
- **Link/Reference**: [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- **Keywords**: Patch-based, channel independence, self-supervised

## Summary
### Vấn đề nghiên cứu
Kiến trúc Transformer truyền thống gặp vấn đề độ phức tạp tính toán và bộ nhớ lớn bậc $O(L^2)$ theo chiều dài chuỗi $L$. Điều này khiến mô hình gặp khó khăn khi xử lý dữ liệu lịch sử dài (lookback window lớn) để dự báo dài hạn.

### Ý tưởng chính & Mô hình đề xuất
Đề xuất hai kỹ thuật cốt lõi:
1. **Patching**: Chia chuỗi thời gian thành các phân đoạn nhỏ (patches, ví dụ độ dài 16 bước) làm token đầu vào. Việc này giúp giảm số lượng token đi khoảng 16 lần, qua đó giảm độ phức tạp tính toán và lưu giữ được cấu trúc lân cận cục bộ tốt hơn.
2. **Channel Independence (Độc lập kênh)**: Xử lý mỗi biến độc lập như một chuỗi đơn biến, chia sẻ chung trọng số của Transformer để tránh nhiễu và tăng độ ổn định của việc huấn luyện.

### Kết quả chính
- Giảm sai số trung bình (MSE) khoảng 21% trên các bộ dữ liệu tiêu chuẩn.
- Đạt hiệu năng vượt trội so với Autoformer, Informer, DLinear và N-BEATS.
- Tiết kiệm bộ nhớ GPU và thời gian chạy đáng kể nhờ số lượng token nhỏ.

### Điểm mạnh & Hạn chế
- **Ưu điểm**: Giảm chiều dài chuỗi token đầu vào giúp huấn luyện nhẹ và nhanh hơn. Hỗ trợ học tự giám sát (Self-supervised learning) bằng cách che đi một phần các patch rồi dự báo lại chúng để tận dụng dữ liệu chưa được gán nhãn.
- **Nhược điểm**: Bỏ qua mối quan hệ tương quan chéo trực tiếp giữa các biến số (do cơ chế Channel Independence), làm giảm hiệu năng nếu các biến phụ thuộc chặt chẽ vào nhau. Kích thước patch phải chỉnh tay. Cần cửa sổ lịch sử dài mới phát huy tốt sức mạnh.

## Khả năng áp dụng cho bài toán của nhóm
Rất cao. PatchTST hiện đang được tích hợp rộng rãi trong các thư viện dự báo chuỗi thời gian. Mô hình này rất phù hợp làm baseline Transformer nâng cao của nhóm để so sánh trực tiếp với DLinear hoặc LSTM.
