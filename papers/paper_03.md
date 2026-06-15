# Paper 3: Timer

## Details
- **Title**: Timer: Generative Pre-trained Transformers Are Large Time Series Models
- **Authors**: Yongqiang Tang, et al.
- **Year**: 2024 (ICML 2024)
- **Link/Reference**: [arXiv:2402.02368](https://arxiv.org/abs/2402.02368)
- **Keywords**: Foundation model, generative, large-scale pre-training

## Summary
### Vấn đề nghiên cứu
Hầu hết các mô hình chuỗi thời gian hiện nay đều là mô hình chuyên biệt cho từng tác vụ (task-specific), đòi hỏi phải huấn luyện lại từ đầu cho từng bộ dữ liệu mới. Lĩnh vực chuỗi thời gian vẫn thiếu các mô hình nền tảng (Foundation Models) có khả năng dự báo tổng quát giống như LLMs trong NLP.

### Ý tưởng chính & Mô hình đề xuất
Đề xuất **Timer** - một mô hình nền tảng dạng Generative (GPT-style) được huấn luyện trước (Pre-trained) trên tập dữ liệu UTSD khổng lồ (chứa hàng tỷ điểm dữ liệu từ nhiều nguồn khác nhau). Mô hình sử dụng kiến trúc Decoder-only với cơ chế Causal Attention và kỹ thuật chia patch đầu vào. Nó học cách dự báo tự hồi quy (autoregressive) để tổng quát hóa sang bất kỳ chuỗi thời gian nào.

### Kết quả chính
- Đạt hiệu năng dự báo Zero-shot (dự báo ngay mà không cần huấn luyện lại) cực kỳ ấn tượng, tiệm cận với các mô hình chuyên biệt được huấn luyện trực tiếp trên tập dữ liệu đó.
- Trong kịch bản Few-shot (chỉ tinh chỉnh với lượng mẫu rất nhỏ), Timer đạt kết quả vượt trội và thiết lập SOTA mới.
- Vượt qua iTransformer trong các thử nghiệm Zero-shot.

### Điểm mạnh & Hạn chế
- **Ưu điểm**: Chỉ cần huấn luyện trước một lần, sau đó có thể áp dụng trực tiếp lên các bộ dữ liệu mới để dự báo (Zero-shot) mà không cần huấn luyện thêm. Rất mạnh khi huấn luyện Few-shot trên dữ liệu nhỏ.
- **Nhược điểm**: Kích thước mô hình lớn (1.3 tỷ tham số), đòi hỏi GPU dung lượng lớn để huấn luyện lại; GPU thông thường (8-12 GB) chỉ chạy được inference. Phương pháp dự báo tự hồi quy (autoregressive) có thể làm tích lũy sai số khi dự báo các horizon rất dài. Hiệu năng Zero-shot thỉnh thoảng vẫn thua mô hình chuyên biệt được tinh chỉnh tốt.

## Khả năng áp dụng cho bài toán của nhóm
Trung bình đến Khá. Việc huấn luyện lại mô hình này là quá lớn cho tài nguyên của nhóm, tuy nhiên chúng ta có thể sử dụng các checkpoint công khai để suy luận Zero-shot trên tập dữ liệu ETT của nhóm để đánh giá khả năng tổng quát hóa của mô hình lớn so với các mô hình nhỏ tự huấn luyện.
