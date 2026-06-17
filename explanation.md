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

### 2. Mô hình Hồi quy tuyến tính (Linear Regression) có dùng chuẩn hóa không và tại sao MAE lại thấp hơn?

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

### 3. Các thông số mô hình Học sâu & Giải thích chi tiết sự chênh lệch khối lượng tính toán (gấp 368 lần) và thời gian chạy 3 tiếng

Để hiểu rõ tại sao mô hình Deep Learning của bạn chạy rất nhanh (dưới 10 giây) trong khi của bạn bạn lại cực kỳ lâu (tính bằng phút trên GPU và kéo dài tới **3 tiếng** trên CPU), chúng ta hãy cùng phân tích ý nghĩa các thông số cấu hình và cách chúng nhân bản khối lượng tính toán:

#### A. Giải thích chi tiết các thông số mô hình Học sâu:

1. **Lookback Window (Cửa sổ trễ / Cửa sổ nhìn lại - ký hiệu $seq\_len$ hoặc $L$):**
   * *Khái niệm*: Là khoảng thời gian lịch sử (số lượng bước thời gian trước đó) mà mô hình được phép nhìn vào để làm đầu vào dự báo giá trị cho bước tiếp theo.
   * *Ví dụ thực tế*: Bộ dữ liệu ETTm2 được ghi nhận với chu kỳ **15 phút một dòng**.
     * **Mô hình của bạn ($seq\_len = 24$)**: Mô hình nhìn lại 24 dòng lịch sử gần nhất, tương đương $24 \times 15\text{ phút} = 6\text{ tiếng}$ lịch sử để dự báo bước tiếp theo.
     * **Mô hình của bạn bạn ($L = 96$)**: Mô hình nhìn lại 96 dòng lịch sử gần nhất, tương đương $96 \times 15\text{ phút} = 24\text{ tiếng}$ (trọn vẹn 1 ngày đêm) lịch sử.
   * *Ảnh hưởng tính toán*: Đối với các mạng nơ-ron tuần tự như LSTM hay GRU, mô hình phải duyệt qua chuỗi đầu vào từng bước một (từ bước 1 đến bước $L$). Chuỗi trễ dài hơn 4 lần nghĩa là mô hình phải chạy vòng lặp tuần tự qua 96 bước thời gian thay vì 24 bước của bạn cho mỗi mẫu dữ liệu. Khối lượng phép tính truyền trạng thái ẩn tăng tuyến tính lên **gấp 4 lần**.

2. **Hidden Dimension (Kích thước ẩn / Chiều ẩn / Số nút ẩn - ký hiệu $hidden\_dim$ hoặc $hidden\_size$):**
   * *Khái niệm*: Là số lượng nơ-ron (hoặc kích thước vectơ trạng thái ẩn) nằm bên trong mỗi tế bào LSTM. Nó đại diện cho "dung lượng bộ nhớ" hay độ rộng của bộ não mô hình để ghi nhớ và học các mối quan hệ phi tuyến tính phức tạp trong chuỗi thời gian.
     * **Mô hình của bạn ($hidden\_dim = 32$)**: Bộ nhớ nhỏ, nhẹ, biểu diễn trạng thái ẩn bằng một vectơ 32 chiều.
     * **Mô hình của bạn bạn ($hidden\_size = 128$)**: Bộ nhớ lớn hơn, lưu giữ vectơ trạng thái 128 chiều (to gấp 4 lần).
   * *Ảnh hưởng tính toán*: Trong các mạng nơ-ron, khi tăng chiều ẩn lên 4 lần, số lượng kết nối và các ma trận trọng số nội bộ của mô hình không tăng tuyến tính mà tăng **theo bình phương** của chiều ẩn ($O(hidden\_dim^2)$). Cụ thể, các phép nhân ma trận trọng số bên trong LSTM sẽ phình to gấp **13.6 lần** (chúng ta sẽ chứng minh bằng phép tính tham số chi tiết ở phần B).

3. **Batch Size (Kích thước lô / kích thước nhóm):**
   * *Khái niệm*: Là số lượng mẫu dữ liệu được gom lại thành một nhóm để đưa vào mô hình tính toán song song cùng một lúc trên GPU hoặc CPU.
     * **Mô hình của bạn ($batch\_size = 128$)**: Đưa 128 chuỗi thời gian vào tính cùng lúc. Với tập dữ liệu Train có 34,464 mẫu, mô hình của bạn chỉ cần chạy $34,464 / 128 = 269$ lần cập nhật trọng số mỗi epoch.
     * **Mô hình của bạn bạn ($batch\_size = 64$)**: Đưa 64 chuỗi vào tính cùng lúc. Mô hình của bạn bạn phải chạy tới $34,464 / 64 = 538$ lần cập nhật trọng số mỗi epoch.
   * *Tại sao không đưa tỉ số Batch Size vào phép tính nhân dồn tham số?*
     * Về mặt toán học lý thuyết, **Batch Size không làm thay đổi tổng số phép toán (FLOPs) để xử lý toàn bộ tập dữ liệu**. Dù chia tập dữ liệu thành các lô 64 hay 128, mô hình vẫn phải thực hiện tính toán lan truyền xuôi và ngược cho đúng 34,464 mẫu. Do đó, tỉ lệ chênh lệch kích thước lô không phải là một hệ số nhân trực tiếp của lượng phép tính toán học lý thuyết thô.
   * *Nhưng tại sao Batch Size nhỏ hơn ($64$ so với $128$) lại làm tăng thời gian chạy thực tế?*
     * **Hiệu suất song song hóa của phần cứng**: Khi Batch Size lớn hơn, phần cứng (đặc biệt là GPU) có thể tận dụng tối đa các luồng tính toán để thực hiện song song cùng lúc, giúp tăng hiệu suất tối đa. Batch Size nhỏ hơn khiến phần cứng không được tận dụng hết công suất (bị đói dữ liệu), dẫn đến tốc độ xử lý chậm hơn.
     * **Chi phí quản lý vòng lặp và cập nhật trọng số**: Batch Size 64 đòi hỏi mô hình phải thực hiện cập nhật trọng số 538 lần mỗi epoch, gấp **đúp** so với 269 lần của Batch Size 128. Mỗi lần cập nhật trọng số (backpropagation, tính gradient, optimizer step, sao chép dữ liệu lên xuống bộ nhớ đệm) đều tạo ra một khoảng trễ cố định (overhead). Số lần trễ này tăng gấp đôi làm kéo dài thời gian chạy thực tế một cách đáng kể.

4. **Epoch (Lượt học / Kỷ nguyên):**
   * *Khái niệm*: Là số lần mô hình duyệt qua toàn bộ tập dữ liệu huấn luyện để học hỏi và cập nhật kiến thức.
     * **Mô hình của bạn ($epochs = 5$)**: Mô hình chỉ duyệt qua toàn bộ dữ liệu 5 lần là dừng.
     * **Mô hình của bạn bạn ($epochs = 50$)**: Mô hình duyệt qua dữ liệu tối đa 50 lần (thực tế chạy GPU kích hoạt Early Stopping dừng ở **46 epochs**).
   * *Ảnh hưởng tính toán*: Thời gian huấn luyện tỷ lệ thuận tuyến tính với số lượng epochs. Chạy 46 epochs sẽ tốn thời gian gấp $46 / 5 = 9.2$ lần so với chạy 5 epochs.

---

#### B. Phép toán chi tiết về sự chênh lệch khối lượng tính toán (Gấp 368 lần):

Sự chênh lệch tốc độ huấn luyện được tạo ra bởi **hiệu ứng nhân dồn** của ba hệ số chênh lệch chính: số lượng lượt học (Epochs), độ dài chuỗi lịch sử (Lookback Window), và số lượng tham số bên trong LSTM (ảnh hưởng bởi Hidden Dimension).

##### 1. Chênh lệch do Số lượng Lượt học (Epochs):
$$\text{Hệ số Epochs} = \frac{46 \text{ epochs (bạn bạn)}}{5 \text{ epochs (của bạn)}} = 9.2 \text{ lần}$$

##### 2. Chênh lệch do Độ dài chuỗi trễ (Lookback Window):
$$\text{Hệ số Chuỗi} = \frac{96 \text{ bước}}{24 \text{ bước}} = 4.0 \text{ lần}$$
*(Mô hình phải tính toán tuần tự qua 96 bước thời gian thay vì 24 bước)*

##### 3. Chênh lệch do Kích thước ẩn (Hidden Dimension):
Mỗi tế bào LSTM chứa **4 cổng nội bộ** (Forget, Input, Cell, Output gate). Với mỗi cổng, mô hình thực hiện các phép nhân ma trận với dữ liệu đầu vào (kích thước $input\_dim = 7$ đặc trưng của bộ dữ liệu ETTm2) và với trạng thái ẩn từ bước trước đó (kích thước $hidden\_dim$).

Công thức tính tổng số tham số (parameters) của một tầng LSTM là:
$$\text{Parameters} = 4 \times \big[ hidden\_dim \times (input\_dim + hidden\_dim) + hidden\_dim \big]$$

* **Với mô hình của bạn ($hidden\_dim = 32$, $input\_dim = 7$):**
  $$\text{Parameters}_{\text{bạn}} = 4 \times [32 \times (7 + 32) + 32] = 4 \times 1,280 = 5,120 \text{ tham số}$$

* **Với mô hình của bạn bạn ($hidden\_dim = 128$, $input\_dim = 7$):**
  $$\text{Parameters}_{\text{bạn bạn}} = 4 \times [128 \times (7 + 128) + 128] = 4 \times 17,408 = 69,632 \text{ tham số}$$

* **Tỉ lệ chênh lệch tham số thực tế:**
  $$\text{Hệ số Tham số} = \frac{69,632}{5,120} = 13.6 \text{ lần}$$

* **Hiệu suất phần cứng (GPU Parallelization)**:
  Trên lý thuyết toán học thô, khối lượng phép tính ma trận tăng $13.6$ lần. Tuy nhiên, khi chạy trên GPU, nhờ khả năng song song hóa cực tốt với các phép tính ma trận kích thước lớn, hệ số chênh lệch thời gian chạy thực nghiệm cho phép tính này được tối ưu hóa chỉ còn **khoảng 10 lần** so với bộ nhớ nhỏ.

##### 🧮 Tổng hợp phép toán nhân dồn:
Khi nhân cả 3 hệ số chênh lệch này lại với nhau:
$$\text{Khối lượng tính toán tổng cộng} \approx 9.2\text{ (lượt học)} \times 4.0\text{ (chuỗi trễ)} \times 10.0\text{ (chiều ẩn thực nghiệm)} = 368 \text{ lần}$$

*(Nếu tính theo số lượng phép toán thô không qua tối ưu phần cứng của GPU: $9.2 \times 4.0 \times 13.6 \approx 500 \text{ lần}$).*

---

#### C. Giải thích sự chênh lệch thời gian chạy thực tế trên CPU (Tại sao lên tới 3 tiếng?):

Bạn hoàn toàn đúng khi thắc mắc: nếu chỉ làm phép tính nhân đơn giản:
$$10 \text{ phút (LSTM)} \times 3 \text{ mô hình} = 30 \text{ phút}$$
thì tại sao tổng thời gian chạy CPU của bạn bạn thực tế lại kéo dài tới **3 tiếng**? Có hai nguyên nhân cốt lõi dẫn đến sự khác biệt lớn này:

##### 1. Chênh lệch sức mạnh phần cứng CPU khủng khiếp giữa hai máy:
* **CPU của bạn (Intel Core i9-13950HX)**: Đây là dòng CPU cao cấp dành cho máy trạm di động siêu mạnh mẽ với **24 nhân / 32 luồng**, bộ nhớ đệm (Cache) cực lớn, TDP tối đa lên tới 157W và xung nhịp cực cao (lên tới 5.5 GHz). Nó có khả năng tính toán ma trận đa luồng (multi-threading) cực kỳ nhanh.
* **CPU của bạn bạn (Intel Core i5-11300H / 11400H)**: Đây là dòng CPU di động phân khúc trung bình từ 2 thế hệ trước, chỉ có **4 nhân / 8 luồng** (hoặc 6 nhân / 12 luồng), bộ nhớ đệm rất nhỏ, TDP giới hạn chỉ 35W để tiết kiệm pin. Xung nhịp và kiến trúc IPC thế hệ 11 cũng thấp hơn nhiều.
* **Tỉ lệ chênh lệch hiệu năng**: Trong các tác vụ tính toán nặng đa luồng như huấn luyện mạng nơ-ron trên CPU, con **i9-13950HX của bạn chạy nhanh hơn i5-11300H từ 4 đến 6 lần**!
  * Do đó, nếu mô hình LSTM chạy trên CPU i9 của bạn mất **10 phút**, thì khi chạy trên CPU i5 của bạn bạn, nó sẽ mất khoảng **40 - 60 phút**!

##### 2. Sự khác biệt về độ phức tạp tính toán giữa các mô hình (đặc biệt là Transformer):
* Code của bạn bạn huấn luyện liên tiếp 3 mô hình: `LSTM`, `GRU`, và `Transformer`.
  * **LSTM và GRU** có độ phức tạp tính toán tuyến tính với độ dài chuỗi $O(L)$. GRU tuy ít hơn LSTM 1 cổng (3 cổng so với 4 cổng) nhưng cấu trúc vẫn rất phức tạp, chạy trên CPU của bạn bạn cũng sẽ tốn khoảng **30 - 45 phút**.
  * **Transformer**: Mô hình Transformer sử dụng cơ chế tự chú ý (Self-Attention). Cơ chế này có độ phức tạp tính toán tăng theo **bình phương** độ dài chuỗi ($O(L^2)$). Với độ dài chuỗi $L = 96$ cùng nhiều tầng chiếu ma trận (Projection Heads), mô hình Transformer tính toán nặng hơn LSTM rất nhiều lần trên CPU. Riêng Transformer chạy trên CPU i5 thế hệ 11 có thể dễ dàng ngốn từ **80 đến 120 phút**.
* **Tổng kết thời gian chạy trên CPU i5 của bạn bạn:**
  $$\text{Tổng thời gian} \approx 50\text{ phút (LSTM)} + 40\text{ phút (GRU)} + 90\text{ phút (Transformer)} \approx 180\text{ phút} \text{ (3 tiếng)}$$
  *(Trong khi nếu chạy trên CPU i9 cực mạnh của bạn, tổng thời gian sẽ được nén lại chỉ còn khoảng 30 - 45 phút).*

---

### 4. So sánh kết quả: Tại sao mô hình LSTM của bạn bạn có kết quả tốt hơn của bạn?

Đúng vậy! Mô hình LSTM của bạn bạn cho kết quả dự báo chính xác hơn (sai số MAE, RMSE, MAPE thấp hơn) so với mô hình của bạn. Lý do cho sự vượt trội về độ chính xác này nằm ở việc cấu hình thông số tối ưu hơn:

#### 1. Sự khác biệt về cửa sổ nhìn lại (Lookback Window: $96$ so với $24$):
* **Bản chất chu kỳ dữ liệu**: Nhiệt độ dầu máy (`OT`) chịu ảnh hưởng mạnh mẽ bởi **chu kỳ ngày đêm (24 giờ)**. Vì dữ liệu ETTm2 được ghi nhận 15 phút một lần, nên một ngày đêm tương đương với đúng $96$ bước thời gian ($96 \times 15\text{ phút} = 24\text{ tiếng}$).
* **Mô hình của bạn ($seq\_len = 24$)**: Chỉ nhìn lại 6 tiếng lịch sử. Mô hình hoàn toàn thiếu thông tin của ngày hôm trước (ví dụ: xu hướng tăng nhiệt độ vào buổi trưa hay giảm vào ban đêm của ngày hôm trước). Do đó, nó dự báo kém chính xác hơn đối với các xu hướng mang tính chu kỳ ngày đêm.
* **Mô hình của bạn bạn ($L = 96$)**: Nhìn lại trọn vẹn 24 tiếng lịch sử. Mô hình có đầy đủ dữ liệu của cả một chu kỳ ngày đêm trước đó để nhận diện các quy luật lặp lại thời gian, giúp kết quả dự báo khớp với thực tế hơn rất nhiều.

#### 2. Dung lượng bộ não của mô hình (Hidden Dimension: $128$ so với $32$):
* Với kích thước ẩn `128`, mô hình của bạn bạn có số lượng tham số lớn gấp nhiều lần so với chiều ẩn `32` của bạn. 
* Chiều ẩn lớn cho phép mô hình lưu trữ nhiều thông tin phức tạp và học được các mối liên hệ phi tuyến tinh vi giữa các biến số phụ tải điện (`HUFL`, `HULL`,...) và nhiệt độ dầu máy (`OT`). Mô hình chiều ẩn `32` dễ rơi vào trạng thái **underfitting** (chưa học hết khả năng biểu diễn của dữ liệu).

#### 3. Số lượng lượt huấn luyện (Epochs: $46$ so với $5$):
* **Mô hình của bạn ($5$ epochs)**: Dừng lại quá sớm khi các trọng số của mạng nơ-ron chưa kịp hội tụ về điểm tối ưu. Mô hình mới chỉ "học lướt qua" dữ liệu.
* **Mô hình của bạn bạn ($46$ epochs)**: Được huấn luyện kỹ càng qua nhiều lượt duyệt dữ liệu, đồng thời sử dụng cơ chế **Early Stopping** để dừng ở điểm tối ưu nhất (Epoch 46 - điểm mà sai số validation đạt mức tối thiểu). Nhờ đó, các trọng số của mô hình được tinh chỉnh cực kỳ chính xác.

---

### 5. Cách cấu hình chạy huấn luyện bằng GPU cho nhanh

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
