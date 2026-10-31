# Movie Recommendation System using Deep Learning (NCF)

## 1. Giới thiệu
Đồ án xây dựng hệ thống gợi ý phim sử dụng Deep Learning với mô hình Neural Collaborative Filtering (NCF).

## 2. Dataset
Sử dụng bộ dữ liệu MovieLens Latest Small:
- 100.836 lượt đánh giá.
- 610 người dùng.
- 9.724 bộ phim.

Tải dữ liệu tại: https://grouplens.org/datasets/movielens/latest/

Giải nén và đặt thư mục `ml-latest-small` đúng đường dẫn mà chương trình yêu cầu.

## 3. Cài đặt môi trường

Cài Python, sau đó mở Terminal tại thư mục dự án và chạy:

```bash
pip install -r requirements.txt
```

## 4. Chạy chương trình

Chạy chương trình huấn luyện và đánh giá:

```bash
python Movie_Recommendation_Deep_Learning.py
```

Chạy giao diện gợi ý phim:

```bash
python Giao_dien_goi_y_phim_Colab.py
```

Lưu ý: Lệnh chạy giao diện cần điều chỉnh theo cách triển khai thực tế của chương trình.

## 5. Kết quả
Hệ thống dự đoán mức độ yêu thích phim của người dùng và đề xuất danh sách Top 10 phim phù hợp.

## 6. Công nghệ sử dụng
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 7. Nhóm thực hiện
Tên nhóm: [Điền tên nhóm]

Thành viên: [Điền tên thành viên]

Môn học: CMP169 - Trí tuệ nhân tạo
