# Phát hiện giao dịch ngân hàng gian lận

## Mục tiêu
Xây dựng mô hình phân loại giao dịch gian lận, ưu tiên **recall** của lớp gian lận
(bắt được càng nhiều vụ gian lận càng tốt).

## Dữ liệu
- 9.998 giao dịch, chỉ có 68 ca gian lận (0,68%) nên dữ liệu mất cân bằng nặng.
- Gian lận chỉ xuất hiện ở hai loại giao dịch: `TRANSFER` (33 ca) và `CASH_OUT` (35 ca).
- Cột `isFlaggedFraud` toàn bằng 0 nên không có giá trị sử dụng.
- Bốn cột số dư bị lỗi định dạng (dạng `1,01E+16`), 67 đến 354 ô mỗi cột.
  Các ô này được chuyển thành NaN rồi thay bằng median.

## Phương pháp
1. Làm sạch dữ liệu, tách train/test (80/20, `random_state=42`).
2. Tiền xử lý: median + StandardScaler cho cột số, OneHotEncoder cho cột `type`.
3. Mô hình SVC với `class_weight="balanced"`.
4. Tune hyperparameter bằng GridSearchCV (4-fold, chọn theo F2):
   `C=100, gamma=scale, kernel=rbf`.

## Kết quả trên tập test (2.000 giao dịch, 16 ca gian lận)

| Mô hình | Recall | Precision | F2 |
|---|---|---|---|
| SVC mặc định | 0,000 | 0,000 | 0,000 |
| SVC balanced | 0,812 | 0,081 | 0,290 |
| SVC đã tune | 0,688 | 0,196 | 0,458 |

Mô hình đã tune bắt được 11 trong 16 vụ gian lận, bỏ sót 5, báo nhầm 45.

## Hạn chế
- Tập test chỉ có 16 ca gian lận nên các chỉ số dao động mạnh (lệch 1 ca khoảng 6% recall).
- Precision còn thấp (khoảng 20%), phù hợp để lọc sơ bộ chứ chưa phù hợp để tự động chặn giao dịch.
- Một phần dữ liệu tiền bị lỗi và được thay bằng median.

## Cấu trúc thư mục

```
Banking__analysis/
├── data/
│   └── raw/
│       └── banking_transactions.csv   # dữ liệu gốc
├── notebooks/
│   └── analysis_banking.ipynb         # notebook phân tích
├── models/
│   └── fraud_svc.joblib               # mô hình đã huấn luyện
├── reports/
│   ├── figures/
│   │   └── confusion_matrix.png       # biểu đồ ma trận nhầm lẫn
│   ├── evaluation_report.csv          # bảng chỉ số đánh giá
│   └── Report_bank.html               # báo cáo profiling dữ liệu
└── README.md
```

## Cách dùng mô hình đã lưu

```python
import joblib
model = joblib.load("models/fraud_svc.joblib")
pred = model.predict(new_data)
```

`new_data` phải có đúng các cột như dữ liệu huấn luyện: `step`, `type`, `amount`,
`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFlaggedFraud`
(không có `nameOrig`, `nameDest`, `isFraud`).