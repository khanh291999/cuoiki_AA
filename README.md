# Graph Subsampling Demo

Demo trực quan 7 thuật toán Graph Subsampling cho môn Giải Thuật Nâng Cao.

## 🎯 Giới thiệu

Ứng dụng web Flask để minh họa và so sánh 7 thuật toán lấy mẫu đồ thị (Graph Subsampling):

### 7 Thuật toán

1. **RN (Random Node)** - Chọn ngẫu nhiên các node
2. **RE (Random Edge)** - Chọn ngẫu nhiên các cạnh và lấy 2 đầu mút
3. **DN (Degree Node)** - Chọn node theo xác suất tỷ lệ với degree
4. **BFS (Breadth-First Search)** - Duyệt theo chiều rộng
5. **DFS (Depth-First Search)** - Duyệt theo chiều sâu
6. **RW (Random Walk)** - Đi bộ ngẫu nhiên trên đồ thị
7. **RNN (Random Node-Neighbor)** - Chọn node và các hàng xóm

### 3 Loại đồ thị

- **Random Graph** - Đồ thị Erdős-Rényi ngẫu nhiên
- **Community Graph** - Đồ thị với 2 communities rõ ràng
- **Core-Periphery** - Đồ thị có cấu trúc core-periphery

## 🚀 Cài đặt

### Yêu cầu
- Python 3.8+
- pip

### Các bước cài đặt

```bash
# Clone hoặc download project

# Tạo virtual environment
python -m venv .venv

# Kích hoạt virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📦 Dependencies

- Flask 3.1.2
- NetworkX 3.2.1
- NumPy 2.4.0
- (xem đầy đủ trong requirements.txt)

## ▶️ Chạy ứng dụng

```bash
# Đảm bảo virtual environment đã được kích hoạt
python app_simple.py
```

Mở trình duyệt và truy cập: **http://127.0.0.1:5000**

## 💡 Tính năng

- ✅ Trực quan hóa đồ thị gốc và subgraph
- ✅ Hiển thị từng bước thực thi thuật toán
- ✅ So sánh hiệu quả của các thuật toán
- ✅ Tùy chỉnh tham số: n (số node), qn (số node cần lấy), seed
- ✅ 3 loại đồ thị khác nhau để demo
- ✅ Đảm bảo **đúng qn nodes** (không vượt, không thiếu)

## 🎓 Sử dụng cho học tập

### Tham số gợi ý:
- **n**: 20-50 nodes (dễ quan sát)
- **qn**: 10-25 nodes (khoảng 30-50% của n)
- **seed**: giữ cố định để reproduce kết quả

### Lưu ý khi demo:
- Chọn thuật toán phù hợp với loại đồ thị
- Community Graph → BFS/DFS dễ thấy được cấu trúc
- Core-Periphery → DN có xu hướng chọn core nodes
- Random Graph → so sánh công bằng các thuật toán

## 📝 Cấu trúc project

```
.
├── app_simple.py          # Flask app chính (7 thuật toán)
├── requirements.txt       # Python dependencies
├── README.md             # File này
├── static/
│   └── style.css         # CSS styling
└── templates/
    └── demo.html         # HTML template chính
```

## 🔧 Chi tiết kỹ thuật

### Độ phức tạp thuật toán:
- **RN**: O(qn) - nhanh nhất
- **RE**: O(attempts × |E|) - phụ thuộc cấu trúc đồ thị  
- **DN**: O(|V| + qn) - với numpy optimized
- **BFS/DFS**: O(|V| + |E|) - duyệt đồ thị
- **RW**: O(steps × avg_degree) - có thể chậm
- **RNN**: O(attempts × avg_degree) - phụ thuộc cấu trúc

### Đảm bảo chất lượng:
- ✅ Tất cả thuật toán trả về **đúng qn nodes**
- ✅ Logic xử lý đồ thị không liên thông
- ✅ Weighted sampling WITHOUT replacement (DN)
- ✅ Kiểm soát không vượt quá qn (RE, RNN)

## 👨‍💻 Tác giả

Đồ án môn Giải Thuật Nâng Cao - UTE

## 📄 License

Sử dụng cho mục đích học tập.
