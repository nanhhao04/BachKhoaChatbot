# BachKhoa Support Chatbot - Hướng dẫn chạy

## 1. Clone repository
```bash
git clone https://github.com/your-repo/bachkhoa-chatbot.git
cd bachkhoa-chatbot
```

## 2. Tạo môi trường ảo

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/MacOS
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Cài đặt thư viện
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Tạo file .env
```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=student_support_chatbot

USE_LOCAL_LLM=false
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=your_groq_api_key_here

RETRIEVER_TYPE=enhanced
```

## 5. Khởi động Docker
```bash
docker-compose up -d
docker-compose ps
```

## 6. Chuẩn bị dữ liệu
```
Đặt file PDF vào: data/pdfs/
```

## 7. Tiền xử lý dữ liệu

### Tạo collection mới
```bash
python data_processor.py --pdf-dir data/pdfs --action create
```

### Thêm dữ liệu
```bash
python data_processor.py --pdf-dir data/pdfs --action add
```

### Xem thông tin
```bash
python data_processor.py --action info
```

## 8. Chạy Streamlit
```bash
streamlit run app_streamlit.py
```

## 9. Truy cập
```
http://localhost:8501
```