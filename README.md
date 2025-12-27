# Bird Classification – Kuş Türü Tanıma Web Uygulaması

Bird Classification, **EfficientNet-B4** derin öğrenme mimarisini kullanarak **200 farklı kuş türünü** görseller üzerinden otomatik olarak tanıyabilen bir Streamlit tabanlı web uygulamasıdır. Proje, bilgisayarlı görü ve derin öğrenme tekniklerini gerçek dünya senaryosuna uyarlamayı amaçlamaktadır.

## 🌟 Özellikler

*   **200 sınıflı kuş türü tanıma**
*   **EfficientNet-B4 tabanlı derin öğrenme modeli**
*   **Streamlit ile etkileşimli web arayüzü**
*   **Görsel yükleme ile anlık tahmin**
*   **Terminal üzerinden de tahmin yapabilme**

## 💻 Kullanılan Teknolojiler

*   Python
*   PyTorch
*   EfficientNet-B4
*   Streamlit
*   NumPy, Pillow, torchvision

## 📋 Gereksinimler

Projeyi çalıştırabilmek için aşağıdaki gereksinimlerin sisteminizde kurulu olması gerekir:

*   Python 3.9 veya üzeri
*   pip (Python paket yöneticisi)

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1. Proje Dizinine Girin
```bash
cd ~/Desktop/clean_project
```

### 2. Sanal Ortam Oluşturun
```bash
python3 -m venv venv
```

### 3. Sanal Ortamı Aktif Edin
```bash
source venv/bin/activate
```
*Terminal başında `(venv)` ifadesini görmelisiniz.*

### 4. pip Güncellemesi
```bash
pip install --upgrade pip
```

### 5. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 6. Web Uygulamasını Başlatın
```bash
streamlit run web_app.py
```

Uygulama varsayılan olarak tarayıcıda aşağıdaki adreste açılır:
👉 **http://localhost:8501** (veya 8502)

---

## � Kullanım

### Web Arayüzü
1.  Web arayüzü üzerinden bir kuş görseli yükleyin.
2.  Model, yüklenen görseli analiz eder.
3.  Tahmin edilen kuş türü ekranda gösterilir.

### Terminal Üzerinden Tahmin
Web arayüzü dışında, terminalden de tahmin alabilirsiniz:

```bash
python tahmin_et.py
```
*(Eğer script parametre alıyorsa: `python tahmin_et.py ornek_resim.jpg`)*

---

## 📂 Proje Yapısı

```
.
├── web_app.py        # Streamlit web arayüzü
├── tahmin_et.py      # Terminal tabanlı tahmin scripti
├── egitim_modeli.py  # Model eğitim kodları (Google Colab)
├── model_son.pth     # Eğitilmiş model dosyası
├── requirements.txt  # Python bağımlılıkları
└── README.md
```

## 🧠 Model Eğitimi Hakkında

*   Model, Google Colab ortamında eğitilmiştir.
*   EfficientNet-B4 mimarisi kullanılmıştır.
*   Eğitim kodları `egitim_modeli.py` dosyasında yer almaktadır.
*   Eğitim veri seti 200 farklı kuş türünden oluşmaktadır.
*   **Not:** Model ağırlıkları (`model_son.pth`) doğrudan kullanıma hazırdır.

## 🗂️ Model ve Test Verileri (indirilebilir bağlantılar)

- **Eğitilmiş model (Google Drive):** https://drive.google.com/file/d/1ernu1ppbuette0XA_ZvIA5NyziPMLJFf/view?usp=sharing
	- İndirdikten sonra proje kök dizinine `model_son.pth` olarak yerleştiriniz.
	- Alternatif olarak `gdown` ile terminalden indirebilirsiniz:
		```bash
		pip install gdown
		gdown 'https://drive.google.com/uc?id=1ernu1ppbuette0XA_ZvIA5NyziPMLJFf' -O model_son.pth
		```

- **Model testi için örnek dosyalar (test.zip - Google Drive):** https://drive.google.com/file/d/1UMvLlCE7DeyIpzpa7g9HkHG5FJSA7Qky/view?usp=sharing
	- İndirip açtıktan sonra test görsellerini örneğin `tests/` dizinine koyabilirsiniz.
	- Terminalde `unzip test.zip -d tests/` ile açabilirsiniz.

- **Eğitim veri seti (Kaggle):** https://www.kaggle.com/datasets/kedarsai/bird-species-classification-220-categories
	- Kaggle üzerinden indirip eğitim/deneme işlemleri için kullanabilirsiniz.


## ℹ️ Notlar

Çalışmayı bitirdiğinizde sanal ortamdan çıkmak için:
```bash
deactivate
```

Uygulamayı tekrar çalıştırmak için yalnızca:
1.  Sanal ortamı aktif etmeniz (`source venv/bin/activate`)
2.  `streamlit run web_app.py` komutunu çalıştırmanız yeterlidir.

## ⚖️ Lisans

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir. Ticari kullanım için uygun değildir.
