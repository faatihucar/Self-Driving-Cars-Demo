# 🚗 Otonom Sürüş Görsel Algılama Paketi (Self-Driving Cars Suite) 🏎️💨

[![GitHub Stars](https://img.shields.io/badge/YOLO-v11-00FFFF.svg?style=for-the-badge&logo=yolo)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Bilgisayarlı%20Görü-5C3EE8.svg?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Aktif-orange.svg?style=for-the-badge&logo=git-lfs)](https://git-lfs.github.com/)
[![License: MIT](https://img.shields.io/badge/Lisans-MIT-yellow.svg?style=for-the-badge)](LICENSE)

Otonom araçlar ve gelişmiş sürücü destek sistemleri (ADAS) için geliştirilmiş; **gerçek zamanlı nesne tespiti**, **geleneksel ve derin öğrenme tabanlı şerit tespiti** ile **sürülebilir alan yol segmentasyonunu** bir araya getiren kapsamlı bilgisayarlı görü ve derin öğrenme algılama paketi.

---

## 🌟 Temel Algılama Yetenekleri

| Modül | Kullanılan Yöntem / Model | Birincil Hedef | Gerçek Zamanlı Hız (FPS) |
| :--- | :--- | :--- | :---: |
| **1. Trafik Nesneleri & Yaya Tespiti** | YOLO11 Large (`yolo11l.pt`) | Araç, Yaya, Bisiklet, Kamyon, Otobüs, Trafik Işıkları | ~45-60+ FPS |
| **2. Klasik Şerit Tespiti** | OpenCV IPM + HSV + Kayan Pencereler | Geometrik şerit eğriliği ve merkez kayması takibi | ~80-120 FPS |
| **3. Derin Öğrenme Şerit Segmentasyonu** | YOLO11 Large Segment (`yolo11l-seg.pt`) | Zorlu ışık ve gölge altında piksel düzeyinde şerit maskesi | ~35-50 FPS |
| **4. Sürülebilir Alan Segmentasyonu** | YOLO11 Large Segment (`yolo11l-seg.pt`) | Sürülebilir yol yüzeyi maskeleme ve güvenli sürüş rotası | ~35-50 FPS |

---

## 📸 Çıkarım (Inference) ve Model Sonuçları

### 1️⃣ Araç ve Yaya Çoklu Sınıf Tespiti
Yoldaki dinamik aktörler için yüksek doğruluklu gerçek zamanlı sınırlayıcı kutu (bounding box) tespiti, sınıflandırma ve hareketli ortalama FPS gösterimi.

<div align="center">
  <img src="assets/car_person_detection_demo.jpg" alt="Araç ve Yaya Tespiti Demosu" width="850"/>
  <p><i>Şekil 1: Güven skoru, sınıflandırma ve anlık FPS göstergeli araç ve yaya tespiti.</i></p>
</div>

---

### 2️⃣ Klasik Şerit Tespiti (OpenCV & IPM Perspektif Dönüşümü)
Ters Perspektif Eşleme (**Inverse Perspective Mapping - IPM**) ile Kuş Bakışı (Bird's-Eye View) dönüşümü, dinamik HSV renk uzayı filtreleme ve histogram tepe noktası kayan pencere (sliding window) algoritması.

<div align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <b>1. İlgilenilen Alan (ROI)</b><br/>
        <img src="assets/opencv_roi.jpg" alt="ROI Seçimi" width="270"/>
      </td>
      <td align="center" width="33%">
        <b>2. Kuş Bakışı Görünüm (IPM)</b><br/>
        <img src="assets/opencv_birdseye.jpg" alt="Kuş Bakışı Dönüşüm" width="270"/>
      </td>
      <td align="center" width="33%">
        <b>3. Kayan Pencere Takibi</b><br/>
        <img src="assets/opencv_sliding_window.jpg" alt="Kayan Pencere Algoritması" width="270"/>
      </td>
    </tr>
  </table>
  <p><i>Şekil 2: Perspektif dönüşüm hattı: Trapezoidal ROI koordinatları &rarr; IPM Kuş Bakışı &rarr; Kayan pencere merkez izleme.</i></p>
</div>

---

### 3️⃣ Derin Öğrenme Tabanlı Şerit Segmentasyonu (YOLO11-Seg)
Özel şerit veri seti üzerinde **250 epoch** boyunca eğitilmiş, karmaşık ışıklandırma koşulları ve gölgelerde bile yüksek keskinlikte şerit çizgisi maskesi çıkaran model.

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <b>Doğrulama (Validation) Toplu Tahminleri</b><br/>
        <img src="assets/lane_val_pred.jpg" alt="Şerit Doğrulama Tahminleri" width="420"/>
      </td>
      <td align="center" width="50%">
        <b>Video Üzerinde Canlı Çıkarım</b><br/>
        <img src="assets/yolo_lane_segmentation_demo.jpg" alt="Canlı Şerit Segmentasyonu" width="420"/>
      </td>
    </tr>
  </table>
  <p><i>Şekil 3: YOLO11-seg şerit segmentasyonu doğrulama çıktısı (sol) ve gerçek sürüş videosu çıkarım sonucu (sağ).</i></p>
</div>

---

### 4️⃣ Sürülebilir Yol Alanı Segmentasyonu (YOLO11-Seg)
Araçların güvenle hareket edebileceği yol yüzeyini arka plandan ve kaldırımlardan ayırt etmek için **100 epoch** eğitilmiş sürülebilir alan segmentasyon modeli.

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <b>Sürülebilir Alan Maskesi</b><br/>
        <img src="assets/yolo_road_segmentation_demo.jpg" alt="Yol Yüzeyi Segmentasyonu" width="420"/>
      </td>
      <td align="center" width="50%">
        <b>Şehir İçi Ortam Testi</b><br/>
        <img src="assets/road_inference_people.jpg" alt="Şehir İçi Yol Segmentasyonu" width="420"/>
      </td>
    </tr>
  </table>
  <p><i>Şekil 4: Otoyol ve yoğun şehir içi sahnelerinde gerçek zamanlı sürülebilir yol yüzeyi segmentasyonu.</i></p>
</div>

---

## 📊 Eğitim Başarımı ve Değerlendirme Grafikleri

### 📈 Kayıp (Loss) ve Metrik İlerlemesi

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <b>Şerit Segmentasyonu Eğitimi (250 Epoch)</b><br/>
        <img src="assets/lane_training_results.png" alt="Şerit Eğitimi Sonuçları" width="420"/>
      </td>
      <td align="center" width="50%">
        <b>Yol Yüzeyi Eğitimi (100 Epoch)</b><br/>
        <img src="assets/road_training_results.png" alt="Yol Yüzeyi Eğitimi Sonuçları" width="420"/>
      </td>
    </tr>
    <tr>
      <td align="center" width="50%">
        <b>Şerit Karmaşıklık Matrisi (Confusion Matrix)</b><br/>
        <img src="assets/lane_confusion_matrix.png" alt="Şerit Karmaşıklık Matrisi" width="320"/>
      </td>
      <td align="center" width="50%">
        <b>Yol Karmaşıklık Matrisi (Confusion Matrix)</b><br/>
        <img src="assets/road_confusion_matrix.png" alt="Yol Karmaşıklık Matrisi" width="320"/>
      </td>
    </tr>
  </table>
</div>

---

## 📁 Proje Dizin Yapısı

```text
Self-Driving-Cars-Demo/
├── 📂 Car&Person Detection/
│   ├── 📜 detect.py                   # YOLO11 gerçek zamanlı tespit & video kaydetme betiği
│   ├── 📜 coco_classes.txt            # COCO veri seti sınıf listesi tanımları
│   ├── 📂 models/
│   │   └── 📦 yolo11l.pt              # Önceden eğitilmiş YOLO11 Large ağırlıkları (Git LFS)
│   ├── 📂 inference/                  # Test sürüşü giriş videoları
│   └── 📂 results/                    # İşlenmiş ve FPS bilgisi eklenmiş sonuç videosu
│
├── 📂 Lane Detection with OpenCV/
│   ├── 📜 detect_lane.py              # Kuş bakışı IPM ve kayan pencere şerit izleme algoritması
│   ├── 📜 roi_selector.py             # İnteraktif ROI koordinat kalibrasyon aracı
│   ├── 🖼️ coordinates.png             # IPM perspektif noktaları görsel kılavuzu
│   └── 📂 test_videos/                # Test sürüş dashcam videoları
│
├── 📂 Lane Detection with YOLO/
│   ├── 📓 Lane segmentation.ipynb     # Uçtan uca eğitim, doğrulama ve çıkarım jupyter defteri
│   ├── 📂 data/
│   │   ├── 📜 config.yaml             # Veri seti yolları ve sınıf tanımları
│   │   └── 📦 dataset.zip             # Etiketli şerit segmentasyon veri seti (Git LFS)
│   └── 📂 runs/segment/               # Kontrol noktaları, ağırlıklar (best.pt) ve metrikler
│
├── 📂 Roading Segmentation (Driveable Area)/
│   ├── 📓 Road Segmentation.ipynb     # Sürülebilir alan segmentasyon modeli eğitim defteri
│   ├── 📂 data/
│   │   ├── 📜 data.yaml               # Yol yüzeyi veri seti yapılandırması
│   │   └── 📦 road_surface_dataset.zip# Yüksek çözünürlüklü yol veri seti (Git LFS)
│   └── 📂 runs/segment/               # Eğitim logları, best.pt ağırlıkları ve tahmin çıktıları
│
├── 📂 assets/                         # Dokümantasyon görselleri, grafikler ve test kareleri
├── 📜 .gitattributes                  # Büyük model ağırlıkları ve videolar için Git LFS kuralları
├── 📜 .gitignore                      # Python/Jupyter temizleme yapılandırması
├── 📜 requirements.txt                # Gerekli Python kütüphaneleri
└── 📜 README.md                       # Ana dokümantasyon dosyası
```

---

## 🚀 Kurulum ve Başlangıç

### 1. Depoyu Klonlama (Git LFS ile)
Depoyu indirmeden önce sisteminizde [Git LFS](https://git-lfs.com/)'nin kurulu olduğundan emin olun:

```bash
# Git LFS eklentisini aktifleştirin
git lfs install

# Depoyu klonlayın
git clone https://github.com/faatihucar/Self-Driving-Cars-Demo.git
cd Self-Driving-Cars-Demo

# Büyük model ağırlıklarını ve videoları çekin
git lfs pull
```

### 2. Sanal Ortam Oluşturma ve Bağımlılıkları Yükleme
```bash
# Sanal ortam oluşturun
python -m venv venv

# Sanal ortamı aktifleştirin
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

---

## 💻 Modüllerin Kullanımı ve Çalıştırma

### 🚗 1. Araç ve Yaya Tespitini Başlatma
```bash
cd "Car&Person Detection"
python detect.py
```
> Canlı önizlemeyi kapatmak için `q` tuşuna basabilirsiniz. İşlenen video otomatik olarak `results/test_vid_res.avi` konumuna kaydedilecektir.

### 🛣️ 2. OpenCV Klasik Şerit Tespitini Başlatma
```bash
cd "Lane Detection with OpenCV"
python detect_lane.py
```
> Açılan penceredeki etkileşimli HSV ayar çubuklarını (`L-H`, `L-S`, `L-V`, `U-H`, `U-S`, `U-V`) kullanarak farklı asfalt ve ışık tonlarına göre renk filtrelemesini anlık ayarlayabilirsiniz. Çıkmak için `ESC` tuşuna basın.

### 🧠 3. YOLO Şerit veya Yol Segmentasyonu Çıkarımı
Eğitilmiş PyTorch modelleri ile doğrudan komut satırından çıkarım yapabilirsiniz:

```bash
# Şerit Segmentasyonu Çıkarımı
yolo segment predict model="Lane Detection with YOLO/runs/segment/yolov11_lane_segmentation/weights/best.pt" source="Lane Detection with OpenCV/test_videos/road.mp4" show_labels=False show_boxes=False

# Sürülebilir Yol Alanı Segmentasyonu Çıkarımı
yolo segment predict model="Roading Segmentation (Driveable Area)/runs/segment/yolov11_road_segmentation2/weights/best.pt" source="Roading Segmentation (Driveable Area)/inference/road.mp4" show_labels=False show_boxes=False
```

Veya Jupyter Notebook dosyalarını açarak eğitim/çıkarım adımlarını adım adım çalıştırabilirsiniz:
- `Lane Detection with YOLO/Lane segmentation.ipynb`
- `Roading Segmentation (Driveable Area)/Road Segmentation.ipynb`

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

- **Derin Öğrenme:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), [PyTorch](https://pytorch.org/), Torchvision
- **Bilgisayarlı Görü:** [OpenCV (cv2)](https://opencv.org/), NumPy, Imutils
- **Veri Görselleştirme:** Matplotlib, Seaborn, Pandas
- **Sürüm Kontrolü & Depolama:** Git Large File Storage (Git LFS)

---

## 👨‍💻 Geliştirici & İletişim

Geliştirici: **[Fatih Uçar](https://github.com/faatihucar)**

Otonom sürüş algoritmalarıyla ilgili soru sormak, katkıda bulunmak veya hata bildirmek için issues veya pull request açabilirsiniz!
