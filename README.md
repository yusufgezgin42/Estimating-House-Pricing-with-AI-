# İstanbul Konut Fiyat Tahmini ve Haber Tabanlı Öneri Sistemi

Bu proje, **İstanbul’daki konut fiyatlarını tahmin eden** ve **Türkiye’deki ekonomik/gayrimenkul haberlerini analiz ederek yatırım önerileri üreten** yapay zeka tabanlı bir sistemdir.  

Amaç, hem sayısal (fiyat tahmini) hem de metinsel (haber analizi) veriyi birleştirerek,
kullanıcıya daha bütüncül bir bakış açısıyla **“Bu dönemde konut almak mantıklı mı?”** sorusuna cevap verebilmektir.

---

## 🔍 Projenin Ana Bileşenleri

Proje iki ana modülden oluşur:

### 1️⃣ İstanbul Konut Fiyat Tahmin Sistemi

Bu modül, İstanbul’daki konutlar için aşağıdaki özellikleri kullanarak **tahmini satış fiyatı** üretir:

- Konum bilgisi (ilçe, mahalle vb.)
- Metrekare (net/brüt)
- Bina yaşı
- Oda sayısı
- Kat bilgisi
- Konut tipi ve diğer emlak özellikleri
- İlgili bölgedeki geçmiş fiyat verileri

Makine öğrenmesi tabanlı bir model (regresyon) kullanılarak,
girdi olarak verilen özelliklerden **konutun tahmini fiyatı** hesaplanır.

Bu sayede:
- Fiyatı olması gerekenden çok yüksek/çok düşük görünen ilanlar tespit edilebilir,
- Yatırım amaçlı alınacak daireler için **bölgeler arası karşılaştırma** yapılabilir,
- İstanbul içindeki fiyat dinamikleri veri temelli olarak incelenebilir.

---

### 2️⃣ Türkiye Gündemine Göre Haber Tabanlı Öneri Sistemi

Bu modül, özellikle **BloombergHT** gibi kaynaklardan çekilen Türkiye ekonomisi, konut piyasası, faiz, enflasyon ve gayrimenkul ile ilgili haberleri analiz eder.

Sistem:

- Haberleri otomatik olarak web’den çeker,
- Metinleri temizler ve özetler,
- Doğal dil işleme (NLP) yöntemleri ile:
  - **Duygu analizi (sentiment)** yapar (olumlu / olumsuz / nötr),
  - **Anahtar kelimeleri** çıkarır (faiz, enflasyon, konut fiyat endeksi, talep vb.),
  - Konut piyasasını etkileyen kritik ifadeleri tespit eder.

Bu analizlerin sonucunda sistem:

- Haberlerden gelen sinyallere göre **AL / SAT / TUT** önerisi üretir,
- Kısa vadeli (3–6 ay) konut piyasası eğilimi hakkında yorum yapar,
- Risk unsurlarını listeler (aşırı değerlenme, yüksek artış oranları, reel değer kaybı vb.).

Örnek çıktılar:
- Duygu dağılımı (pozitif / negatif / nötr oranları),
- Etkin olan kurallar (düşük faiz ortamı, arz azlığı, enflasyona karşı koruma vb.),
- Toplam puana göre **“AL” veya “DİKKATLİ OL”** gibi yorumlar.

---

## 🧠 Kullanılan Temel Teknolojiler

- **Python 3**
- **Web scraping:** `requests`, `BeautifulSoup`
- **Doğal Dil İşleme (NLP):**
  - Metin temizleme ve ön işleme
  - Duygu analizi
  - Anahtar kelime çıkarımı
- **Makine Öğrenmesi:**
  - Konut fiyat tahmini için regresyon tabanlı modeller
- **Veri Formatı:**
  - Sonuçların dışa aktarımı için `JSON`

---

## 🚀 Çalıştırma (Örnek)

```bash
python deneme2.py
