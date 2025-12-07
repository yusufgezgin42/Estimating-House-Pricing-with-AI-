import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import numpy as np
from datetime import datetime
import warnings
from typing import List, Dict, Tuple, Optional
import json
warnings.filterwarnings('ignore')

# ==================== BLOOMBERGHT CRAWLER KODU ====================

BASE_URL = "https://www.bloomberght.com"
NEWS_LIST_URL = f"{BASE_URL}/haberler"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TITLE_KEYWORDS = [
    "konut",
    "gayrimenkul",
    "kira",
    "konut fiyat",
    "konut kredisi",
    "kfe",  # konut fiyat endeksi
]

COUNTRY_CITY_KEYWORDS = [
    "türkiye",
    "istanbul",
    "Türkiye Cumhuriyet Merkez Bankası (TCMB)"
]

def fetch_listing_html():
    resp = requests.get(NEWS_LIST_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.text

def fetch_article_raw(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def is_tr_istanbul_related(soup: BeautifulSoup) -> bool:
    """Haber gövdesinde Türkiye / İstanbul geçiyor mu?"""
    full_text = soup.get_text(" ", strip=True).lower()
    return any(kw in full_text for kw in COUNTRY_CITY_KEYWORDS)

def find_tr_istanbul_real_estate_links(max_results=20):
    html = fetch_listing_html()
    soup = BeautifulSoup(html, "html.parser")

    candidates = []

    for a in soup.find_all("a"):
        title = a.get_text(strip=True)
        href = a.get("href")

        if not title or not href:
            continue

        t_lower = title.lower()

        # Başlıkta konutla ilgili bir şey yoksa geç
        if not any(kw in t_lower for kw in TITLE_KEYWORDS):
            continue

        # URL tam hale getir
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = urljoin(BASE_URL, href)

        if not href.startswith(BASE_URL):
            continue

        candidates.append({"title": title, "url": href})

    # URL bazlı uniq
    uniq = {}
    for item in candidates:
        uniq[item["url"]] = item["title"]
    deduped = [{"url": u, "title": t} for u, t in uniq.items()]

    # Şimdi her biri için sayfa çek, "Türkiye/İstanbul içermeyenleri" at
    filtered = []
    for item in deduped:
        try:
            soup_article = fetch_article_raw(item["url"])
        except Exception as e:
            print("Hata (ön kontrol):", e)
            continue

        if is_tr_istanbul_related(soup_article):
            filtered.append(item)

        if len(filtered) >= max_results:
            break
        time.sleep(0.7)

    return filtered

def parse_article_text(soup: BeautifulSoup) -> str:
    """Paragrafları birleştirerek temiz bir metin döndür."""
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)
    return text

def extract_tr_ist_features_from_text(text: str) -> dict:
    """
    BloombergHT konut haberi metninden Türkiye geneli ve İstanbul için
    bazı sayısal özellikleri çekmeye çalışır.
    """
    t = text.lower()
    features = {
        "tr_yoy_change": None,
        "tr_mom_change": None,
        "tr_index_level": None,
        "ist_yoy_change": None,
        "ist_mom_change": None,
    }

    # 1) Türkiye için yıllık artış (yüzde 32,2 / %32,2 / yüzde 32.2)
    # pattern: "türkiye genelinde ... yüzde xx,x" veya "bir önceki yılın aynı ayına göre ... yüzde xx,x"
    m_tr_yoy = re.search(
        r"(türkiye(?: genelinde)?|konut fiyat endeksi).*?(yüzde|%)[\s]*([\d]+[.,]\d+)",
        t
    )
    if not m_tr_yoy:
        # yedek: "bir önceki yılın aynı ayına göre nominal olarak yüzde 32,2"
        m_tr_yoy = re.search(
            r"bir önceki yılın aynı ayına göre.*?(yüzde|%)[\s]*([\d]+[.,]\d+)",
            t
        )
        if m_tr_yoy:
            value = m_tr_yoy.group(2)
            features["tr_yoy_change"] = float(value.replace(",", "."))
    else:
        value = m_tr_yoy.group(3)
        features["tr_yoy_change"] = float(value.replace(",", "."))

    # 2) Türkiye için aylık artış (bir önceki aya göre yüzde 1,7 artan KFE)
    m_tr_mom = re.search(
        r"bir önceki aya göre.*?(yüzde|%)[\s]*([\d]+[.,]\d+)\s*oranında artan kfe",
        t
    )
    if m_tr_mom:
        value = m_tr_mom.group(2)
        features["tr_mom_change"] = float(value.replace(",", "."))

    # 3) KFE seviye (195,7 seviyesine yükseldi)
    m_index = re.search(
        r"kfe.*?([\d]+[.,]\d+)\s*seviyesine",
        t
    )
    if m_index:
        value = m_index.group(1)
        features["tr_index_level"] = float(value.replace(",", "."))

    # 4) İstanbul için yıllık artış (haberlerde genelde "istanbul'da yıllık artış %xx,x")
    m_ist_yoy = re.search(
        r"istanbul.*?(yıllık|yıl bazında).*?(yüzde|%)[\s]*([\d]+[.,]\d+)",
        t
    )
    if m_ist_yoy:
        value = m_ist_yoy.group(3)
        features["ist_yoy_change"] = float(value.replace(",", "."))

    # 5) İstanbul için aylık artış (daha nadir ama koyalım)
    m_ist_mom = re.search(
        r"istanbul.*?bir önceki aya göre.*?(yüzde|%)[\s]*([\d]+[.,]\d+)",
        t
    )
    if m_ist_mom:
        value = m_ist_mom.group(2)
        features["ist_mom_change"] = float(value.replace(",", "."))

    return features

def crawl_bloomberght_konut_tr_ist(max_results=3, delay_seconds=1.0):
    """
    1) BloombergHT haber listesinden sadece başlığı konutla ilgili
       ve metni Türkiye/İstanbul içeren haberleri bulur.
    2) En fazla 'max_results' kadar haber döndürür. (Varsayılan: 3)
    """
    links = find_tr_istanbul_real_estate_links(max_results=10)  # geniş tutuyoruz

    results = []
    for item in links:

        # ❗️ 3 HABER LİMİTİ
        if len(results) >= max_results:
            break

        url = item["url"]
        print(f"Haber çekiliyor: {item['title']} → {url}")

        try:
            soup = fetch_article_raw(url)

            # Başlık
            title_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else item["title"]

            # Giriş tarihi (çok kaba, ama iş görür)
            full_text = soup.get_text("\n", strip=True)
            giris = None
            m_giris = re.search(r"Giriş:\s*(.+)", full_text)
            if m_giris:
                giris = m_giris.group(1).strip()

            # Metin
            text = parse_article_text(soup)

            # Feature çıkar
            features = extract_tr_ist_features_from_text(text)

            results.append({
                "url": url,
                "title": title,
                "giris": giris,
                "text": text,
                "features": features
            })

        except Exception as e:
            print("Hata:", e)

        time.sleep(delay_seconds)

    return results

# ==================== NLP ANALİZ KODU ====================

# NLTK verilerini indir (ilk çalıştırmada gerekli)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class ImprovedTurkishNLPAnalyzer:
    def __init__(self):
        # Türkçe stopwords
        self.turkish_stopwords = set(stopwords.words('turkish'))
        # Ek stopwords ekleyelim
        self.turkish_stopwords.update([
            'bir', 've', 'ile', 'olarak', 'için', 'kadar', 'göre',
            'da', 'de', 'bu', 'şu', 'o', 'ise', 'mi', 'mı', 'mu', 'mü',
            'haber', 'haberi', 'haberler', 'bloomberg', 'ht', 'tcmb',
            'ise', 'iken', 'ile', 'idi', 'imiş', 'yok', 'var', 'dır',
            'dir', 'dur', 'dür', 'tır', 'tir', 'tur', 'tür'
        ])
        
        # Anahtar kelime kategorileri - GÜNCELLENMİŞ
        self.keyword_categories = {
            'faiz': ['faiz', 'faizi', 'faizler', 'faizleri', 'faiz oranı', 'faiz indirimi', 
                    'faiz artışı', 'politika faizi', 'referans faiz', 'tcmb'],
            
            'kredi': ['kredi', 'kredisi', 'krediler', 'konut kredisi', 'mortgage', 
                     'ipotek', 'kredi faizi', 'kredi oranı', 'kredi talebi'],
            
            'fiyat': ['fiyat', 'fiyatı', 'fiyatlar', 'fiyatları', 'konut fiyatı', 
                     'ev fiyatı', 'kira fiyatı', 'fiyat artışı', 'fiyat düşüşü',
                     'fiyat endeksi', 'kfe'],
            
            'enflasyon': ['enflasyon', 'enflasyonu', 'enflasyonda', 'enflasyonist',
                         'reel', 'nominal', 'enflasyon baskısı', 'reel değer'],
            
            'talep': ['talep', 'talebi', 'talep artışı', 'talepte', 'talep yönelimi',
                     'tüketici talebi', 'yatırımcı talebi', 'arttı', 'artış'],
            
            'arz': ['arz', 'arzı', 'arz azlığı', 'arz fazlası', 'arz-talep', 
                   'piyasa arzı', 'konut arzı'],
            
            'semt_bölge': ['istanbul', 'ankara', 'izmir', 'kadıköy', 'beşiktaş',
                          'şişli', 'avrupa yakası', 'anadolu yakası', 'bölge',
                          'semt', 'ilçe', 'mahalle'],
            
            'ekonomi': ['ekonomi', 'ekonomik', 'büyüme', 'gsyh', 'yatırım',
                       'piyasa', 'finans', 'ekonomi politikası', 'merkez bankası'],
            
            'değer': ['değer', 'değeri', 'değer artışı', 'değer kaybı', 'değerleme',
                     'değer saklama', 'yatırım değeri', 'reel değer'],
            
            'risk': ['risk', 'riski', 'riskler', 'risk faktörü', 'risk algısı',
                    'belirsizlik', 'volatilite', 'istikrar', 'kayıp', 'kaybı']
        }
        
        # Duygu yüklü kelimeler - GÜNCELLENMİŞ
        self.sentiment_words = {
            'pozitif': ['artış', 'yükseliş', 'kazanç', 'getiri', 'olumlu', 'iyi',
                       'güçlü', 'cazip', 'avantaj', 'fırsat', 'talep', 'büyüme',
                       'gelişme', 'iyileşme', 'kazandırıyor', 'kazançlı', 'artan',
                       'yükseldi', 'arttı', 'pozitif', 'yukarı', 'güçlü'],
            
            'negatif': ['düşüş', 'kayıp', 'zarar', 'risk', 'olumsuz', 'kötü',
                       'zayıf', 'tehlike', 'dezavantaj', 'tehdit', 'azalma',
                       'gerileme', 'kaybediyor', 'zararlı', 'düşük', 'kaybı',
                       'düştü', 'azaldı', 'negatif', 'aşağı', 'zayıf', 'kayıp'],
            
            'nötr': ['stabil', 'durağan', 'sabit', 'koruma', 'beklenti', 'tahmin',
                    'projeksiyon', 'öngörü', 'bekleniyor', 'nominal', 'reel',
                    'endeks', 'seviye', 'oran', 'yüzde']
        }
    
    def preprocess_text(self, text: str, use_stemming: bool = False) -> List[str]:
        """Metni temizle ve token'lara ayır"""
        # Küçük harfe çevir
        text = text.lower()
        
        # Özel karakterleri temizle (sayıları koru)
        text = re.sub(r'[^\w\s%.,]', ' ', text)
        
        try:
            # Tokenize
            tokens = word_tokenize(text, language='turkish')
        except:
            # Fallback: basit split
            tokens = text.split()
        
        # Stopwords'leri ve kısa kelimeleri kaldır
        tokens = [token for token in tokens 
                 if token not in self.turkish_stopwords 
                 and len(token) > 2 
                 and not token.isdigit()]
        
        # Kök bulmayı kaldırdık - çok agresifti
        # Sadece basit son ekleri kaldır
        if use_stemming:
            tokens = [self._simple_stem(token) for token in tokens]
        
        return tokens
    
    def _simple_stem(self, word: str) -> str:
        """Çok basit kök bulma - sadece son 'ler', 'lar', 'ı', 'i', 'u', 'ü' eklerini kaldır"""
        if len(word) <= 3:
            return word
        
        # Sadece çoğul eklerini kaldır
        if word.endswith(('ler', 'lar')):
            return word[:-3]
        if word.endswith(('ları', 'leri')):
            return word[:-4]
        
        return word
    
    def extract_keywords(self, text: str, top_n: int = 20) -> Dict:
        """Metinden anahtar kelimeler çıkar"""
        # Kök bulma OLMADAN
        tokens = self.preprocess_text(text, use_stemming=False)
        
        # Kelime frekansları
        word_freq = Counter(tokens)
        
        # Kategori bazlı anahtar kelimeleri bul
        category_keywords = {}
        text_lower = text.lower()
        
        for category, keywords in self.keyword_categories.items():
            found_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
            if found_keywords:
                category_keywords[category] = found_keywords
        
        # En sık geçen kelimeler
        top_keywords = dict(word_freq.most_common(top_n))
        
        return {
            'top_keywords': top_keywords,
            'category_keywords': category_keywords,
            'total_tokens': len(tokens),
            'unique_tokens': len(set(tokens))
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Gelişmiş duygu analizi"""
        try:
            sentences = sent_tokenize(text, language='turkish')
        except:
            # Basit cümle bölme
            sentences = re.split(r'[.!?]+', text)
        
        # Başlık sentiment'i için özel kontrol
        sentiment_scores = {
            'pozitif': 0,
            'negatif': 0,
            'nötr': 0
        }
        
        # Başlıkta negatif kelimeler varsa ek puan
        title_negatives = ['kayıp', 'kaybı', 'düşüş', 'zarar', 'kötü', 'olumsuz']
        for word in title_negatives:
            if word in text[:100].lower():  # İlk 100 karakter (başlık ve giriş)
                sentiment_scores['negatif'] += 2
        
        # Her cümle için sentiment analizi
        sentence_sentiments = []
        for sentence in sentences:
            if len(sentence.strip()) < 5:
                continue
                
            sentence_lower = sentence.lower()
            sentence_score = {'pozitif': 0, 'negatif': 0, 'nötr': 0}
            
            # Kelime bazlı sentiment
            for sentiment, words in self.sentiment_words.items():
                for word in words:
                    if word in sentence_lower:
                        sentence_score[sentiment] += 1
            
            # Finansal terimler için ek puan
            financial_positives = ['artış', 'artan', 'yükseliş', 'büyüme', 'olumlu']
            financial_negatives = ['düşüş', 'azalan', 'kayıp', 'kaybı', 'olumsuz']
            
            for word in financial_positives:
                if word in sentence_lower:
                    sentence_score['pozitif'] += 2
            
            for word in financial_negatives:
                if word in sentence_lower:
                    sentence_score['negatif'] += 2
            
            # Cümlenin dominant sentiment'i
            if sentence_score['pozitif'] > sentence_score['negatif']:
                sentence_sentiments.append('pozitif')
            elif sentence_score['negatif'] > sentence_score['pozitif']:
                sentence_sentiments.append('negatif')
            else:
                sentence_sentiments.append('nötr')
            
            # Toplam sentiment puanlarına ekle
            for sentiment in sentiment_scores:
                sentiment_scores[sentiment] += sentence_score[sentiment]
        
        # Toplam sentiment belirleme
        total = sum(sentiment_scores.values())
        if total > 0:
            sentiment_ratio = {k: v/total for k, v in sentiment_scores.items()}
            
            # Ağırlıklı sentiment belirle
            weighted_scores = {
                'pozitif': sentiment_scores['pozitif'] * 1.0,
                'negatif': sentiment_scores['negatif'] * 1.2,  # Negatife daha fazla ağırlık
                'nötr': sentiment_scores['nötr'] * 0.5
            }
            dominant_sentiment = max(weighted_scores.items(), key=lambda x: x[1])[0]
        else:
            sentiment_ratio = {'pozitif': 0, 'negatif': 0, 'nötr': 1}
            dominant_sentiment = 'nötr'
        
        return {
            'sentiment_scores': sentiment_scores,
            'sentiment_ratio': sentiment_ratio,
            'dominant_sentiment': dominant_sentiment,
            'sentence_sentiments': sentence_sentiments,
            'positive_sentences': len([s for s in sentence_sentiments if s == 'pozitif']),
            'negative_sentences': len([s for s in sentence_sentiments if s == 'negatif']),
            'neutral_sentences': len([s for s in sentence_sentiments if s == 'nötr'])
        }
    
    def extract_financial_entities(self, text: str) -> Dict:
        """Finansal varlıkları ve sayısal verileri çıkar - GELİŞTİRİLMİŞ"""
        # Gelişmiş yüzde oranları bulma
        percentages = []
        
        # Pattern 1: "yüzde 32,2" 
        matches1 = re.findall(r'yüzde\s+(\d+[.,]?\d*)', text.lower())
        percentages.extend([float(m.replace(',', '.')) for m in matches1])
        
        # Pattern 2: "%32,2"
        matches2 = re.findall(r'%(\d+[.,]?\d*)', text.lower())
        percentages.extend([float(m.replace(',', '.')) for m in matches2])
        
        # Pattern 3: "oranında 32,2"
        matches3 = re.findall(r'oranında\s+(\d+[.,]?\d*)', text.lower())
        percentages.extend([float(m.replace(',', '.')) for m in matches3])
        
        # Tüm sayıları bul (binlik, milyonluk değerler)
        all_numbers = re.findall(r'\b(\d+[.,]?\d*)\b', text)
        numbers = []
        for n in all_numbers:
            try:
                num = float(n.replace(',', '.'))
                if num > 1:  # 1'den büyük sayıları al
                    numbers.append(num)
            except:
                pass
        
        # Para birimleri
        currency_patterns = {
            'tl': r'(\d+[.,]?\d*)\s*(tl|₺|türk lirası)',
            'dolar': r'(\d+[.,]?\d*)\s*(\$|dolar|usd)',
            'euro': r'(\d+[.,]?\d*)\s*(€|euro|eur)'
        }
        
        currency_values = {}
        for currency, pattern in currency_patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                currency_values[currency] = [float(m[0].replace(',', '.')) for m in matches]
        
        # Zaman ifadeleri
        time_expressions = re.findall(
            r'(\d+\s*(ay|yıl|hafta|gün|saat)\s*(önce|içinde|sonra)?)', 
            text.lower()
        )
        
        # KFE (Konut Fiyat Endeksi) değerleri
        kfe_values = re.findall(r'kfe.*?(\d+[.,]?\d+)', text.lower())
        kfe_values = [float(v.replace(',', '.')) for v in kfe_values]
        
        return {
            'percentages': percentages,
            'significant_numbers': numbers[:20],
            'currency_values': currency_values,
            'time_expressions': time_expressions,
            'kfe_values': kfe_values,
            'max_percentage': max(percentages) if percentages else None,
            'min_percentage': min(percentages) if percentages else None,
            'avg_percentage': np.mean(percentages) if percentages else None,
            'total_percentages': len(percentages)
        }
    
    def analyze_temporal_context(self, text: str, publish_date: Optional[str] = None) -> Dict:
        """Zamansal bağlam analizi"""
        temporal_keywords = {
            'kısa_vade': ['kısa vadede', 'yakın dönemde', 'önümüzdeki ay', 'birkaç ay içinde',
                         '3 ay', '6 ay', 'kısa sürede', 'yakın zamanda'],
            
            'orta_vade': ['orta vadede', 'gelecek yıl', '12 ay', '1 yıl içinde',
                         'önümüzdeki yıl', 'orta dönemde'],
            
            'uzun_vade': ['uzun vadede', '2 yıl', '5 yıl', 'uzun dönemde',
                         'gelecek yıllarda', 'uzun süreli']
        }
        
        time_context = {}
        for period, keywords in temporal_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text.lower())
            time_context[period] = count
        
        # Yayın tarihi analizi
        date_analysis = {}
        if publish_date:
            try:
                # Tarihi parse et
                if 'giriş' in publish_date.lower():
                    date_str = publish_date.split(':')[-1].strip()
                else:
                    date_str = publish_date
                
                # Tarihi parse etmeye çalış
                date_formats = ['%d %B %Y', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d']
                parsed_date = None
                
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_str, fmt)
                        break
                    except:
                        continue
                
                if parsed_date:
                    date_analysis = {
                        'published_date': parsed_date.strftime('%Y-%m-%d'),
                        'days_ago': (datetime.now() - parsed_date).days,
                        'recency': 'recent' if (datetime.now() - parsed_date).days <= 7 else 'old'
                    }
            except:
                pass
        
        return {
            'time_context': time_context,
            'dominant_timeframe': max(time_context.items(), key=lambda x: x[1])[0] if any(time_context.values()) else 'bilinmiyor',
            'date_analysis': date_analysis
        }


class ImprovedHousingNewsAnalyzer:
    """Gelişmiş konut haberleri analiz sınıfı"""
    
    def __init__(self):
        self.nlp_analyzer = ImprovedTurkishNLPAnalyzer()
        
        # Gelişmiş karar kuralları
        self.decision_rules = {
            'K1': {
                'name': 'Düşük Kredi Faizi ve Talep Artışı',
                'keywords': ['faiz indirimi', 'faiz oranı', 'kredi oranı', 'talep artışı', 
                           'mortgage', 'tcmb', 'politika faizi'],
                'score': 3,
                'description': 'Düşük faiz ortamı ve artan talep'
            },
            'K2': {
                'name': 'Arz Azlığı ve Fiyat Artışı',
                'keywords': ['arz azlığı', 'fiyat artışı', 'konut fiyatı', 'kira artışı',
                           'konut fiyat endeksi', 'kfe', 'arttı', 'yükseldi'],
                'score': 2,
                'description': 'Arz kısıtlılığı fiyatları yukarı çekiyor'
            },
            'K3': {
                'name': 'Enflasyona Karşı Koruma',
                'keywords': ['enflasyon', 'değer saklama', 'yatırım aracı', 'koruma',
                           'reel değer', 'enflasyon baskısı'],
                'score': 1,
                'description': 'Gayrimenkul enflasyona karşı koruma sağlıyor'
            },
            'K4': {
                'name': 'Aşırı Değerlenme Riski',
                'keywords': ['aşırı değerlenme', 'balon', 'risk', 'düşüş riski',
                           'kayıp', 'kaybı', 'düşüş', 'zarar'],
                'score': -2,  # Daha güçlü negatif etki
                'description': 'Aşırı değerlenme riski mevcut'
            },
            'K5': {
                'name': 'İstanbul Özelinde Güçlü Performans',
                'keywords': ['istanbul', 'kadıköy', 'beşiktaş', 'avrupa yakası',
                           'anadolu yakası', 'semt', 'bölge'],
                'score': 1,
                'description': 'İstanbul özelinde güçlü performans'
            },
            'K6': {
                'name': 'Yüksek Artış Oranları',
                'keywords': ['yüzde', '%', 'oranında', 'artış', 'yükseliş'],
                'score': 1,
                'description': 'Yüksek yüzdelik artış oranları'
            }
        }
    
    def analyze_article(self, article: Dict) -> Dict:
        """Tek bir haberi kapsamlı analiz et"""
        
        text = article.get('text', '')
        title = article.get('title', '')
        publish_date = article.get('giris', '')
        
        print(f"\n📝 Analiz edilen metin özeti: {text[:200]}...")
        
        # 1. Anahtar kelime analizi
        keywords = self.nlp_analyzer.extract_keywords(text + ' ' + title)
        
        # 2. Duygu analizi
        sentiment = self.nlp_analyzer.analyze_sentiment(text)
        
        # 3. Finansal varlık çıkarma
        financial_entities = self.nlp_analyzer.extract_financial_entities(text)
        
        # 4. Zamansal bağlam analizi
        temporal_context = self.nlp_analyzer.analyze_temporal_context(text, publish_date)
        
        # 5. Karar kurallarını uygula
        rule_scores, rule_details = self.apply_decision_rules(text, keywords, financial_entities)
        total_score = sum(rule_scores.values())
        
        # 6. Öneri oluştur
        recommendation = self.generate_recommendation(total_score, rule_scores, sentiment, financial_entities)
        
        # 7. Risk analizi
        risk_analysis = self.analyze_risks(rule_scores, financial_entities, sentiment)
        
        # 8. Özet çıkar
        summary = self.generate_summary(title, sentiment, rule_scores, recommendation, rule_details)
        
        # 9. Sayısal analiz
        numerical_analysis = self.analyze_numerical_data(financial_entities)
        
        return {
            'article_info': {
                'title': title,
                'url': article.get('url'),
                'publish_date': publish_date,
                'features': article.get('features', {})
            },
            'nlp_analysis': {
                'keywords': keywords,
                'sentiment': sentiment,
                'financial_entities': financial_entities,
                'temporal_context': temporal_context,
                'numerical_analysis': numerical_analysis
            },
            'decision_analysis': {
                'rule_scores': rule_scores,
                'rule_details': rule_details,
                'total_score': total_score,
                'recommendation': recommendation,
                'risk_analysis': risk_analysis
            },
            'summary': summary
        }
    
    def apply_decision_rules(self, text: str, keywords: Dict, financial_entities: Dict) -> Tuple[Dict, Dict]:
        """Karar kurallarını uygula ve puanları hesapla"""
        rule_scores = {}
        rule_details = {}
        text_lower = text.lower()
        
        for rule_id, rule_info in self.decision_rules.items():
            score = 0
            triggered_keywords = []
            
            # Anahtar kelimeleri kontrol et
            for keyword in rule_info['keywords']:
                if keyword in text_lower:
                    score = rule_info['score']
                    triggered_keywords.append(keyword)
            
            # Kategori anahtar kelimelerini de kontrol et
            category_keywords = keywords.get('category_keywords', {})
            for category, words in category_keywords.items():
                for word in words:
                    if any(kw in word.lower() for kw in rule_info['keywords']):
                        score = rule_info['score']
                        triggered_keywords.append(word)
            
            # Özel kurallar
            if rule_id == 'K6':  # Yüksek artış oranları
                percentages = financial_entities.get('percentages', [])
                if any(p > 20 for p in percentages):  # %20'den yüksek artış
                    score = rule_info['score']
                    high_percentages = [p for p in percentages if p > 20]
                    triggered_keywords.append(f"Yüksek oranlar: {high_percentages}")
            
            rule_scores[rule_id] = score
            rule_details[rule_id] = {
                'name': rule_info['name'],
                'score': score,
                'triggered_keywords': triggered_keywords,
                'description': rule_info['description']
            }
        
        return rule_scores, rule_details
    
    def generate_recommendation(self, total_score: int, rule_scores: Dict, 
                               sentiment: Dict, financial_entities: Dict) -> Dict:
        """Toplam puana göre öneri oluştur"""
        
        # Finansal verilere göre ayarlama
        percentages = financial_entities.get('percentages', [])
        high_growth = any(p > 20 for p in percentages)
        
        # Temel öneri
        if total_score >= 4:
            base_recommendation = 'AL'
            confidence = 'yüksek'
        elif total_score >= 2:
            base_recommendation = 'TUT'
            confidence = 'orta'
        elif total_score >= 0:
            base_recommendation = 'DİKKATLİ TUT'
            confidence = 'düşük'
        else:
            base_recommendation = 'SAT/KAÇ'
            confidence = 'orta'
        
        # Duyguya göre ayarlama
        if sentiment['dominant_sentiment'] == 'pozitif':
            if base_recommendation == 'AL':
                confidence = 'çok yüksek'
            elif base_recommendation in ['TUT', 'DİKKATLİ TUT']:
                base_recommendation = 'TUT'
                confidence = 'orta-yüksek'
        elif sentiment['dominant_sentiment'] == 'negatif':
            if base_recommendation == 'AL':
                base_recommendation = 'DİKKATLİ TUT'
                confidence = 'düşük'
            elif base_recommendation == 'TUT':
                base_recommendation = 'DİKKATLİ TUT'
                confidence = 'orta'
        
        # Yüksek büyüme varsa daha agresif öneri
        if high_growth and base_recommendation in ['TUT', 'DİKKATLİ TUT']:
            base_recommendation = 'AL'
            confidence = 'yüksek'
        
        # Kural bazlı detaylar
        details = []
        if rule_scores.get('K1', 0) > 0:
            details.append("Düşük kredi faizleri alım için uygun ortam")
        if rule_scores.get('K2', 0) > 0:
            details.append("Arz azlığı fiyatları destekliyor")
        if rule_scores.get('K3', 0) > 0:
            details.append("Enflasyona karşı koruma özelliği var")
        if rule_scores.get('K4', 0) < 0:
            details.append("Aşırı değerlenme riski mevcut")
        if rule_scores.get('K5', 0) > 0:
            details.append("İstanbul özelinde güçlü performans")
        if rule_scores.get('K6', 0) > 0:
            details.append("Yüksek artış oranları gözleniyor")
        
        return {
            'action': base_recommendation,
            'confidence': confidence,
            'total_score': total_score,
            'details': details,
            'time_horizon': self.determine_time_horizon(rule_scores, sentiment, financial_entities)
        }
    
    def determine_time_horizon(self, rule_scores: Dict, sentiment: Dict, financial_entities: Dict) -> str:
        """Zaman dilimi belirle"""
        percentages = financial_entities.get('percentages', [])
        
        if rule_scores.get('K1', 0) > 0 or (percentages and max(percentages) > 30):
            return 'kısa vadeli (3-6 ay)'
        elif rule_scores.get('K3', 0) > 0:
            return 'uzun vadeli (1+ yıl)'
        else:
            return 'orta vadeli (6-12 ay)'
    
    def analyze_risks(self, rule_scores: Dict, financial_entities: Dict, sentiment: Dict) -> Dict:
        """Risk analizi yap"""
        risks = []
        
        # Aşırı değerlenme riski
        if rule_scores.get('K4', 0) < 0:
            risks.append({
                'type': 'Aşırı değerlenme',
                'level': 'yüksek',
                'description': 'Fiyatlar temel göstergelerin üzerinde seyrediyor'
            })
        
        # Yüksek yüzde artışları
        percentages = financial_entities.get('percentages', [])
        high_percentages = [p for p in percentages if p > 30]
        if high_percentages:
            risks.append({
                'type': 'Çok yüksek artış oranları',
                'level': 'orta-yüksek',
                'description': f'Bazı göstergelerde %{max(high_percentages):.1f} gibi çok yüksek artışlar'
            })
        
        # Negatif sentiment
        if sentiment['dominant_sentiment'] == 'negatif':
            risks.append({
                'type': 'Olumsuz piyasa sentimenti',
                'level': 'orta',
                'description': 'Haber tonu genel olarak olumsuz'
            })
        
        # Reel değer kaybı riski
        if 'reel değer kaybı' in financial_entities.get('text_preview', '').lower():
            risks.append({
                'type': 'Reel değer kaybı',
                'level': 'yüksek',
                'description': 'Enflasyon karşısında reel değer kaybı yaşanıyor'
            })
        
        return {
            'identified_risks': risks,
            'risk_level': 'yüksek' if any(r['level'] == 'yüksek' for r in risks) else 'orta' if any(r['level'] == 'orta' for r in risks) else 'düşük',
            'risk_count': len(risks)
        }
    
    def analyze_numerical_data(self, financial_entities: Dict) -> Dict:
        """Sayısal verileri analiz et"""
        percentages = financial_entities.get('percentages', [])
        
        if not percentages:
            return {'status': 'yetersiz_veri', 'message': 'Yeterli sayısal veri bulunamadı'}
        
        analysis = {
            'total_percentages': len(percentages),
            'max_percentage': max(percentages),
            'min_percentage': min(percentages),
            'avg_percentage': np.mean(percentages),
            'median_percentage': np.median(percentages)
        }
        
        # Büyüme analizi
        if analysis['avg_percentage'] > 20:
            analysis['growth_trend'] = 'çok_yüksek'
            analysis['growth_message'] = 'Çok yüksek büyüme oranları'
        elif analysis['avg_percentage'] > 10:
            analysis['growth_trend'] = 'yüksek'
            analysis['growth_message'] = 'Yüksek büyüme oranları'
        elif analysis['avg_percentage'] > 5:
            analysis['growth_trend'] = 'orta'
            analysis['growth_message'] = 'Orta düzeyde büyüme'
        else:
            analysis['growth_trend'] = 'düşük'
            analysis['growth_message'] = 'Düşük büyüme oranları'
        
        return analysis
    
    def generate_summary(self, title: str, sentiment: Dict, rule_scores: Dict, 
                        recommendation: Dict, rule_details: Dict) -> str:
        """Analiz özeti oluştur"""
        
        sentiment_map = {
            'pozitif': '📈 Olumlu',
            'negatif': '📉 Olumsuz', 
            'nötr': '⚖️ Tarafsız'
        }
        
        sentiment_desc = sentiment_map.get(sentiment['dominant_sentiment'], '⚖️ Tarafsız')
        
        active_rules = []
        for rule_id, score in rule_scores.items():
            if score != 0:
                rule_detail = rule_details.get(rule_id, {})
                active_rules.append(f"{rule_id}: {rule_detail.get('name', '')} ({score} puan)")
        
        # Detayları formatla
        details_text = ""
        if recommendation['details']:
            details_text = "\n📋 Detaylar:\n" + "\n".join([f"  • {d}" for d in recommendation['details']])
        
        summary = f"""
        📊 HABER ANALİZ ÖZETİ
        {'='*60}
        📰 Başlık: {title}
        🎭 Sentiment: {sentiment_desc} 
          - Pozitif: %{sentiment['sentiment_ratio']['pozitif']*100:.1f}
          - Negatif: %{sentiment['sentiment_ratio']['negatif']*100:.1f}
          - Nötr: %{sentiment['sentiment_ratio']['nötr']*100:.1f}
        
        🎯 Aktif Kurallar:
        {chr(10).join(['  • ' + r for r in active_rules]) if active_rules else '  • Hiçbir kural tetiklenmedi'}
        
        💰 Toplam Puan: {recommendation['total_score']}
        
        ⚡ ÖNERİ: 🟢 {recommendation['action']}
        🎯 Güven Düzeyi: {recommendation['confidence'].upper()}
        ⏰ Zaman Dilimi: {recommendation['time_horizon']}
        {details_text}
        """
        
        return summary


def run_analysis_on_article(article: Dict):
    """Tek bir haber için analiz çalıştır"""
    analyzer = ImprovedHousingNewsAnalyzer()
    
    print("🔍 Haber analiz ediliyor...")
    print(f"📰 Haber: {article.get('title')}")
    print(f"📅 Tarih: {article.get('giris', 'Bilinmiyor')}")
    print(f"🔗 URL: {article.get('url')}")
    print("-" * 80)
    
    # Analizi çalıştır
    analysis = analyzer.analyze_article(article)
    
    # Sonuçları göster
    print(analysis['summary'])
    
    # Detaylı bilgiler
    print("\n📊 DETAYLI ANALİZ:")
    print(f"Top 10 Anahtar Kelimeler:")
    for word, freq in list(analysis['nlp_analysis']['keywords']['top_keywords'].items())[:10]:
        print(f"  📌 {word}: {freq}")
    
    print(f"\n🏷️ Kategori Anahtar Kelimeleri:")
    cats = analysis['nlp_analysis']['keywords']['category_keywords']
    for category, words in cats.items():
        if words:
            print(f"  🔹 {category}: {', '.join(words[:3])}")
    
    print(f"\n📈 Finansal Veriler:")
    fin = analysis['nlp_analysis']['financial_entities']
    if fin['percentages']:
        print(f"  📊 Yüzde Oranları: {fin['percentages']}")
        print(f"  📈 En Yüksek: %{fin['max_percentage']:.1f}")
        print(f"  📉 En Düşük: %{fin['min_percentage']:.1f}")
        print(f"  ⚖️ Ortalama: %{fin['avg_percentage']:.1f}")
    else:
        print("  ❌ Finansal veri bulunamadı")
    
    print(f"\n🎭 Duygu Analizi Detayı:")
    sent = analysis['nlp_analysis']['sentiment']
    print(f"  😊 Pozitif Cümleler: {sent['positive_sentences']}")
    print(f"  ☹️ Negatif Cümleler: {sent['negative_sentences']}")
    print(f"  😐 Nötr Cümleler: {sent['neutral_sentences']}")
    
    print(f"\n⚖️ Karar Analizi Detayı:")
    for rule_id, detail in analysis['decision_analysis']['rule_details'].items():
        if detail['score'] != 0:
            print(f"  ✅ {detail['name']}: {detail['score']} puan")
            if detail['triggered_keywords']:
                print(f"     Tetikleyenler: {', '.join(detail['triggered_keywords'][:3])}")
    
    print(f"\n⚠️ Risk Analizi:")
    risks = analysis['decision_analysis']['risk_analysis']['identified_risks']
    if risks:
        for risk in risks:
            level_icon = '🔴' if risk['level'] == 'yüksek' else '🟡' if risk['level'] == 'orta' else '🟢'
            print(f"  {level_icon} {risk['type']} ({risk['level']}): {risk['description']}")
    else:
        print("  ✅ Belirgin risk bulunamadı")
    
    print("\n" + "="*80)
    
    return analysis


# Ana çalıştırma kodu
if __name__ == "__main__":
    print("🚀 GELİŞMİŞ BLOOMBERGHT KONUT HABER ANALİZ SİSTEMİ")
    print("="*80)
    
    # Önce haberleri çek
    print("\n📥 BloombergHT'den konut haberleri çekiliyor...")
    articles = crawl_bloomberght_konut_tr_ist(max_results=1)
    
    if articles:
        # İlk haberi analiz et
        article = articles[0]
        
        # NLP analizini çalıştır
        analysis_result = run_analysis_on_article(article)
        
        # JSON olarak kaydet
        with open('haber_analizi_detayli.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2, default=str)
        
        print("✅ Analiz tamamlandı ve 'haber_analizi_detayli.json' dosyasına kaydedildi.")
        
        # Senaryodaki formatta öneri oluştur
        print("\n" + "="*80)
        print("📋 SENARYO FORMATINDA ÖNERİ:")
        print("="*80)
        
        rec = analysis_result['decision_analysis']['recommendation']
        risks = analysis_result['decision_analysis']['risk_analysis']
        
        print(f"\nA. Ev Değeri Tahmini (Risk/Potansiyel)")
        print(f"   Toplam Etki Puanı: {rec['total_score']}")
        
        if rec['total_score'] >= 4:
            degerlendirme = "🚀 Güçlü Yükselme Potansiyeli"
        elif rec['total_score'] >= 2:
            degerlendirme = "📈 Orta Seviyede Yükselme Potansiyeli"
        elif rec['total_score'] >= 0:
            degerlendirme = "⚖️ Sınırlı Yükselme Potansiyeli"
        else:
            degerlendirme = "⚠️ Düşüş Riski Mevcut"
        
        print(f"   Değerlendirme: {degerlendirme}")
        print(f"   Risk Seviyesi: {risks['risk_level'].upper()}")
        
        if rec['details']:
            gerekce = " ".join(rec['details'][:2])
        else:
            gerekce = "Temel analiz göstergeleri sınırlı"
        
        print(f"   Gerekçe: {gerekce}")
        
        # Sayısal tahmin
        fin = analysis_result['nlp_analysis']['financial_entities']
        if fin.get('percentages'):
            avg_growth = fin['avg_percentage']
            tahmin = f"Önümüzdeki 6 ay içinde İstanbul genelinde %{avg_growth:.1f} - %{avg_growth+2:.1f} arası değer artışı beklentisi"
        else:
            tahmin = "Yeterli sayısal veri olmadığından tahmin yapılamıyor"
        
        print(f"   Tahmini Artış: {tahmin}")
        
        print(f"\nB. Öneri (Al/Sat/Tut)")
        print(f"   Öneri Kategorisi: {rec['action']}")
        print(f"   Güven Düzeyi: {rec['confidence'].upper()}")
        
        if rec['action'] == 'AL':
            aciklama = "ALIM için uygun bir zaman diliminde bulunuluyor. Düşük faiz ortamı ve artan talep fiyatları destekliyor."
        elif rec['action'] == 'TUT':
            aciklama = "TUTMAK mantıklı görünüyor. Piyasa dengeleri korunuyor, enflasyona karşı koruma özelliği devam ediyor."
        elif rec['action'] == 'DİKKATLİ TUT':
            aciklama = "DİKKATLİ TUTMAK önerilir. Bazı risk faktörleri mevcut, yakın takip gerekiyor."
        else:
            aciklama = "SAT/KAÇ önerilmektedir. Risk faktörleri baskın, korunma amaçlı hareket edilmeli."
        
        print(f"   Açıklama: {aciklama}")
        print(f"   Eylem Tavsiyesi: {rec['time_horizon']} perspektifle hareket edilmesi önerilir.")
        
        print(f"\nC. Risk Uyarıları:")
        if risks['identified_risks']:
            for risk in risks['identified_risks']:
                print(f"   ⚠️ {risk['type']}: {risk['description']}")
        else:
            print("   ✅ Önemli risk faktörü tespit edilmedi")
    else:
        print("❌ Analiz edilecek haber bulunamadı.")