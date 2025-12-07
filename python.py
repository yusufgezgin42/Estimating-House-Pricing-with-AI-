from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import json
import random
from typing import Dict, List, Optional, Tuple

app = Flask(__name__)
CORS(app)

# Global değişkenler
model = None
scaler = StandardScaler()
label_encoders = {}
feature_columns = []

# ==================== HABER ANALİZ MODÜLÜ ====================

class HaberAnalizSistemi:
    """Haberlerden piyasa analizi yapan sistem"""
    
    def __init__(self):
        self.haber_kaynaklari = [
            'https://www.emlakkulisi.com/',
            'https://www.emlaknews.com.tr/',
            'https://www.hurriyetemlak.com/haberler',
            'https://www.milliyet.com.tr/emlak/',
            'https://www.sozcu.com.tr/kategori/emlak/'
        ]
        
        # Anahtar kelimeler ve etkileri
        self.anahtar_kelimeler = {
            # Pozitif haberler
            'faiz indirimi': {'etki': 2, 'kategori': 'ekonomi'},
            'talep artışı': {'etki': 1.5, 'kategori': 'piyasa'},
            'konut kredisi': {'etki': 1, 'kategori': 'kredi'},
            'yatırım teşviği': {'etki': 1.5, 'kategori': 'ekonomi'},
            'altyapı yatırımı': {'etki': 1, 'kategori': 'altyapı'},
            'metro istasyonu': {'etki': 1.2, 'kategori': 'altyapı'},
            'proje geliştirme': {'etki': 1, 'kategori': 'proje'},
            'dış yatırım': {'etki': 1.5, 'kategori': 'yatırım'},
            
            # Negatif haberler
            'faiz artışı': {'etki': -2, 'kategori': 'ekonomi'},
            'talep düşüşü': {'etki': -1.5, 'kategori': 'piyasa'},
            'işsizlik artışı': {'etki': -1, 'kategori': 'ekonomi'},
            'enflasyon': {'etki': -0.5, 'kategori': 'ekonomi'},
            'inşaat maliyeti': {'etki': -0.8, 'kategori': 'maliyet'},
            'deprem riski': {'etki': -1.5, 'kategori': 'risk'},
            'arz fazlası': {'etki': -1, 'kategori': 'piyasa'},
            'kredi daralması': {'etki': -1.2, 'kategori': 'kredi'},
            
            # Nötr/ilçe spesifik
            'rezidans': {'etki': 0.5, 'kategori': 'proje'},
            'site': {'etki': 0.3, 'kategori': 'proje'},
            'toplu konut': {'etki': 0.2, 'kategori': 'proje'},
            'kentsel dönüşüm': {'etki': 0.8, 'kategori': 'dönüşüm'}
        }
        
        # İlçe bazlı haber ağırlıkları
        self.ilce_agirliklari = {
            'kadikoy': 1.2, 'besiktas': 1.3, 'sisli': 1.1, 'beyoglu': 1.1,
            'uskudar': 1.0, 'atasehir': 1.2, 'maltepe': 0.9, 'kartal': 0.9,
            'pendik': 0.8, 'umraniye': 0.9, 'esenler': 0.7, 'kagithane': 0.8,
            'fatih': 1.0, 'eminonu': 1.0, 'sariyer': 1.1, 'beylikduzu': 0.9,
            'buyukcekmece': 0.8, 'kucukcekmece': 0.8, 'avcilar': 0.7,
            'bahcelievler': 0.8, 'bakirkoy': 1.0, 'yesilkoy': 1.1,
            'zeytinburnu': 0.8, 'gungoren': 0.7, 'bayrampasa': 0.8,
            'gaziosmanpasa': 0.7, 'ortakoy': 1.2, 'etiler': 1.4,
            'nisantasi': 1.4, 'bebek': 1.3, 'levent': 1.2, 'maslak': 1.2
        }
    
    def haber_cek(self, ilce: str = None, limit: int = 10) -> List[Dict]:
        """Haberleri çek (simülasyon - gerçek uygulamada BeautifulSoup ile çekilecek)"""
        
        # Simüle edilmiş haberler (gerçek uygulamada web scraping yapılacak)
        bugun = datetime.now()
        haberler = []
        
        # Pozitif haber örnekleri
        pozitif_haberler = [
            {
                'baslik': f'{ilce.title() if ilce else "İstanbul"}\'da Konut Kredisi Faizleri Düştü',
                'icerik': f'{ilce.title() if ilce else "İstanbul"} bölgesinde konut kredisi faiz oranları yeni düzenlemeyle birlikte %1.5 seviyesine geriledi. Uzmanlar bu durumun talebi artıracağını belirtiyor.',
                'kaynak': 'EmlakKulisi',
                'tarih': (bugun - timedelta(days=1)).strftime('%d.%m.%Y'),
                'url': '#',
                'etki': 2.0,
                'kategori': 'kredi',
                'ilgili_ilce': ilce if ilce else 'genel'
            },
            {
                'baslik': 'Dev Altyapı Projesi İstanbul\'u Dönüştürüyor',
                'icerik': 'Yeni metro hatları ve ulaşım ağları ile İstanbul\'un birçok ilçesinde gayrimenkul değerleri artış bekleniyor.',
                'kaynak': 'EmlakNews',
                'tarih': (bugun - timedelta(days=2)).strftime('%d.%m.%Y'),
                'url': '#',
                'etki': 1.5,
                'kategori': 'altyapı',
                'ilgili_ilce': 'genel'
            },
            {
                'baslik': f'{ilce.title() if ilce else "İstanbul"} Bölgesinde Talep Patlaması',
                'icerik': f'{ilce.title() if ilce else "İstanbul"} bölgesinde konut talebi son 3 ayda %35 arttı. Uzmanlar fiyat artışlarının devam edeceğini öngörüyor.',
                'kaynak': 'HürriyetEmlak',
                'tarih': (bugun - timedelta(days=3)).strftime('%d.%m.%Y'),
                'url': '#',
                'etki': 1.8,
                'kategori': 'piyasa',
                'ilgili_ilce': ilce if ilce else 'genel'
            }
        ]
        
        # Negatif haber örnekleri
        negatif_haberler = [
            {
                'baslik': 'İnşaat Maliyetlerinde Rekor Artış',
                'icerik': 'Yapı malzemelerindeki fiyat artışları inşaat maliyetlerini tırmandırıyor. Yeni projelerde yavaşlama bekleniyor.',
                'kaynak': 'Milliyet',
                'tarih': (bugun - timedelta(days=4)).strftime('%d.%m.%Y'),
                'url': '#',
                'etki': -1.2,
                'kategori': 'maliyet',
                'ilgili_ilce': 'genel'
            },
            {
                'baslik': 'Kredi Daralması Emlak Piyasasını Etkileyebilir',
                'icerik': 'Bankaların konut kredisi verme kriterlerini sıkılaştırması bekleniyor. Bu durum talebi olumsuz etkileyebilir.',
                'kaynak': 'Sözcü',
                'tarih': (bugun - timedelta(days=5)).strftime('%d.%m.%Y'),
                'url': '#',
                'etki': -1.0,
                'kategori': 'kredi',
                'ilgili_ilce': 'genel'
            }
        ]
        
        # İlçeye özel haberler
        if ilce:
            ilce_haberleri = [
                {
                    'baslik': f'{ilce.title()} Kentsel Dönüşümde Öncelikli Bölge İlan Edildi',
                    'icerik': f'{ilce.title()} ilçesinde yeni kentsel dönüşüm projeleri için hazırlıklar başladı. Bölgedeki gayrimenkul değerlerinin artması bekleniyor.',
                    'kaynak': 'EmlakKulisi',
                    'tarih': (bugun - timedelta(days=2)).strftime('%d.%m.%Y'),
                    'url': '#',
                    'etki': 1.5,
                    'kategori': 'dönüşüm',
                    'ilgili_ilce': ilce
                },
                {
                    'baslik': f'{ilce.title()}\'de Yeni Rezidans Projesi',
                    'icerik': f'{ilce.title()} bölgesinde lüks rezidans projesi hayata geçiriliyor. Projenin bölge değerlerini artırması bekleniyor.',
                    'kaynak': 'EmlakNews',
                    'tarih': (bugun - timedelta(days=1)).strftime('%d.%m.%Y'),
                    'url': '#',
                    'etki': 1.0,
                    'kategori': 'proje',
                    'ilgili_ilce': ilce
                }
            ]
            haberler.extend(ilce_haberleri)
        
        # Karışık haber listesi oluştur
        haberler.extend(pozitif_haberler[:min(limit//2, len(pozitif_haberler))])
        haberler.extend(negatif_haberler[:min(limit//4, len(negatif_haberler))])
        
        # Rastgele sırala ve limit uygula
        random.shuffle(haberler)
        return haberler[:limit]
    
    def haber_analizi_yap(self, haber_listesi: List[Dict], ilce: str = None) -> Dict:
        """Haber analizi yap ve puan hesapla"""
        
        toplam_etki = 0
        pozitif_haber_sayisi = 0
        negatif_haber_sayisi = 0
        kategoriler = {}
        
        for haber in haber_listesi:
            etki = haber.get('etki', 0)
            toplam_etki += etki
            
            kategori = haber.get('kategori', 'diğer')
            if kategori not in kategoriler:
                kategoriler[kategori] = {'toplam': 0, 'sayi': 0}
            kategoriler[kategori]['toplam'] += etki
            kategoriler[kategori]['sayi'] += 1
            
            if etki > 0:
                pozitif_haber_sayisi += 1
            elif etki < 0:
                negatif_haber_sayisi += 1
        
        # İlçe ağırlığını uygula
        ilce_agirligi = self.ilce_agirliklari.get(ilce.lower() if ilce else 'ortalam', 1.0)
        toplam_etki *= ilce_agirligi
        
        # Ortalama etki ve yoğunluk
        ortalama_etki = toplam_etki / len(haber_listesi) if haber_listesi else 0
        haber_yogunlugu = len(haber_listesi) / 10  # Normalize edilmiş yoğunluk
        
        # Haber tabanlı puan hesapla (0-10 arası)
        haber_puani = 5 + (ortalama_etki * 2)  # 5 nötr puan, etkiye göre ayarla
        
        # Puan sınırlarını kontrol et
        haber_puani = max(0, min(10, haber_puani))
        
        return {
            'toplam_etki': toplam_etki,
            'ortalama_etki': ortalama_etki,
            'haber_puani': haber_puani,
            'pozitif_haber_sayisi': pozitif_haber_sayisi,
            'negatif_haber_sayisi': negatif_haber_sayisi,
            'nötr_haber_sayisi': len(haber_listesi) - pozitif_haber_sayisi - negatif_haber_sayisi,
            'haber_yogunlugu': haber_yogunlugu,
            'kategori_analizi': kategoriler,
            'ilce_agirligi': ilce_agirligi
        }

# ==================== KİŞİSELLEŞTİRİLMİŞ ÖNERİ SİSTEMİ ====================

class KisisellestirilmisOneriSistemi:
    """Haber verilerine dayalı kişiselleştirilmiş öneri sistemi"""
    
    def __init__(self):
        self.oneri_seviyeleri = {
            'acil_sat': {'skor': (0, 3), 'emoji': '🔴', 'oneri': 'ACİL SAT', 'aciklama': 'Yüksek risk, hemen satış yapın'},
            'sat': {'skor': (3, 5), 'emoji': '🟠', 'oneri': 'SAT', 'aciklama': 'Satış için uygun zaman'},
            'bekle': {'skor': (5, 6), 'emoji': '🟡', 'oneri': 'BEKLE/GÖZLE', 'aciklama': 'Karar için bekleyin'},
            'tut': {'skor': (6, 7), 'emoji': '🟢', 'oneri': 'TUT', 'aciklama': 'Değer artışı bekleniyor'},
            'iyi_tut': {'skor': (7, 8), 'emoji': '🟢', 'oneri': 'İYİ TUT', 'aciklama': 'Kesinlikle tutun'},
            'al': {'skor': (8, 9), 'emoji': '🔵', 'oneri': 'AL', 'aciklama': 'Alım için uygun zaman'},
            'acil_al': {'skor': (9, 10), 'emoji': '🔵', 'oneri': 'ACİL AL', 'aciklama': 'Fırsat kaçırmayın'}
        }
        
        # Kullanıcı tipleri ve ağırlıklar
        self.kullanici_tipleri = {
            'yatirimci': {'risk': 0.8, 'vade': 1.2, 'getiri': 1.3},
            'oturan': {'risk': 0.5, 'vade': 1.0, 'getiri': 1.0},
            'spekülatör': {'risk': 1.2, 'vade': 0.7, 'getiri': 1.5},
            'nakit_ihtiyaci': {'risk': 0.3, 'vade': 0.5, 'getiri': 0.8}
        }
    
    def kullanici_profili_analizi(self, kullanici_bilgileri: Dict) -> Dict:
        """Kullanıcı profilini analiz et"""
        
        profil = {
            'kullanici_tipi': kullanici_bilgileri.get('kullanici_tipi', 'oturan'),
            'yatirim_vadesi': kullanici_bilgileri.get('yatirim_vadesi', 'orta'),  # kısa/orta/uzun
            'risk_toleransi': kullanici_bilgileri.get('risk_toleransi', 'orta'),  # düşük/orta/yüksek
            'aciliyet': kullanici_bilgileri.get('aciliyet', 'yok'),  # yok/düşük/yüksek
            'hedef': kullanici_bilgileri.get('hedef', 'deger_koruma'),  # kar/kira/değer_koruma/nakit
            'alternatif_yatirim': kullanici_bilgileri.get('alternatif_yatirim', True)
        }
        
        # Puan ağırlıklarını hesapla
        tip_agirlik = self.kullanici_tipleri.get(profil['kullanici_tipi'], self.kullanici_tipleri['oturan'])
        
        # Risk ağırlığı
        if profil['risk_toleransi'] == 'yüksek':
            risk_agirlik = 1.2
        elif profil['risk_toleransi'] == 'düşük':
            risk_agirlik = 0.8
        else:
            risk_agirlik = 1.0
        
        # Vade ağırlığı
        if profil['yatirim_vadesi'] == 'uzun':
            vade_agirlik = 1.3
        elif profil['yatirim_vadesi'] == 'kısa':
            vade_agirlik = 0.7
        else:
            vade_agirlik = 1.0
        
        # Aciliyet ağırlığı
        if profil['aciliyet'] == 'yüksek':
            aciliyet_agirlik = 0.6
        elif profil['aciliyet'] == 'düşük':
            aciliyet_agirlik = 0.9
        else:
            aciliyet_agirlik = 1.0
        
        return {
            'profil': profil,
            'agirliklar': {
                'tip': tip_agirlik,
                'risk': risk_agirlik,
                'vade': vade_agirlik,
                'aciliyet': aciliyet_agirlik
            }
        }
    
    def oneri_hesapla(self, ev_degeri: Dict, haber_analizi: Dict, 
                     piyasa_puani: float, kullanici_profili: Dict) -> Dict:
        """Kişiselleştirilmiş öneri hesapla"""
        
        # 1. Temel puan: Haber analizi puanı
        temel_puan = haber_analizi['haber_puani']
        
        # 2. Piyasa puanı ekle
        temel_puan = (temel_puan + piyasa_puani) / 2
        
        # 3. Kullanıcı profili ağırlıklarını uygula
        agirliklar = kullanici_profili['agirliklar']
        
        # Risk toleransına göre ayarla
        if agirliklar['risk'] < 1:  # Düşük risk
            if temel_puan < 5:  # Negatif piyasa
                temel_puan -= 0.5
        else:  # Yüksek risk
            if temel_puan > 5:  # Pozitif piyasa
                temel_puan += 0.5
        
        # Aciliyete göre ayarla
        temel_puan *= agirliklar['aciliyet']
        
        # Kullanıcı tipine göre ayarla
        tip_agirlik = agirliklar['tip']
        temel_puan *= tip_agirlik['getiri']
        
        # 4. Vadeye göre ayarla
        if agirliklar['vade'] < 1 and temel_puan < 6:  # Kısa vade + düşük puan
            temel_puan -= 0.5
        elif agirliklar['vade'] > 1 and temel_puan > 6:  # Uzun vade + yüksek puan
            temel_puan += 0.5
        
        # 5. Skoru sınırla (0-10)
        final_puan = max(0, min(10, temel_puan))
        
        # 6. Öneri seviyesini belirle
        oneri_seviyesi = None
        for seviye, bilgi in self.oneri_seviyeleri.items():
            min_skor, max_skor = bilgi['skor']
            if min_skor <= final_puan < max_skor:
                oneri_seviyesi = seviye
                break
        
        if not oneri_seviyesi:
            oneri_seviyesi = 'bekle'
        
        oneri_bilgi = self.oneri_seviyeleri[oneri_seviyesi]
        
        # 7. Detaylı açıklama oluştur
        aciklama = self.aciklama_olustur(
            oneri_bilgi['oneri'],
            ev_degeri,
            haber_analizi,
            kullanici_profili['profil'],
            final_puan
        )
        
        # 8. Eylem planı oluştur
        eylem_plani = self.eylem_plani_olustur(
            oneri_bilgi['oneri'],
            ev_degeri,
            kullanici_profili['profil']
        )
        
        # 9. Risk analizi
        risk_analizi = self.risk_analizi_yap(
            final_puan,
            haber_analizi,
            kullanici_profili['profil']
        )
        
        return {
            'oneri': oneri_bilgi['oneri'],
            'emoji': oneri_bilgi['emoji'],
            'puan': round(final_puan, 1),
            'aciklama': aciklama,
            'eylem_plani': eylem_plani,
            'risk_analizi': risk_analizi,
            'haber_bazli_puan': haber_analizi['haber_puani'],
            'kullanici_profili': kullanici_profili['profil']
        }
    
    def aciklama_olustur(self, oneri: str, ev_degeri: Dict, 
                        haber_analizi: Dict, kullanici_profili: Dict, puan: float) -> str:
        """Öneri açıklaması oluştur"""
        
        ilce = ev_degeri.get('ilce', 'İstanbul')
        pozitif_haber = haber_analizi.get('pozitif_haber_sayisi', 0)
        negatif_haber = haber_analizi.get('negatif_haber_sayisi', 0)
        
        temel_aciklamalar = {
            'ACİL SAT': f"⚠️ {ilce} bölgesinde yüksek risk var ({negatif_haber} negatif haber). Acilen satış yapmanız önerilir.",
            'SAT': f"📉 {ilce} piyasasında satış için uygun zaman. {negatif_haber} negatif haber mevcut.",
            'BEKLE/GÖZLE': f"⚖️ {ilce} piyasası dengede. {pozitif_haber} pozitif, {negatif_haber} negatif haber. Karar için bekleyin.",
            'TUT': f"📊 {ilce} bölgesinde değer artışı bekleniyor ({pozitif_haber} pozitif haber). Evinizi tutun.",
            'İYİ TUT': f"📈 {ilce} piyasası çok olumlu ({pozitif_haber} pozitif haber). Kesinlikle tutun, değer artacak.",
            'AL': f"💰 {ilce} bölgesinde alım fırsatları var ({pozitif_haber} pozitif haber). Araştırma yapın.",
            'ACİL AL': f"🚀 {ilce} piyasasında acil alım fırsatı! {pozitif_haber} pozitif haber, fırsat kaçırmayın."
        }
        
        temel = temel_aciklamalar.get(oneri, "Piyasa analizi devam ediyor...")
        
        # Kullanıcı profiline göre özelleştirme
        kisi_ek = ""
        if kullanici_profili['hedef'] == 'nakit':
            kisi_ek = " Nakit ihtiyacınız olduğu için satış daha mantıklı."
        elif kullanici_profili['hedef'] == 'kira':
            kisi_ek = " Kira geliri hedefiniz için tutmak avantajlı."
        elif kullanici_profili['hedef'] == 'kar':
            if puan > 7:
                kisi_ek = " Kar hedefiniz için alım veya tutma düşünebilirsiniz."
            else:
                kisi_ek = " Kar hedefiniz için mevcut piyasa riskli."
        
        return f"{temel}{kisi_ek} Öneri puanı: {puan}/10"
    
    def eylem_plani_olustur(self, oneri: str, ev_degeri: Dict, 
                          kullanici_profili: Dict) -> List[Dict]:
        """Eylem planı oluştur"""
        
        planlar = {
            'ACİL SAT': [
                {'eylem': 'Hemen ilan verin', 'sure': '24 saat', 'oncelik': 'yuksek'},
                {'eylem': '3 farklı ekspertiz alın', 'sure': '3 gün', 'oncelik': 'yuksek'},
                {'eylem': 'Fiyatı piyasa ortalamasının %5 altında belirleyin', 'sure': '1 gün', 'oncelik': 'yuksek'},
                {'eylem': 'Tüm tapu belgelerinizi hazırlayın', 'sure': '2 gün', 'oncelik': 'orta'}
            ],
            'SAT': [
                {'eylem': 'İlan verin', 'sure': '1 hafta', 'oncelik': 'yuksek'},
                {'eylem': '2 ekspertiz değerlemesi alın', 'sure': '5 gün', 'oncelik': 'yuksek'},
                {'eylem': 'Fiyat araştırması yapın', 'sure': '3 gün', 'oncelik': 'orta'},
                {'eylem': 'Alıcı görüşmeleri planlayın', 'sure': '2 hafta', 'oncelik': 'orta'}
            ],
            'BEKLE/GÖZLE': [
                {'eylem': 'Piyasayı takip edin', 'sure': 'sürekli', 'oncelik': 'yuksek'},
                {'eylem': 'Haftalık haber analizi yapın', 'sure': 'her hafta', 'oncelik': 'orta'},
                {'eylem': 'Komşu satış fiyatlarını araştırın', 'sure': '2 hafta', 'oncelik': 'orta'},
                {'eylem': 'Profesyonel danışmanlık alın', 'sure': '1 ay', 'oncelik': 'dusuk'}
            ],
            'TUT': [
                {'eylem': 'Evin bakımını yapın', 'sure': '1 ay', 'oncelik': 'orta'},
                {'eylem': 'Kira geliri elde etmeyi düşünün', 'sure': '2 ay', 'oncelik': 'orta'},
                {'eylem': 'Piyasa takibine devam edin', 'sure': 'sürekli', 'oncelik': 'orta'},
                {'eylem': 'Küçük iyileştirmeler yapın', 'sure': '3 ay', 'oncelik': 'dusuk'}
            ],
            'İYİ TUT': [
                {'eylem': 'Kesinlikle satmayın', 'sure': '1+ yıl', 'oncelik': 'yuksek'},
                {'eylem': 'Uzun vadeli yatırım planı yapın', 'sure': '1 ay', 'oncelik': 'yuksek'},
                {'eylem': 'Kira gelirini optimize edin', 'sure': '3 ay', 'oncelik': 'orta'},
                {'eylem': 'Evin değerini artıracak iyileştirmeler yapın', 'sure': '6 ay', 'oncelik': 'dusuk'}
            ],
            'AL': [
                {'eylem': 'Piyasa araştırması yapın', 'sure': '2 hafta', 'oncelik': 'yuksek'},
                {'eylem': 'Finansman seçeneklerini araştırın', 'sure': '1 hafta', 'oncelik': 'yuksek'},
                {'eylem': 'Potansiyel bölgeleri belirleyin', 'sure': '3 hafta', 'oncelik': 'orta'},
                {'eylem': 'Uzman danışmanlık alın', 'sure': '1 ay', 'oncelik': 'orta'}
            ],
            'ACİL AL': [
                {'eylem': 'Hemen araştırmaya başlayın', 'sure': '24 saat', 'oncelik': 'yuksek'},
                {'eylem': 'Finansmanı ayarlayın', 'sure': '3 gün', 'oncelik': 'yuksek'},
                {'eylem': 'Fırsatları günlük takip edin', 'sure': 'her gün', 'oncelik': 'yuksek'},
                {'eylem': 'Acil alım için hazırlık yapın', 'sure': '1 hafta', 'oncelik': 'yuksek'}
            ]
        }
        
        return planlar.get(oneri, [
            {'eylem': 'Piyasayı takip edin', 'sure': 'sürekli', 'oncelik': 'yuksek'},
            {'eylem': 'Profesyonel danışın', 'sure': '1 ay', 'oncelik': 'orta'}
        ])
    
    def risk_analizi_yap(self, puan: float, haber_analizi: Dict, 
                        kullanici_profili: Dict) -> Dict:
        """Risk analizi yap"""
        
        if puan < 4:
            risk_seviyesi = 'yüksek'
            risk_aciklamasi = 'Piyasa koşulları olumsuz, yüksek risk var'
        elif puan < 6:
            risk_seviyesi = 'orta'
            risk_aciklamasi = 'Piyasa dengeli, orta risk seviyesi'
        elif puan < 8:
            risk_seviyesi = 'düşük'
            risk_aciklamasi = 'Piyasa olumlu, düşük risk'
        else:
            risk_seviyesi = 'çok düşük'
            risk_aciklamasi = 'Piyasa çok olumlu, çok düşük risk'
        
        # Kullanıcı risk toleransı ile karşılaştır
        uyum = ""
        if kullanici_profili['risk_toleransi'] == 'yüksek' and risk_seviyesi in ['yüksek', 'orta']:
            uyum = "Kullanıcı yüksek risk toleranslı, bu risk seviyesi kabul edilebilir"
        elif kullanici_profili['risk_toleransi'] == 'düşük' and risk_seviyesi in ['yüksek', 'orta']:
            uyum = "Dikkat: Kullanıcı düşük risk toleranslı, bu risk seviyesi yüksek"
        else:
            uyum = "Risk seviyesi kullanıcı profiliyle uyumlu"
        
        return {
            'risk_seviyesi': risk_seviyesi,
            'risk_aciklamasi': risk_aciklamasi,
            'kullanici_risk_uyumu': uyum,
            'pozitif_haber_orani': haber_analizi.get('pozitif_haber_sayisi', 0) / 
                                   max(1, haber_analizi.get('pozitif_haber_sayisi', 0) + 
                                       haber_analizi.get('negatif_haber_sayisi', 0))
        }

# ==================== ANA SİSTEM ====================

class GelismisEvDegerlemeSistemi:
    """Gelişmiş ev değerleme ve öneri sistemi"""
    
    def __init__(self):
        self.haber_analiz = HaberAnalizSistemi()
        self.oneri_sistemi = KisisellestirilmisOneriSistemi()
    
    def komple_analiz_yap(self, ev_bilgileri: Dict, kullanici_bilgileri: Dict) -> Dict:
        """Tam analiz yap: Değerleme + Haber analizi + Öneri"""
        
        # 1. Ev değerlemesi yap (mevcut sistemden)
        ev_degeri = self.ev_degeri_hesapla(ev_bilgileri)
        
        # 2. Haber analizi yap
        ilce = ev_degeri.get('ilce', '').lower()
        haberler = self.haber_analiz.haber_cek(ilce=ilce, limit=15)
        haber_analizi = self.haber_analiz.haber_analizi_yap(haberler, ilce)
        
        # 3. Piyasa puanı (model tahmini + haber analizi)
        piyasa_puani = (haber_analizi['haber_puani'] + 
                       self.piyasa_puani_hesapla(ev_bilgileri)) / 2
        
        # 4. Kullanıcı profili analizi
        kullanici_profili = self.oneri_sistemi.kullanici_profili_analizi(kullanici_bilgileri)
        
        # 5. Kişiselleştirilmiş öneri
        oneri = self.oneri_sistemi.oneri_hesapla(
            ev_degeri, haber_analizi, piyasa_puani, kullanici_profili
        )
        
        # 6. Gelecek tahmini
        gelecek_tahmini = self.gelecek_tahmini_yap(
            ev_degeri['tahmini_deger'],
            haber_analizi,
            oneri['puan']
        )
        
        return {
            'ev_degerleme': ev_degeri,
            'haber_analizi': {
                'toplam_haber': len(haberler),
                'analiz': haber_analizi,
                'haberler': haberler[:5]  # İlk 5 haberi göster
            },
            'kullanici_profili': kullanici_profili,
            'oneri_sistemi': oneri,
            'gelecek_tahmini': gelecek_tahmini,
            'piyasa_puani': piyasa_puani,
            'tarih': datetime.now().isoformat()
        }
    
    def ev_degeri_hesapla(self, ev_bilgileri: Dict) -> Dict:
        """Basit ev değeri hesaplama (mevcut sistemden)"""
        # Bu fonksiyon mevcut modelinizle entegre edilecek
        # Şimdilik basit bir hesaplama yapıyoruz
        
        net_m2 = int(ev_bilgileri.get('net_metrekare', 100))
        brut_m2 = int(ev_bilgileri.get('brut_metrekare', 110))
        ilce = ev_bilgileri.get('ilce', 'Ortalam')
        
        # İlçe katsayıları
        ilce_katsayilari = {
            'kadikoy': 85000, 'besiktas': 95000, 'sisli': 80000,
            'atasehir': 85000, 'maltepe': 50000, 'umraniye': 55000,
            'ortalam': 55000
        }
        
        ilce_katsayi = ilce_katsayilari.get(ilce.lower(), 55000)
        temel_deger = net_m2 * ilce_katsayi
        
        # Diğer katsayılar
        kat_katsayi = 0.95 if int(ev_bilgileri.get('bulundugu_kat_int', 0)) < 3 else 1.0
        yas_katsayi = 0.95 if int(ev_bilgileri.get('bina_yasi', 5)) > 10 else 1.0
        site_katsayi = 1.05 if ev_bilgileri.get('site_icinde_code', 0) == 1 else 1.0
        
        tahmini_deger = temel_deger * kat_katsayi * yas_katsayi * site_katsayi
        
        return {
            'tahmini_deger': round(tahmini_deger, 2),
            'temel_deger': round(temel_deger, 2),
            'net_m2': net_m2,
            'brut_m2': brut_m2,
            'ilce': ilce,
            'ilce_katsayisi': ilce_katsayi,
            'katsayilar': {
                'kat': kat_katsayi,
                'yas': yas_katsayi,
                'site': site_katsayi
            }
        }
    
    def piyasa_puani_hesapla(self, ev_bilgileri: Dict) -> float:
        """Piyasa puanı hesapla (0-10)"""
        # Mevcut modelin tahmin güvenilirliği ve diğer faktörler
        return random.uniform(6.5, 8.5)  # Simülasyon
    
    def gelecek_tahmini_yap(self, suanki_deger: float, haber_analizi: Dict, 
                           oneri_puani: float) -> Dict:
        """Gelecek değer tahmini yap"""
        
        # Haber etkisi
        haber_etkisi = haber_analizi.get('ortalama_etki', 0)
        
        # Öneri puanına göre artış tahmini
        if oneri_puani < 4:
            artis_orani = random.uniform(-5, 0)  # Düşüş
        elif oneri_puani < 6:
            artis_orani = random.uniform(0, 3)  # Düşük artış
        elif oneri_puani < 8:
            artis_orani = random.uniform(3, 8)  # Orta artış
        else:
            artis_orani = random.uniform(8, 15)  # Yüksek artış
        
        # Haber etkisini ekle
        artis_orani += haber_etkisi * 5
        
        # Kısa, orta, uzun vade tahminleri
        gelecek_deger_6ay = suanki_deger * (1 + artis_orani/100)
        gelecek_deger_1yil = suanki_deger * (1 + (artis_orani * 1.5)/100)
        gelecek_deger_2yil = suanki_deger * (1 + (artis_orani * 2.2)/100)
        
        return {
            'suanki_deger': suanki_deger,
            'beklenen_artis_orani': round(artis_orani, 1),
            '6_ay_sonrasi': round(gelecek_deger_6ay, 2),
            '1_yil_sonrasi': round(gelecek_deger_1yil, 2),
            '2_yil_sonrasi': round(gelecek_deger_2yil, 2),
            'tahmin_guvenilirligi': min(0.95, 0.7 + (oneri_puani/10 * 0.3))
        }

# ==================== FLASK API ENDPOINT'LERİ ====================

# Global sistem örneği
sistem = GelismisEvDegerlemeSistemi()

@app.route('/advanced-predict', methods=['POST'])
def advanced_predict():
    """Gelişmiş tahmin ve öneri endpoint'i"""
    try:
        data = request.get_json()
        
        if not data or 'emlakDegerleme' not in data:
            return jsonify({'error': 'Geçersiz veri formatı'}), 400
        
        emlak_data = data['emlakDegerleme']
        ozellikler = emlak_data['ozellikler']
        konum = emlak_data.get('konumBilgisi', {})
        
        # Kullanıcı bilgileri (isteğe bağlı)
        kullanici_bilgileri = data.get('kullaniciBilgileri', {
            'kullanici_tipi': 'oturan',
            'yatirim_vadesi': 'orta',
            'risk_toleransi': 'orta',
            'aciliyet': 'yok',
            'hedef': 'deger_koruma',
            'alternatif_yatirim': True
        })
        
        # Ev bilgilerini hazırla
        ev_bilgileri = {
            'net_metrekare': ozellikler.get('net_metrekare', 100),
            'brut_metrekare': ozellikler.get('brut_metrekare', 110),
            'ilce': konum.get('adres', {}).get('ilce', 'Ortalam') if konum else 'Ortalam',
            'bulundugu_kat_int': ozellikler.get('bulundugu_kat_int', 0),
            'site_icinde_code': ozellikler.get('site_icinde_code', 0),
            'bina_yasi': 5  # Varsayılan
        }
        
        # Komple analiz yap
        sonuc = sistem.komple_analiz_yap(ev_bilgileri, kullanici_bilgileri)
        
        return jsonify({
            'success': True,
            'tahmin': {
                'suanki_deger': sonuc['ev_degerleme']['tahmini_deger'],
                'birim_fiyat': sonuc['ev_degerleme']['tahmini_deger'] / 
                              max(1, sonuc['ev_degerleme']['net_m2']),
                'ilce': sonuc['ev_degerleme']['ilce'],
                'metrekare': sonuc['ev_degerleme']['net_m2']
            },
            'haber_bazli_analiz': {
                'toplam_haber': sonuc['haber_analizi']['toplam_haber'],
                'pozitif_haber': sonuc['haber_analizi']['analiz']['pozitif_haber_sayisi'],
                'negatif_haber': sonuc['haber_analizi']['analiz']['negatif_haber_sayisi'],
                'haber_puani': sonuc['haber_analizi']['analiz']['haber_puani'],
                'son_haberler': sonuc['haber_analizi']['haberler']
            },
            'kisisellestirilmis_oneri': {
                'oneri': sonuc['oneri_sistemi']['oneri'],
                'emoji': sonuc['oneri_sistemi']['emoji'],
                'puan': sonuc['oneri_sistemi']['puan'],
                'aciklama': sonuc['oneri_sistemi']['aciklama'],
                'eylem_plani': sonuc['oneri_sistemi']['eylem_plani'],
                'risk_analizi': sonuc['oneri_sistemi']['risk_analizi']
            },
            'gelecek_tahmini': sonuc['gelecek_tahmini'],
            'kullanici_profili': sonuc['kullanici_profili']['profil'],
            'piyasa_puani': sonuc['piyasa_puani'],
            'timestamp': sonuc['tarih']
        })
        
    except Exception as e:
        print(f"Advanced predict hatası: {str(e)}")
        return jsonify({'error': f'İşlem hatası: {str(e)}'}), 500

@app.route('/haber-analizi', methods=['POST'])
def haber_analizi():
    """Sadece haber analizi endpoint'i"""
    try:
        data = request.get_json()
        ilce = data.get('ilce', 'İstanbul')
        
        haberler = sistem.haber_analiz.haber_cek(ilce=ilce, limit=20)
        analiz = sistem.haber_analiz.haber_analizi_yap(haberler, ilce)
        
        return jsonify({
            'success': True,
            'ilce': ilce,
            'analiz': analiz,
            'haberler': haberler[:10],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/oneri-sistemi', methods=['POST'])
def oneri_sistemi():
    """Sadece öneri sistemi endpoint'i"""
    try:
        data = request.get_json()
        
        # Gerekli veriler
        ev_degeri = data.get('ev_degeri', {})
        haber_puani = data.get('haber_puani', 5)
        kullanici_bilgileri = data.get('kullanici_bilgileri', {})
        
        # Haber analizi simülasyonu
        haber_analizi = {
            'haber_puani': haber_puani,
            'pozitif_haber_sayisi': 3 if haber_puani > 5 else 1,
            'negatif_haber_sayisi': 1 if haber_puani < 5 else 0,
            'ortalama_etki': (haber_puani - 5) / 2
        }
        
        # Kullanıcı profili
        kullanici_profili = sistem.oneri_sistemi.kullanici_profili_analizi(kullanici_bilgileri)
        
        # Öneri hesapla
        oneri = sistem.oneri_sistemi.oneri_hesapla(
            ev_degeri, haber_analizi, haber_puani, kullanici_profili
        )
        
        return jsonify({
            'success': True,
            'oneri': oneri,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== MEVCUT MODEL FONKSİYONLARI (Korundu) ====================

def prepare_and_train_model():
    """Mevcut model eğitim fonksiyonu"""
    global model, scaler, label_encoders, feature_columns
    
    try:
        # Mevcut kodunuz buraya gelecek
        print("Model eğitiliyor...")
        return True
    except Exception as e:
        print(f"Model eğitme hatası: {str(e)}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    """Mevcut predict endpoint'i (geriye uyumluluk için)"""
    try:
        data = request.get_json()
        # Mevcut kodunuz buraya gelecek
        return jsonify({'success': True, 'message': 'Mevcut API çalışıyor'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== ANA ÇALIŞTIRMA ====================

if __name__ == '__main__':
    print("""
    ============================================
    🏠 GELİŞMİŞ EV DEĞERLEME VE ÖNERİ SİSTEMİ
    ============================================
    
    📊 Özellikler:
    1. Haber bazlı piyasa analizi
    2. Kişiselleştirilmiş Sat/Al/Tut önerileri
    3. Kullanıcı profiline göre özelleştirme
    4. Eylem planları ve risk analizi
    5. Gelecek değer tahminleri
    
    🌐 API Endpoint'leri:
    - POST /advanced-predict : Tam analiz
    - POST /haber-analizi    : Haber analizi
    - POST /oneri-sistemi    : Öneri sistemi
    - POST /predict          : Mevcut tahmin (geriye uyumlu)
    
    🚀 API http://localhost:5001 adresinde çalışıyor...
    """)
    
    # Model eğitimi (isteğe bağlı)
    # prepare_and_train_model()
    
    app.run(debug=True, port=5001, host='0.0.0.0')