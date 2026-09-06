# 🛡️ PactumRAG: Local Multi-Agent & Multi-Hop Belge ve Sözleşme Denetçisi

PactumRAG; hukuki metinlerde ve sözleşmelerde yer alan gizli cezai şartları, istisna maddelerini ve çapraz atıfları tespit etmek için geliştirilmiş, tamamen yerel (offline) çalışan çok adımlı (*multi-hop*) ve çoklu ajanlı (*multi-agent*) bir sözleşme analiz platformudur.

Geleneksel tek adımlı RAG yaklaşımları, kullanıcının sorduğu hakka odaklanıp ("30 gün önceden bildirerek fesih yapılabilir") sözleşmenin başka bir sayfasında saklanan taahhüt ve cezai yaptırım maddelerini ("12 aydan önce fesihte indirim bedelleri cezai şart olarak tahsil edilir") gözden kaçırabilir. PactumRAG, karşıt görüşlü denetçi ajanlar ve iki adımlı arama mimarisi ile bu hukuki tuzakları otomatik olarak ortaya çıkarır.

---

## 🏛️ Mimari Yapı

Sistem, harici bulut servislerine veya ağır dış veritabanlarına ihtiyaç duymayan 4 ana katmandan oluşur:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             KULLANICI SORUSU                             │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. VERİ & VEKTÖR KATMANI (SQLite & Embedding)                           │
│  • Regex Madde Ayrıştırma (pypdf, Madde X.Y / Sayfa Metadata)           │
│  • Yerel Embedding (Microsoft Foundry Local SDK / Semantik Fallback)     │
│  • Hafif Depolama: Python sqlite3 (contract_chunks tablosu)              │
│  • Benzerlik Araması: NumPy Normalize Kosinüs Benzerliği (Dot Product)   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. ÇOK ADIMLI ARAMA MOTORU (Multi-Hop Retrieval Engine)                  │
│  • 1. Hop: Kullanıcı sorusuna en yakın hak/hüküm maddesini bulur.        │
│  • 2. Hop: Maddelerdeki atıf ve kısıtlamaları hedef alan 2. vektör araması│
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. ÇOKLU AJAN DENETİM PROTOKOLÜ (Multi-Agent Debate)                     │
│  • Proposer Agent: 1. Hop verisini inceleyerek hak tanıyıcı görüş üretir. │
│  • Challenger Agent: İstisnaları ve riskleri yakalar, 2. Hop'u tetikler. │
│  • Judge Agent: Kanıtları sentezler, risk derecesini ve hükmü bağlar.    │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. SORUMLU YAPAY ZEKA & ALAKA EŞİĞİ (Strict Retrieval Grader)             │
│  • Sözleşmede düzenlenmeyen konularda halüsinasyon üretmez.              │
│  • Çekimser kalarak "Bilgi Sözleşmede Bulunamadı" kararı üretir.         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1. Veri ve İndeksleme Katmanı (Ingestion & SQLite Vector Store)
- **Madde Bazlı Ayrıştırma:** Yüklenen PDF belgeleri sabit karakter bloklarına bölünmez; `r"(Madde\s+\d+(\.\d+)?|Article\s+\d+(\.\d+)?)"` regex deseniyle hukuki madde bütünlüğü korunarak ayrıştırılır ve sayfa numaraları metadata olarak kaydedilir.
- **Hafif SQLite Vektör Depolama:** Harici bir vektör veritabanı (Chroma, Pinecone, FAISS vb.) kurulmaz. Python'ın dahili `sqlite3` modülü üzerinden `contract_chunks` tablosunda metin ve vektör dizileri saklanır.
- **Vektör Motoru:** Microsoft Foundry Local SDK (`foundry-local-sdk`) entegrasyonu üzerinden yerel modeller (`qwen3-embedding` ve `Phi-3.5 Mini`) kullanılır. SDK aktif olmadığında ise deterministik semantik NumPy fallback motoruna geçiş yapılır. Arama, vektörlerin normalize kosinüs benzerliği formülü üzerinden saf NumPy ile gerçekleştirilir.

### 2. Çok Adımlı Arama Motoru (Multi-Hop Engine)
- **1. Hop (Sorgu Odaklı Arama):** Kullanıcının doğal dil sorusu vektörleştirilir ve sözleşmedeki en ilgili birincil madde tespit edilir.
- **2. Hop (Çapraz Atıf ve İstisna Arama):** Birinci maddede yer alan kısıtlayıcı terimler (*"saklıdır"*, *"taahhüt"*, *"istisna"*, *"cezai şart"*, *"madde X uyarınca"*) analiz edilir. Challenger ajanı tarafından üretilen hedeflenmiş alt sorgu ile sözleşmenin farklı bölümlerindeki bağlı yaptırım maddeleri getirilir.

### 3. Çoklu Ajan Denetim Döngüsü (Multi-Agent Debate Protocol)
Sistem üç uzman rol ile yapılandırılmıştır:
- **Proposer Agent (Savunucu):** 1. Hop bulgularını değerlendirir; kullanıcının talep ettiği hakkın sözleşmede tanımlı olup olmadığını tespit eder.
- **Challenger Agent (Denetçi / İtirazcı):** Proposer'ın sunduğu görüşü kritik eder; sözleşmedeki çapraz atıfları, taahhüt sürelerini ve kısıtlayıcı istisnaları inceler. Gerekli durumlarda 2. Hop aramasını tetikler.
- **Judge Agent (Hakem):** Her iki ajanın sunduğu kanıtları ve sözleşme maddelerini sentezler; 0-100 arası güven skoru, risk durumu, kesin madde ve sayfa atıflarını içeren bağlayıcı nihai kararı üretir.

### 4. Sorumlu AI ve Alaka Eşiği (Responsible AI & Anti-Hallucination)
- **Strict Grounding & Action Filtering:** Sözleşmede yer almayan (örneğin *"aboneliği geçici dondurma"*) konular sorulduğunda, kelime benzerliği nedeniyle ezbere fesih cezası veya varsayımsal hüküm uydurulmasını engeller.
- Sözleşmede karşılığı olmayan taleplerde sistem otomatik olarak çekimser kalır ve gerekçesiyle birlikte kapsam dışı bildirimi yapar.

---

## 🔍 Desteklenen Temel Hukuki Senaryolar

Arayüzde tek tıkla test edilebilen veya serbest metin kutusundan sorgulanabilen senaryolar:

1. **🔴 Çapraz Cezai Şart ve Taahhüt Riski (Multi-Hop Denetimi)**
   - *Örnek Soru:* "Müşteri sözleşmenin 3. ayında 30 gün önceden bildirerek cezasız fesih yapabilir mi?"
   - *Sonuç:* 1. Hop genel fesih hakkını (Madde 4.1), 2. Hop ise 12 aylık taahhüt istisnası ve cezai şartı (Madde 8.2) yakalar. Hüküm: **Talebiniz Koşullu ve Riskli (Cezai Şart Riski)**.

2. **🟢 Doğrudan Tanınan Hak ve Usul Tespiti (Cezasız İşlem)**
   - *Örnek Soru:* "Sözleşmeye ilişkin tüm tebligat ve yazılı bildirimler hangi usulle yapılmalıdır?"
   - *Sonuç:* Madde 7.1'deki bildirim prosedürü tespit edilir; kısıtlayıcı bir engel bulunmadığı için 2. Hop atlanır. Hüküm: **Talep Uygundur ve Doğrudan Uygulanabilir**.

3. **🟡 Düzenlenmemiş Konu ve Çekimserlik (Kapsam Dışı)**
   - *Örnek Soru:* "Aboneliğimi 1 yıl içinde en fazla kaç ay süreyle geçici olarak dondurabilirim?"
   - *Sonuç:* Eylem sözleşmede geçmediği için arama filtresi halüsinasyonu engeller. Hüküm: **Bilgi Sözleşmede Bulunamadı (%99 Güvenle Çekimser)**.

---

## 💻 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.10+
- Standart kütüphaneler dışında yalnızca arayüz ve veri işleme için temel paketler:
  `streamlit`, `pydantic`, `pypdf`, `numpy`

### Çalıştırma Adımları

```bash
# 1. Depoyu klonlayın ve proje dizinine gidin
git clone https://github.com/hamzamalikyildiz/PactumRAG.git
cd PactumRAG

# 2. Sanal ortamı aktif edin (veya oluşturun)
.\contract-rag\venv\Scripts\Activate.ps1

# 3. Bağımlılıkları yükleyin (İlk kurulumda)
pip install streamlit pydantic pypdf numpy

# 4. Streamlit arayüzünü başlatın
streamlit run app.py
```

Uygulama yerel tarayıcınızda otomatik olarak açılır:
🌐 **`http://localhost:8501`**
