# 🛡️ PactumRAG: Local Multi-Agent & Multi-Hop Belge/Sözleşme Denetçisi

Microsoft Foundry Local mimarisine ve yerel SQLite vektör arama ilkelerine tam uyumlu, %100 yerel ve gizlilik odaklı **Çoklu Ajan (Proposer, Challenger, Judge)** sözleşme denetim asistanı.

---

## 🎯 Proje Teslim Kriterleri ve Mimari Uygunluk

| Kriter | Mimari Çözüm | Durum |
|---|---|---|
| **Çevrimdışı Çalışma Zorunluluğu** | Microsoft Foundry Local SDK (`Phi-3.5 Mini` & yerel embedding) desteği; SDK olmasa dahi çökmeden çalışan deterministik semantik NumPy fallback motoru. Sıfır bulut API bağımlılığı. | ✅ TAM UYUMLU |
| **Hafif Veri Tabanı Mimarisi** | Harici vektör DB (Chroma, FAISS vb.) yerine saf Python `sqlite3` (`documents.db`) tabanlı depolama ve `numpy` Dot Product / Kosinüs benzerliği. | ✅ TAM UYUMLU |
| **Sorumlu Yapay Zeka (Responsible AI)** | Sözleşmede düzenlenmeyen konularda (örn. geçici hat dondurma) halüsinasyon ve sahte cezai şart üretilmez; alaka eşiği filtresiyle %99 güvenle çekimser kalınır (**Durum C**). | ✅ TAM UYUMLU |
| **Çalışan Demo ve Raporlama** | Tek dosya (`app.py`), interaktif Streamlit arayüzü, canlı ajan tartışma adımları, anlık metrikler ve 3 farklı hukuki durumu tek tıkla test eden demo paneli. | ✅ TAM UYUMLU |

---

## 🚀 3 Temel Demo Senaryosu (Tek Tıkla Test)

Arayüzde ana panelde yer alan hazır senaryo butonları ile tüm hukuki durumlar anında test edilebilir:

1. **🔴 Senaryo A: Tuzak Fesih Denetimi (Durum A - Cezai Şart Riski)**
   - **Soru:** *"Müşteri sözleşmenin 3. ayında 30 gün önceden bildirerek cezasız fesih yapabilir mi?"*
   - **Mekanizma:** `Proposer` Madde 4.1'e dayanarak fesih yapılabileceğini belirtir; `Challenger` 2. Hop aramasıyla Madde 8.2'deki 12 aylık taahhüt kuralını yakalar; `Judge` erken fesihte kalan ayların ve indirimlerin cezai şart doğuracağı hükmünü bağlar (**YÜKSEK RİSK**).

2. **🟢 Senaryo B: Doğrudan Hak Sorgusu (Durum B - Cezasız / Uygun)**
   - **Soru:** *"Sözleşmeye ilişkin tüm tebligat ve yazılı bildirimler hangi usulle yapılmalıdır?"*
   - **Mekanizma:** Madde 7.1'deki bildirim ve tebligat prosedürü doğrulanır. Sözleşmede bu hakkı kısıtlayan bir taahhüt veya cezai şart olmadığı teyit edilerek doğrudan onaylanır (**TALEP UYGUNDUR / DÜŞÜK RİSK**).

3. **🟡 Senaryo C: Sorumlu AI / Halüsinasyon Engeli (Durum C - Kapsam Dışı)**
   - **Soru:** *"Aboneliğimi 1 yıl içinde en fazla kaç ay süreyle geçici olarak dondurabilirim?"*
   - **Mekanizma:** `Retrieval Grader` sorudaki eylemin ("dondurma") sözleşmede yer almadığını tespit eder. Sistem 2. Hop aramaya zorlanmaz; ezbere fesih cezası uydurmaz ve %99 güvenle **"BİLGİ SÖZLEŞMEDE BULUNAMADI"** uyarısı verir.

---

## 🛠️ Kurulum ve Çalıştırma

```bash
# 1. Sanal ortamı aktif edin (Windows PowerShell)
.\contract-rag\venv\Scripts\Activate.ps1

# 2. Bağımlılıkları kontrol edin (Hepsi yereldir)
pip install streamlit pydantic pypdf numpy

# 3. Streamlit uygulamasını başlatın
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.
