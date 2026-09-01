# 🛡️ Local Multi-Agent & Multi-Hop Belge/Sözleşme Denetçisi

Microsoft Foundry Local mimarisine ve yerel SQLite vektör arama ilkelerine tam uyumlu, %100 yerel ve gizlilik odaklı **Çoklu Ajan (Proposer, Challenger, Judge)** sözleşme denetim asistanı.

## 🚀 Temel Özellikler
- **Sıfır Bulut / Sıfır Dış API:** Dış ağ çağrısı olmaksızın yerel embedding ve çıkarım mimarisi.
- **SQLite3 Vektör Depolama:** Harici vektör veritabanı olmaksızın saf SQLite + NumPy Kosinüs benzerliği.
- **Çoklu Ajan ve Multi-Hop RAG:**
  - **Proposer Agent (🤖):** 1. Hop araması ile ilk taslak hukuki görüşü üretir.
  - **Challenger Agent (🕵️):** Kısıtlayıcı çapraz atıf ve cezai şartları avlayarak 2. Hop aramasını tetikler.
  - **Judge Agent (⚖️):** Kanıtları sentezleyip Pydantic `JudgeVerdict` şeması ile nihai kararı bağlar.
- **Canlı Ajan Tartışma Odası:** Ajanların argüman ve itirazlarını adım adım animasyonlu olarak gösteren etkileşimli arayüz.

## 🛠️ Kurulum ve Çalıştırma

```bash
# Sanal ortamı aktif edin
.\contract-rag\venv\Scripts\Activate.ps1

# Uygulamayı başlatın
streamlit run app.py
```
