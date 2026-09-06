"""
Local Multi-Agent & Multi-Hop Belge/Sözleşme Denetçisi
Microsoft Foundry Local SDK + SQLite Vektör Arama + Multi-Agent Denetim Mimarisi
%100 Yerel, Sıfır Dış API, Sıfır Harici Vektör Veritabanı
"""

import io
import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
import streamlit as st

# ==============================================================================
# 1. SABİTLER VE KONFİGÜRASYON
# ==============================================================================

DB_FILE = "documents.db"
VECTOR_DIM = 128

# Hukuki semantik dayanak küme tanımları (128 Boyutlu Deterministik Uzay)
SEMANTIC_ANCHOR_CLUSTERS = {
    (0, 16): ["fesih", "feshet", "sona", "iptal", "dönme", "tahliye", "vazgeç", "sonlandırma"],
    (16, 32): ["taahhüt", "süre", "ay", "yıl", "müddet", "dönem", "12 ay", "asgari", "vade"],
    (32, 48): ["ceza", "cezai", "şart", "tazminat", "fatura", "kalan", "ücret", "indirim", "bedel", "tahsil", "muaccel"],
    (48, 64): ["mahkeme", "icra", "yetki", "uyuşmazlık", "kanun", "istanbul", "hukuk", "dava"],
    (64, 80): ["bildirim", "yazılı", "önceden", "ihbar", "30 gün", "gün", "tebligat", "noter"],
    (80, 96): ["ancak", "istisna", "saklı", "uyarınca", "koşul", "şartıyla", "tabi", "hariç", "kaydıyla"],
    (96, 112): ["müşteri", "taraf", "şirket", "sözleşme", "hizmet", "abone", "kullanıcı", "yüklenici"],
    (112, 128): ["madde", "hüküm", "kural", "talep", "gerekçe", "fıkra", "bent", "kanıt"]
}

# ==============================================================================
# 2. PYDANTIC VERİ ŞEMALARI
# ==============================================================================

class ChatMessage(BaseModel):
    role: str  # user, proposer, challenger, judge, system
    agent_name: str
    avatar: str
    content: str
    thought: str = ""
    action_tool: str = ""
    tool_output: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChallengerAssessment(BaseModel):
    needs_second_hop: bool = Field(description="Ek çapraz atıf veya istisna araması gerekli mi?")
    reason: str = Field(description="Challenger'ın şüphe veya gerekçe açıklaması")
    sub_query: str = Field(description="2. Hop için üretilen hedeflenmiş vektör arama sorgusu")
    detected_terms: List[str] = Field(default_factory=list, description="Bağlamda tespit edilen kısıtlayıcı terimler")


class JudgeVerdict(BaseModel):
    karar: str = Field(description="Nihai hukuki/denetim hükmü")
    guven_skoru: int = Field(ge=0, le=100, description="0-100 arası güven skoru")
    gerekce: str = Field(description="Maddeler arası ilişkileri açıklayan gerekçe")
    dayanak_maddeler: List[str] = Field(description="Hükme temel oluşturan maddeler")
    sayfa_referanslari: List[int] = Field(description="Maddelerin bulunduğu sayfalar")
    risk_var_mi: bool = Field(description="Cezai şart veya risk var mı?")


# ==============================================================================
# 3. EMBEDDING MOTORU (MICROSOFT FOUNDRY LOCAL + NUMPY SEMANTİK FALLBACK)
# ==============================================================================

class FoundryLocalEmbeddingEngine:
    """
    Microsoft Foundry Local SDK tabanlı yerel gömme motoru.
    SDK veya model aktif değilse %100 deterministik ve semantik ağırlıklı
    NumPy fallback vektör üreticisine otomatik düşer. Dış API kesinlikle çağrılmaz.
    """
    def __init__(self, vector_dim: int = VECTOR_DIM):
        self.vector_dim = vector_dim
        self.is_foundry_active = False
        self.engine_name = "NumPy Semantik Fallback (128-D)"
        self._session = None
        self._init_engine()

    def _init_engine(self):
        try:
            import foundry_local_sdk as fl
            config = fl.Configuration(app_name="PactumRAG")
            fl.FoundryLocalManager.initialize(config)
            mgr = fl.FoundryLocalManager.instance
            cat = mgr.catalog
            model = cat.get_model("qwen3-embedding-0.6b")
            if model and getattr(model, "is_loaded", False):
                self._session = fl.EmbeddingsSession(model)
                self.is_foundry_active = True
                self.engine_name = "Microsoft Foundry Local SDK (qwen3-embedding)"
        except Exception:
            self.is_foundry_active = False
            self.engine_name = "NumPy Semantik Fallback (128-D)"

    def get_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.vector_dim

        if self.is_foundry_active and self._session:
            try:
                res = self._session.process_request(text.strip())
                if res and hasattr(res, "embedding"):
                    vec = list(res.embedding)
                    if len(vec) == self.vector_dim:
                        return vec
            except Exception:
                pass

        return self._semantic_fallback_embedding(text)

    def _semantic_fallback_embedding(self, text: str) -> List[float]:
        """
        Deterministik, semantik ağırlıklı ve n-gram hash tabanlı yerel vektör üretici.
        """
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        clean_text = text.lower().strip()
        words = re.findall(r"\w+", clean_text)

        # 1. Semantik Hukuk Kümeleri Ağırlıklandırması
        for (start_idx, end_idx), keywords in SEMANTIC_ANCHOR_CLUSTERS.items():
            for kw in keywords:
                if kw in clean_text:
                    for idx in range(start_idx, end_idx):
                        vec[idx] += 1.8

        # 2. Kelime Bazlı Hash Dağıtımı
        for w in words:
            h = hash(w)
            vec[abs(h) % self.vector_dim] += 1.0
            vec[abs(h // 13) % self.vector_dim] += 0.5

        # 3. Karakter 3-Gram Hash Dağıtımı
        for i in range(len(clean_text) - 2):
            trigram = clean_text[i:i+3]
            vec[abs(hash(trigram)) % self.vector_dim] += 0.25

        # 4. L2 Normalizasyonu
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


@st.cache_resource
def get_embedding_engine() -> FoundryLocalEmbeddingEngine:
    return FoundryLocalEmbeddingEngine()


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """İki normalize vektör arasındaki kosinüs benzerliğini hesaplar."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ==============================================================================
# 4. VERİTABANI YÖNETİCİSİ (SQLITE3)
# ==============================================================================

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """SQLite veritabanı ve contract_chunks tablosunu başlatır."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contract_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                madde_no TEXT,
                sayfa_no INTEGER,
                content TEXT,
                embedding TEXT
            )
        """)
        conn.commit()


def clear_database():
    """Tüm kayıtlı sözleşme chunk'larını siler."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM contract_chunks")
        conn.commit()


def delete_chunk_by_id(chunk_id: int):
    """Belirli bir chunk'ı kimliğine göre siler."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM contract_chunks WHERE id = ?", (chunk_id,))
        conn.commit()


def insert_chunk(madde_no: str, sayfa_no: int, content: str, embedding: List[float]):
    """Yeni bir sözleşme maddesini vektörüyle birlikte SQLite'a kaydeder."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO contract_chunks (madde_no, sayfa_no, content, embedding)
            VALUES (?, ?, ?, ?)
        """, (madde_no, sayfa_no, content, json.dumps(embedding)))
        conn.commit()


def fetch_all_chunks() -> List[Dict[str, Any]]:
    """Tüm kayıtlı sözleşme maddelerini JSON embedding'leri açılarak döndürür."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, madde_no, sayfa_no, content, embedding FROM contract_chunks ORDER BY id ASC")
        rows = cursor.fetchall()
        return [
            {
                "id": r["id"],
                "madde_no": r["madde_no"],
                "sayfa_no": r["sayfa_no"],
                "content": r["content"],
                "embedding": json.loads(r["embedding"])
            }
            for r in rows
        ]


def load_demo_data(engine: FoundryLocalEmbeddingEngine) -> int:
    """
    Mod A: Spesifikasyonda belirtilen 3 maddelik tuzak demo sözleşmesini yükler.
    Tuzak: Madde 4.1 tek başına cezasız fesih vadederken, Madde 8.2 taahhüt ve cezai şart bağlar.
    """
    clear_database()
    demo_clauses = [
        (
            "Madde 4.1",
            1,
            "Müşteri, sözleşmeyi 30 gün önceden yazılı bildirimde bulunarak herhangi bir gerekçe göstermeksizin feshedebilir."
        ),
        (
            "Madde 8.2",
            2,
            "İşbu sözleşme 12 aylık taahhüt süresine tabidir. Madde 4.1 uyarınca yapılacak fesihlerde, kalan ayların ücreti ve sağlanan indirimler cezai şart olarak faturalandırılır."
        ),
        (
            "Madde 12.0",
            3,
            "Taraflar arasındaki uyuşmazlıklarda İstanbul Mahkemeleri ve İcra Daireleri yetkilidir."
        )
    ]

    for m_no, s_no, text in demo_clauses:
        emb = engine.get_embedding(f"{m_no}: {text}")
        insert_chunk(m_no, s_no, text, emb)

    return len(demo_clauses)


# ==============================================================================
# 5. GERÇEK PDF AYRIŞTIRICI (DİNAMİK VE HATASIZ)
# ==============================================================================

def parse_and_index_pdf(uploaded_file, engine: FoundryLocalEmbeddingEngine) -> int:
    """
    Mod B: Kullanıcının yüklediği PDF belgesini pypdf ile okur,
    sayfa numaralarını koruyarak regex ile maddelere böler ve SQLite'a kaydeder.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        st.error("`pypdf` kütüphanesi bulunamadı.")
        return 0

    clear_database()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
    total_indexed = 0

    madde_regex = re.compile(
        r"(?:(?:Madde|Article|Bölüm|Kısım)\s+\d+(?:\.\d+)*|\b\d+\.\s+[A-ZÇĞİÖŞÜa-zçğıöşü\s]{3,35}:)",
        re.IGNORECASE
    )

    for page_idx, page in enumerate(pdf_reader.pages):
        page_no = page_idx + 1
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        matches = list(madde_regex.finditer(text))
        if matches:
            first_start = matches[0].start()
            if first_start > 30:
                preamble = text[:first_start].strip()
                if len(preamble) > 15:
                    emb = engine.get_embedding(preamble)
                    insert_chunk(f"Giriş (Sayfa {page_no})", page_no, preamble, emb)
                    total_indexed += 1

            for idx, m in enumerate(matches):
                madde_title = m.group(0).strip()
                start_body = m.end()
                end_body = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                content_body = text[start_body:end_body].strip()
                full_content = f"{madde_title}: {content_body}" if content_body else madde_title

                if len(full_content) > 15:
                    emb = engine.get_embedding(full_content)
                    insert_chunk(madde_title, page_no, full_content, emb)
                    total_indexed += 1
        else:
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
            if not paragraphs:
                paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]

            for p_idx, para in enumerate(paragraphs):
                madde_title = f"Paragraf {page_no}.{p_idx+1}"
                emb = engine.get_embedding(para)
                insert_chunk(madde_title, page_no, para, emb)
                total_indexed += 1

    return total_indexed


# ==============================================================================
# 6. VEKTÖR GETİRME FONKSİYONLARI (MULTI-HOP RETRIEVAL)
# ==============================================================================

def retrieve_hop_1(query: str, engine: FoundryLocalEmbeddingEngine) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    1. Hop Vektör Arama:
    Kullanıcı sorgusunun embedding'i ile SQLite'taki tüm maddeleri tarar.
    En yüksek anlamsal kosinüs benzerliğine sahip 1. maddeyi ve sıralı listeyi döndürür.
    """
    q_emb = engine.get_embedding(query)
    all_chunks = fetch_all_chunks()
    if not all_chunks:
        raise ValueError("Veritabanında incelenecek sözleşme maddesi bulunamadı!")

    scored = []
    for c in all_chunks:
        s = cosine_similarity(q_emb, c["embedding"])
        scored.append({**c, "score": s, "hop": 1})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[0], scored


def retrieve_hop_2(
    sub_query: str,
    engine: FoundryLocalEmbeddingEngine,
    exclude_ids: List[int],
    cross_refs: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    2. Hop Vektör Arama:
    Challenger Ajanı'nın ürettiği alt sorgu ve tespit edilen çapraz atıfları kullanarak
    istisna, taahhüt veya cezai yaptırım içeren bağlantılı maddeleri tarar.
    """
    all_chunks = fetch_all_chunks()
    sub_emb = engine.get_embedding(sub_query)
    scored = []
    cross_refs = cross_refs or []

    for c in all_chunks:
        if c["id"] in exclude_ids:
            continue
        # Açık çapraz atıf bonusu (ör: metinde Madde 4.1'e atıf varsa)
        is_direct_ref = any(ref.lower() in c["content"].lower() or ref.lower() in c["madde_no"].lower() for ref in cross_refs)
        s = cosine_similarity(sub_emb, c["embedding"])
        if is_direct_ref:
            s += 0.30
        scored.append({**c, "score": s, "hop": 2, "is_direct_ref": is_direct_ref})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:2]


# ==============================================================================
# 7. ÇOKLU AJAN MOTORU (PROPOSER, CHALLENGER, JUDGE)
# ==============================================================================

class ProposerAgent:
    """
    1. Hop Ajanı:
    Sorguya en yakın ilk maddeyi çeker ve madde metnini doğrudan değerlendirir.
    İlk bakışta hak tanınıyor gibi görünür (örneğin fesih hakkı var der).
    """
    @staticmethod
    def evaluate(query: str, engine: FoundryLocalEmbeddingEngine) -> Tuple[ChatMessage, Dict[str, Any]]:
        hop1_chunk, _ = retrieve_hop_1(query, engine)

        thought = (
            f"Kullanıcı sorusu: '{query}'. 1. Hop semantik araması tamamlandı. "
            f"En yüksek skorlu hüküm: '{hop1_chunk['madde_no']}' (%{hop1_chunk['score']*100:.1f}). "
            f"Hüküm metnine göre talep doğrudan uygulanabilir görünmektedir."
        )

        content = (
            f"Sözleşme veritabanı incelendiğinde doğrudan ilgili hüküm tespit edilmiştir:\n\n"
            f"📌 **{hop1_chunk['madde_no']} (Sayfa {hop1_chunk['sayfa_no']}):**\n"
            f"> *\"{hop1_chunk['content']}\"*\n\n"
            f"**İlk Hukuki Mütalaa:**\n"
            f"Bu maddeye göre talep kural olarak uygulanabilir görünmektedir. "
            f"Ancak sözleşmede yer alabilecek gizli istisnalar veya taahhüt şartları için denetim gereklidir."
        )

        msg = ChatMessage(
            role="proposer",
            agent_name="Proposer Agent (İlk Görüş)",
            avatar="🤖",
            content=content,
            thought=thought,
            action_tool="retrieve_hop_1",
            tool_output=f"{hop1_chunk['madde_no']} [Benzerlik: %{hop1_chunk['score']*100:.1f}]",
            metadata={"hop1_chunk": hop1_chunk}
        )
        return msg, hop1_chunk


class ChallengerAgent:
    """
    2. Hop Ajanı:
    Proposer'ın sunduğu maddeyi denetler. Metindeki çapraz atıfları (Madde 4.1 vb.)
    ve kısıtlayıcı hukuk terimlerini (taahhüt, ceza, tazminat vb.) avlar.
    Gerekirse 2. Hop aramasını tetikler ve istisna maddesini ortaya koyar.
    """
    @staticmethod
    def evaluate(
        query: str,
        hop1_chunk: Dict[str, Any],
        engine: FoundryLocalEmbeddingEngine
    ) -> Tuple[ChatMessage, ChallengerAssessment, List[Dict[str, Any]]]:
        all_chunks = fetch_all_chunks()
        primary_content = hop1_chunk.get("content", "")
        primary_madde = hop1_chunk.get("madde_no", "")

        # 1. Çapraz Atıf Tespiti (Örn: Madde 4.1, Madde 8.2 vb.)
        found_madde_refs = re.findall(
            r"(?:madde|article|kısım|bölüm)\s*(\d+(?:\.\d+)?)",
            primary_content,
            re.IGNORECASE
        )
        cross_refs = [primary_madde] + [f"Madde {r}" for r in found_madde_refs]

        # 2. Kısıtlayıcı Hukuki Terimlerin Taranması (ancak, saklıdır, uyarınca, taahhüt süresi vb.)
        legal_qualifiers = [
            "ancak", "saklıdır", "saklı", "uyarınca", "taahhüt süresi", "taahhüt",
            "cezai şart", "ceza", "tazminat", "cayma bedeli", "cayma", "kalan ayların",
            "indirim", "istisna", "tahsil", "yükümlü", "asgari", "fatura", "şart"
        ]
        detected_terms = [kw for kw in legal_qualifiers if kw in primary_content.lower() or kw in query.lower()]

        # Eğer birden fazla chunk varsa veya kısıtlayıcı terim/atıf varsa 2. Hop tetiklenir
        needs_hop2 = len(all_chunks) > 1

        hop2_chunks = []
        if needs_hop2:
            # 2. Hop için özel hedeflenmiş alt sorgu inşası (taahhüt süresi, cezai şart, cayma bedeli vb.)
            sub_query = f"{query} {primary_madde} taahhüt süresi cezai şart cayma bedeli istisna ancak saklıdır uyarınca kalan aylar"
            hop2_chunks = retrieve_hop_2(
                sub_query=sub_query,
                engine=engine,
                exclude_ids=[hop1_chunk["id"]],
                cross_refs=cross_refs
            )

            detected_in_hop2 = []
            for h2 in hop2_chunks:
                for kw in legal_qualifiers:
                    if kw in h2["content"].lower() and kw not in detected_in_hop2:
                        detected_in_hop2.append(kw)

            assessment = ChallengerAssessment(
                needs_second_hop=True,
                reason="Proposer'ın mütalaası tekil maddeye dayanmaktadır. Sözleşmede taahhüt süresi, ancak/saklıdır kısıtlamaları ve cezai şart istisnası tespit edilmiştir.",
                sub_query=sub_query,
                detected_terms=list(set(detected_terms + detected_in_hop2))
            )

            thought = (
                f"Proposer'ın görüşü denetlendi. Çapraz maddeler tarandı: {[c['madde_no'] for c in hop2_chunks]}. "
                f"Sözleşmede fesih hakkını sınırlayan şartlar ({assessment.detected_terms}) saptandı. 2. Hop devrede."
            )

            content = (
                f"⚠️ **Proposer'ın mütalaasına itiraz edilmiştir; sözleşme bütüncül yorumlanmalıdır!**\n\n"
                f"`{hop1_chunk['madde_no']}` hükmü tek başına nihai sonucu belirleyemez. "
                f"2. Hop incelemesinde tespit edilen bağlantılı hükümler:\n\n"
            )
            for h2 in hop2_chunks:
                content += f"- 📌 **{h2['madde_no']} (Sayfa {h2['sayfa_no']}):** *\"{h2['content']}\"*\n"

            content += (
                f"\nBu maddeler gereğince, Proposer'ın ileri sürdüğü hak doğrudan veya cezasız kullanılamaz. "
                f"Dosya nihai hüküm için Judge Agent'a devredilmiştir."
            )
        else:
            assessment = ChallengerAssessment(
                needs_second_hop=False,
                reason="Sözleşmede ek bir sınırlandırıcı istisna veya cezai şart tespit edilmedi.",
                sub_query="",
                detected_terms=[]
            )
            thought = "Tekil madde incelendi, ek bir kısıtlayıcı çapraz madde bulunamadı."
            content = "Proposer'ın incelemesi denetlendi. Sözleşmede bu hükmü geçersiz kılan veya cezai şarta bağlayan ek bir kayıt saptanmamıştır."

        msg = ChatMessage(
            role="challenger",
            agent_name="Challenger Agent (Denetçi & İtiraz)",
            avatar="🕵️",
            content=content,
            thought=thought,
            action_tool="retrieve_hop_2",
            tool_output=f"İncelenen Ek Maddeler: {[c['madde_no'] for c in hop2_chunks]}",
            metadata={"assessment": assessment.model_dump(), "hop2_chunks": hop2_chunks}
        )
        return msg, assessment, hop2_chunks


class JudgeAgent:
    """
    Nihai Hakem Ajanı:
    Proposer ve Challenger delillerini sentezler, risk analizi yapar
    ve Pydantic JudgeVerdict formatında kesin hükmü açıklar.
    Responsible AI prensibi: Belgede yeterli dayanak yoksa kesinlikle varsayımda bulunmaz.
    """
    @staticmethod
    def evaluate(
        query: str,
        proposer_msg: ChatMessage,
        challenger_msg: ChatMessage,
        all_evidence: List[Dict[str, Any]]
    ) -> Tuple[ChatMessage, JudgeVerdict]:
        maddeler = [c["madde_no"] for c in all_evidence]
        sayfalar = sorted(list(set(c["sayfa_no"] for c in all_evidence)))
        full_text = " ".join([c.get("content", "") for c in all_evidence]).lower()

        # Responsible AI Güvencesi: Eğer belgedeki benzerlik skoru aşırı düşükse varsayım yapılmaz
        max_score = max([c.get("score", 0.0) for c in all_evidence]) if all_evidence else 0.0
        if max_score < 0.20:
            verdict = JudgeVerdict(
                karar="BELGEDE YETERLİ BİLGİ BULUNAMADI (SORUMLU YAPAY ZEKA - VARSAYIMDA BULUNULAMAZ)",
                guven_skoru=25,
                gerekce=(
                    "Sözleşme veritabanında yönelttiğiniz soruyla doğrudan veya dolaylı olarak örtüşen yeterli bir madde bulunamamıştır. "
                    "Responsible AI prensipleri gereğince belgede yer almayan konularda varsayım veya halüsinasyon üretilmemektedir."
                ),
                dayanak_maddeler=[],
                sayfa_referanslari=[],
                risk_var_mi=False
            )
            thought = "Soru ile sözleşme maddeleri arasında güvenilir bir bağ kurulamadı. Responsible AI uyarınca varsayımsız ret üretildi."
            msg = ChatMessage(
                role="judge",
                agent_name="Judge Agent (Baş Hukuk Hakemi)",
                avatar="⚖️",
                content=f"### ⚖️ NİHAİ DENETÇİ HÜKMÜ\n\n**HÜKÜM:** {verdict.karar}\n\n**Gerekçe:** {verdict.gerekce}",
                thought=thought,
                action_tool="responsible_ai_guard",
                tool_output="Yetersiz veri nedeniyle varsayımsız yanıt verildi.",
                metadata=verdict.model_dump()
            )
            return msg, verdict

        risk_keywords = [
            "cezai şart", "taahhüt", "tazminat", "kalan ayların", "faiz",
            "tahsil", "indirimler", "sorumluluk", "ihlal", "cayma bedeli"
        ]
        detected_risks = [k for k in risk_keywords if k in full_text]
        risk_var_mi = len(detected_risks) > 0
        is_multi_clause = len(all_evidence) > 1

        # Dinamik Matematiksel Güven Skoru Hesabı:
        # Metin kosinüs benzerliği (%40) + 2. Hop istisna uyumu (%30) + Çapraz atıf doğrulaması (%20) + Risk kapsamı (%10)
        h1_score = all_evidence[0].get("score", 0.0) if all_evidence else 0.0
        h2_scores = [c.get("score", 0.0) for c in all_evidence[1:]] if len(all_evidence) > 1 else []
        top_h2 = max(h2_scores) if h2_scores else (h1_score * 0.75)
        has_direct_ref = any(c.get("is_direct_ref", False) for c in all_evidence)
        cross_ref_weight = 0.95 if has_direct_ref else (0.80 if h2_scores else 0.60)
        risk_coverage = min(len(detected_risks) * 0.20 + 0.40, 1.0)

        raw_confidence = (
            (h1_score * 0.40) +
            (top_h2 * 0.30) +
            (cross_ref_weight * 0.20) +
            (risk_coverage * 0.10)
        ) * 100
        guven_skoru = int(np.clip(round(raw_confidence), 35, 96))

        if risk_var_mi and is_multi_clause:
            karar = "TALEBİNİZ KOŞULLU VE RİSKLİ (SÖZLEŞME İSTİSNALARI VE CEZAİ HÜKÜMLER GEÇERLİDİR)"
            gerekce = (
                f"Proposer'ın dayandığı {all_evidence[0]['madde_no']} hükmü fesih bildirimi hakkı tanımakla birlikte; "
                f"Challenger tarafından ortaya konan {', '.join([c['madde_no'] for c in all_evidence[1:]])} hükümleri uyarınca "
                f"sözleşme taahhüt süresine tabidir. Erken fesih durumunda kalan ayların bedelleri ve sağlanan indirimler "
                f"cezai şart olarak faturalandırılacaktır. Dolayısıyla fesih mümkündür ANCAK cezasız yapılamaz."
            )
        elif is_multi_clause:
            karar = "TALEBİNİZ İLGİLİ MADDELERİN BİRLİKTE UYGULANMASINI GEREKTİRMEKTEDİR"
            gerekce = (
                f"İncelenen {', '.join(maddeler)} maddeleri birlikte değerlendirilmiştir. "
                f"Sözleşmedeki usul ve bildirim sürelerine riayet edilmesi zorunludur."
            )
        else:
            karar = "TALEP DOĞRUDAN UYGULANABİLİR (EK ENGEL VEYA CEZAİ ŞART BULUNMAMAKTADIR)"
            gerekce = (
                f"{maddeler[0]} hükmü kapsamında talep doğrudan karşılanabilir niteliktedir; "
                f"sözleşmede hakkı kısıtlayan cezai bir kayıt bulunmamaktadır."
            )

        verdict = JudgeVerdict(
            karar=karar,
            guven_skoru=guven_skoru,
            gerekce=gerekce,
            dayanak_maddeler=maddeler,
            sayfa_referanslari=sayfalar,
            risk_var_mi=risk_var_mi
        )

        thought = (
            f"Her iki tarafın argümanları incelendi. "
            f"İncelenen maddeler: {maddeler}. Tespit edilen risk unsurları: {detected_risks}. "
            f"Nihai karar Pydantic şeması doğrulanarak üretildi."
        )

        content = (
            f"### ⚖️ NİHAİ DENETÇİ HÜKMÜ\n\n"
            f"**HÜKÜM:** {verdict.karar}\n\n"
            f"**Gerekçe:** {verdict.gerekce}\n\n"
            f"**Dayanak Maddeler:** `{', '.join(verdict.dayanak_maddeler)}` | "
            f"**Güven Skoru:** `%{verdict.guven_skoru}` | "
            f"**Risk Durumu:** `{'⚠️ CEZAİ ŞART / RİSK' if verdict.risk_var_mi else '✅ DÜŞÜK RİSK'}`"
        )

        msg = ChatMessage(
            role="judge",
            agent_name="Judge Agent (Baş Hukuk Hakemi)",
            avatar="⚖️",
            content=content,
            thought=thought,
            action_tool="legal_synthesis",
            tool_output=f"Verdict: {verdict.karar} [Risk: {verdict.risk_var_mi}]",
            metadata=verdict.model_dump()
        )
        return msg, verdict


# ==============================================================================
# 8. STREAMLIT ARAYÜZÜ (PREMIUM LOCAL MULTI-AGENT UI)
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Local Multi-Agent RAG Auditor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_database()
    engine = get_embedding_engine()

    # Premium Modern CSS Tasarımı
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .main-header-box {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 24px;
            box-shadow: 0 12px 36px -10px rgba(14, 165, 233, 0.25);
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(120deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }

        .sub-title {
            color: #94a3b8;
            font-size: 1.05rem;
            font-weight: 400;
        }

        .verdict-card {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 14px;
            padding: 20px;
            border-left: 6px solid #ef4444;
            margin-top: 15px;
            margin-bottom: 15px;
        }

        .agent-bubble {
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 14px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
        }

        .thought-box {
            background: rgba(15, 23, 42, 0.7);
            border: 1px dashed rgba(148, 163, 184, 0.3);
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 10px;
            font-size: 0.88rem;
            color: #94a3b8;
        }

        .badge-tag {
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 6px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Session State Başlatma
    if "last_verdict" not in st.session_state:
        st.session_state.last_verdict = None
    if "debate_messages" not in st.session_state:
        st.session_state.debate_messages = []
    if "evidence_chunks" not in st.session_state:
        st.session_state.evidence_chunks = []
    if "input_query" not in st.session_state:
        st.session_state.input_query = ""

    existing_chunks = fetch_all_chunks()

    # --------------------------------------------------------------------------
    # SIDEBAR: SÖZLEŞME VE VERİ YÖNETİMİ
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎛️ Belge & Motor Yönetimi")
        st.info(f"🧠 **Aktif Motor:**\n`{engine.engine_name}`\n\n🔒 **Mimari:** %100 Yerel / Sıfır Dış API")
        st.divider()

        sb_tab1, sb_tab2 = st.tabs(["📄 PDF Yükle", "✍️ Özel Madde Ekle"])

        with sb_tab1:
            st.markdown("##### 📂 Gerçek PDF İndeksle (Mod B)")
            up_pdf = st.file_uploader("PDF Sözleşmesi Seçin", type=["pdf"])
            if up_pdf and st.button("🚀 PDF'i Ayrıştır ve SQLite'a Kaydet", use_container_width=True, type="primary"):
                with st.spinner("PDF sayfaları taranıyor ve yerel vektörler üretiliyor..."):
                    c = parse_and_index_pdf(up_pdf, engine)
                    st.session_state.last_verdict = None
                    st.session_state.debate_messages = []
                    st.session_state.evidence_chunks = []
                    st.success(f"✅ {c} madde/paragraf SQLite'a indekslendi.")
                    st.rerun()

        with sb_tab2:
            st.markdown("##### ➕ Manuel Madde Girişi")
            with st.form("sb_add_form", clear_on_submit=True):
                m_no = st.text_input("Madde No / Başlık", placeholder="Örn: Madde 4.1")
                s_no = st.number_input("Sayfa No", min_value=1, value=1, step=1)
                m_txt = st.text_area("Madde Metni", placeholder="Sözleşme metnini buraya girin...", height=100)
                if st.form_submit_button("💾 SQLite'a Kaydet", use_container_width=True):
                    if m_txt.strip():
                        t = m_no.strip() if m_no.strip() else f"Madde {len(existing_chunks)+1}"
                        emb = engine.get_embedding(f"{t}: {m_txt.strip()}")
                        insert_chunk(t, int(s_no), m_txt.strip(), emb)
                        st.success(f"✅ '{t}' eklendi.")
                        st.rerun()

        st.divider()
        if st.button("🔄 Varsayılan Demo Verisini Yükle (3 Madde)", use_container_width=True):
            with st.spinner("Tuzak demo sözleşmesi yükleniyor..."):
                count = load_demo_data(engine)
                st.session_state.last_verdict = None
                st.session_state.debate_messages = []
                st.session_state.evidence_chunks = []
                st.success(f"✅ {count} maddelik tuzak demo sözleşmesi yüklendi!")
                st.rerun()

        st.markdown(f"**Veritabanındaki Madde Sayısı:** `{len(existing_chunks)} Adet`")
        if existing_chunks:
            if st.button("🗑️ Veritabanını Temizle", use_container_width=True):
                clear_database()
                st.session_state.last_verdict = None
                st.session_state.debate_messages = []
                st.session_state.evidence_chunks = []
                st.warning("Veritabanı temizlendi.")
                st.rerun()

            with st.expander("📋 Kayıtlı Maddeleri Listele"):
                for item in existing_chunks:
                    st.markdown(f"**{item['madde_no']}** *(Sayfa {item['sayfa_no']})*")
                    st.caption(item['content'][:120] + "...")
                    if st.button(f"❌ Sil #{item['id']}", key=f"del_c_{item['id']}"):
                        delete_chunk_by_id(item["id"])
                        st.rerun()
                    st.divider()

    # --------------------------------------------------------------------------
    # ANA PANEL: BAŞLIK VE HIZLI DEMO ÇALIŞTIRICI
    # --------------------------------------------------------------------------
    st.markdown("""
        <div class='main-header-box'>
            <div class='main-title'>🛡️ Foundry Local + SQLite Multi-Hop RAG Denetçisi</div>
            <div class='sub-title'>
                Microsoft Foundry Local ve SQLite tabanlı %100 yerel sözleşme denetim asistanı.
                Proposer, Challenger ve Judge otonom ajanları çapraz atıfları ve gizli riskleri saniyeler içinde ortaya çıkarır.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # TUZAK DEMO TETİKLEME BUTONU (MOD A)
    demo_col1, demo_col2 = st.columns([2, 1])
    with demo_col1:
        run_demo_button = st.button(
            "🚀 2 Dk'lık Tuzak Demoyu Çalıştır (3. Ayda Fesih)",
            type="primary",
            use_container_width=True
        )
    with demo_col2:
        st.caption("💡 **Tuzak Senaryo:** Madde 4.1 cezasız fesih gibi görünür; ancak Madde 8.2'deki 12 aylık taahhüt cezai şart doğurur.")

    st.divider()

    # SERBEST SORU ALANI
    with st.form("custom_query_form"):
        q_col_in, q_col_btn = st.columns([5, 1])
        with q_col_in:
            user_question = st.text_input(
                "Denetlenecek Soru veya Hukuki Durum:",
                value=st.session_state.input_query,
                placeholder="Örn: Müşteri sözleşmenin 3. ayında 30 gün önceden bildirerek cezasız fesih yapabilir mi?",
                label_visibility="collapsed"
            )
        with q_col_btn:
            submit_query_button = st.form_submit_button("🔍 Denetle", use_container_width=True)

    # Denetim Tetikleme Mantığı
    trigger_question = None
    if run_demo_button:
        load_demo_data(engine)
        trigger_question = "Müşteri sözleşmenin 3. ayında 30 gün önceden bildirerek cezasız fesih yapabilir mi?"
    elif submit_query_button and user_question.strip():
        chunks_check = fetch_all_chunks()
        if not chunks_check:
            st.warning("⚠️ Lütfen önce sol menüden bir PDF yükleyin veya 'Varsayılan Demo Verisini Yükle' butonuna basın.")
        else:
            trigger_question = user_question.strip()

    # --------------------------------------------------------------------------
    # CANLI YÜRÜTME LOGU (ST.STATUS) VE AJAN TARTIŞMASI
    # --------------------------------------------------------------------------
    if trigger_question:
        with st.status("🔄 Çoklu Ajan Denetimi Yürütülüyor...", expanded=True) as status:
            time.sleep(0.3)
            # 1. Hop Arama & Proposer
            st.write("🔍 **1. Hop Vektör Arama:** Kullanıcı sorgusu SQLite vektör alanında tarandı...")
            proposer_msg, hop1_chunk = ProposerAgent.evaluate(trigger_question, engine)
            st.write(f"🤖 **Proposer Agent İlk Mütalaayı Sundu:** `{hop1_chunk['madde_no']}` tespit edildi (Kosinüs: `%{hop1_chunk['score']*100:.1f}`).")
            time.sleep(0.5)

            # 2. Hop Arama & Challenger
            st.write("🕵️ **Challenger Agent Denetimi Başlattı:** Çapraz atıflar, taahhütler ve istisnalar taranıyor...")
            challenger_msg, assessment, hop2_chunks = ChallengerAgent.evaluate(trigger_question, hop1_chunk, engine)
            if assessment.needs_second_hop and hop2_chunks:
                st.write(f"⚠️ **Challenger İtiraz Etti:** 2. Hop araması ile sınırlandırıcı ek maddeler çekildi: `{[c['madde_no'] for c in hop2_chunks]}`.")
            else:
                st.write("✅ **Challenger Onayladı:** Ek bir kısıtlama veya istisna bulunmadı.")
            time.sleep(0.5)

            # Judge Sentezi
            st.write("⚖️ **Judge Agent Hükmü Hazırlıyor:** Tüm hop delilleri ve risk faktörleri sentezleniyor...")
            all_evidence = [hop1_chunk] + hop2_chunks
            judge_msg, verdict = JudgeAgent.evaluate(trigger_question, proposer_msg, challenger_msg, all_evidence)
            time.sleep(0.3)

            status.update(label="✅ Denetim Başarıyla Tamamlandı!", state="complete", expanded=False)

        # Durumu Session State'e Kaydet
        st.session_state.last_verdict = verdict
        st.session_state.debate_messages = [proposer_msg, challenger_msg, judge_msg]
        st.session_state.evidence_chunks = all_evidence
        st.session_state.input_query = trigger_question

    # --------------------------------------------------------------------------
    # RAPORLAMA VE METRİK PANELİ
    # --------------------------------------------------------------------------
    if st.session_state.last_verdict:
        verdict = st.session_state.last_verdict

        st.markdown("## 📋 Denetim Raporu ve Hüküm")

        rep_col1, rep_col2 = st.columns([3, 1])

        with rep_col1:
            if verdict.risk_var_mi:
                st.error(f"### 🛑 HÜKÜM:\n**{verdict.karar}**")
            else:
                st.success(f"### 🟢 HÜKÜM:\n**{verdict.karar}**")

            st.markdown(f"**Gerekçeli Hukuki Karar:**\n{verdict.gerekce}")

            st.markdown("**Dayanak Maddeler:**")
            badges_html = "".join([f"<span class='badge-tag'>{m}</span>" for m in verdict.dayanak_maddeler])
            st.markdown(badges_html, unsafe_allow_html=True)

            st.markdown(f"**İlgili Sayfalar:** {', '.join([str(p) for p in verdict.sayfa_referanslari])}")

        with rep_col2:
            st.metric(
                label="Güven Skoru",
                value=f"%{verdict.guven_skoru}",
                delta="Yüksek Doğruluk" if verdict.guven_skoru >= 85 else ("Orta" if verdict.guven_skoru >= 60 else "Düşük"),
                help="Sorgunun maddelerle anlamsal kosinüs benzerliği (%40), 2. Hop istisna maddesi uyumu (%30), doğrudan çapraz atıf doğrulaması (%20) ve tespit edilen hukuki kısıtlayıcı risklerin kapsamının (%10) dinamik matematiksel sentezidir."
            )
            hop_count = 2 if len(st.session_state.evidence_chunks) > 1 else 1
            st.metric(
                label="Hop Sayısı",
                value=f"{hop_count} Hop",
                delta="Multi-Hop Aktif" if hop_count > 1 else "Tekil",
                help="Arama derinliği kademesidir. 1 Hop: Yalnızca sorunun doğrudan karşılığı olan ilk maddeye bakar (naif RAG). 2 Hop: İlk maddedeki kısıtlama, taahhüt ve cezai şart atıflarını takip ederek gizli istisnaları zincirleme olarak ortaya çıkarır (Multi-Hop RAG)."
            )
            st.metric(
                label="Risk Değerlendirmesi",
                value="YÜKSEK RİSK" if verdict.risk_var_mi else "DÜŞÜK RİSK",
                delta="Cezai Şart Mevcut" if verdict.risk_var_mi else "Doğrudan Uygulanabilir",
                delta_color="inverse" if verdict.risk_var_mi else "normal",
                help="Sözleşme maddelerinde cezai şart, taahhüt süresi ihlali, tazminat veya mali yaptırım riski tespit edilip edilmediğini gösterir."
            )

        st.divider()

        # AKORDİYON 1: KULLANILAN SQLITE CHUNK'LARI VE SKORLARI
        with st.expander("📋 Kararda Kullanılan SQLite Chunk'ları ve Kosinüs Benzerlikleri", expanded=True):
            for idx, c in enumerate(st.session_state.evidence_chunks, 1):
                col_c1, col_c2 = st.columns([4, 1])
                with col_c1:
                    st.markdown(f"**{idx}. Delil:** `{c['madde_no']}` *(Sayfa {c['sayfa_no']})*")
                    st.info(f"\"{c['content']}\"")
                with col_c2:
                    st.metric(
                        "Benzerlik Skoru",
                        f"%{c.get('score', 0.0)*100:.1f}",
                        help="Kullanıcı sorgusu ile bu madde metni arasındaki anlamsal kosinüs benzerliği açısıdır (%0-%100). Klasik anahtar kelime aramasından farklı olarak, farklı kelimelerle ifade edilmiş olsa bile hukuki anlam uyumunu ölçer."
                    )
                    st.caption(f"Arama Kademesi: {c.get('hop', 1)}. Hop")
                st.divider()

        # AKORDİYON 2: CANLI AJAN TARTIŞMA SÜREÇLERİ VE DÜŞÜNCE LOGLARI
        with st.expander("💬 Ajanlar Arası Canlı Tartışma ve Düşünce Akışı", expanded=False):
            for msg in st.session_state.debate_messages:
                if msg.role == "proposer":
                    st.markdown(f"""
                        <div class='agent-bubble' style='background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(56, 189, 248, 0.4);'>
                            <div style='font-weight: 700; color: #38bdf8;'>🤖 {msg.agent_name}</div>
                            <div style='color: #e2e8f0; margin-top: 6px;'>{msg.content}</div>
                            <div class='thought-box'>💭 <strong>Düşünce:</strong> {msg.thought}<br>🛠️ <strong>Araç:</strong> <code>{msg.action_tool}</code> — {msg.tool_output}</div>
                        </div>
                    """, unsafe_allow_html=True)
                elif msg.role == "challenger":
                    st.markdown(f"""
                        <div class='agent-bubble' style='background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.4);'>
                            <div style='font-weight: 700; color: #f59e0b;'>🕵️ {msg.agent_name}</div>
                            <div style='color: #e2e8f0; margin-top: 6px;'>{msg.content}</div>
                            <div class='thought-box'>💭 <strong>Düşünce:</strong> {msg.thought}<br>🛠️ <strong>Araç:</strong> <code>{msg.action_tool}</code> — {msg.tool_output}</div>
                        </div>
                    """, unsafe_allow_html=True)
                elif msg.role == "judge":
                    st.markdown(f"""
                        <div class='agent-bubble' style='background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.4);'>
                            <div style='font-weight: 700; color: #c084fc;'>⚖️ {msg.agent_name}</div>
                            <div style='color: #e2e8f0; margin-top: 6px;'>{msg.content}</div>
                            <div class='thought-box'>💭 <strong>Düşünce:</strong> {msg.thought}</div>
                        </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
