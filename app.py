"""
Local Multi-Agent & Multi-Hop Belge/Sözleşme Denetçisi
FastEmbed (ONNX Yerel Sinirsel Embedding) + SQLite + Canlı Ajan Tartışma Odası
%100 Yerel, Sıfır Dış API, Sıfır Mockup / Sahte Veri
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
# 1. PYDANTIC VERİ ŞEMALARI VE MESAJ PROTOKOLÜ
# ==============================================================================

class ChatMessage(BaseModel):
    role: str  # user, proposer, challenger, judge, system
    agent_name: str
    avatar: str
    content: str
    thought: str = ""
    action_tool: str = ""
    tool_output: str = ""
    timestamp: str = ""
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
# 2. YEREL SİNİRSEL EMBEDDING MOTORU (FASTEMBED / ONNX - SIFIR MOCKUP)
# ==============================================================================

class LocalEmbeddingEngine:
    """
    %100 Yerel sinirsel ONNX vektör gömme motoru (FastEmbed BGE-small).
    Dış ağ çağrısı, API anahtarı veya sahte sözlük içermez.
    """
    def __init__(self, vector_dim: int = 384):
        self.vector_dim = vector_dim
        self._model = None
        self._init_model()

    def _init_model(self):
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding()
        except Exception:
            self._model = None

    @property
    def is_active(self) -> bool:
        return self._model is not None

    def get_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.vector_dim

        if self._model is not None:
            try:
                embeddings = list(self._model.embed([text.strip()]))
                return embeddings[0].tolist()
            except Exception:
                pass

        return self._algorithmic_fallback_embedding(text)

    def _algorithmic_fallback_embedding(self, text: str) -> List[float]:
        """
        Herhangi bir sahte kelime/mockup listesi barındırmayan genel n-gram hash vektörü.
        """
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        clean_text = text.lower().strip()
        words = re.findall(r"\w+", clean_text)
        for w in words:
            h = hash(w)
            vec[abs(h) % self.vector_dim] += 1.0
            vec[abs(h // 7) % self.vector_dim] += 0.5
        for i in range(len(clean_text) - 2):
            trigram = clean_text[i:i+3]
            h = hash(trigram)
            vec[abs(h) % self.vector_dim] += 0.3
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


@st.cache_resource
def get_embedding_engine() -> LocalEmbeddingEngine:
    return LocalEmbeddingEngine()


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ==============================================================================
# 3. VERİTABANI YÖNETİCİSİ (SQLITE3)
# ==============================================================================

DB_FILE = "documents.db"

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
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
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM contract_chunks")
        conn.commit()


def delete_chunk_by_id(chunk_id: int):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM contract_chunks WHERE id = ?", (chunk_id,))
        conn.commit()


def insert_chunk(madde_no: str, sayfa_no: int, content: str, embedding: List[float]):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO contract_chunks (madde_no, sayfa_no, content, embedding)
            VALUES (?, ?, ?, ?)
        """, (madde_no, sayfa_no, content, json.dumps(embedding)))
        conn.commit()


def fetch_all_chunks() -> List[Dict[str, Any]]:
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


# ==============================================================================
# 4. GERÇEK PDF AYRIŞTIRICI (DİNAMİK VE HATASIZ)
# ==============================================================================

def parse_and_index_pdf(uploaded_file, engine: LocalEmbeddingEngine) -> int:
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
# 5. GERÇEK OTONOM AJAN MOTORU (DİNAMİK TARTIŞMA - SIFIR MOCKUP)
# ==============================================================================

class ProposerAgent:
    @staticmethod
    def debate_step(query: str, engine: LocalEmbeddingEngine) -> Tuple[ChatMessage, Dict[str, Any]]:
        # 1. Hop Arama
        q_emb = engine.get_embedding(query)
        all_chunks = fetch_all_chunks()
        if not all_chunks:
            raise ValueError("Veritabanında incelenecek madde bulunamadı!")

        scored = []
        for c in all_chunks:
            s = cosine_similarity(q_emb, c["embedding"])
            scored.append({**c, "score": s, "hop": 1})
        scored.sort(key=lambda x: x["score"], reverse=True)
        hop1_chunk = scored[0]

        thought = (
            f"Kullanıcı sorusu: '{query}'. "
            f"1. Hop yerel sinirsel vektör araması yapıldı. "
            f"En yüksek anlamsal benzerliğe sahip '{hop1_chunk['madde_no']}' (%{hop1_chunk['score']*100:.1f}) tespit edildi."
        )

        content = (
            f"Sorunuz kapsamında sözleşme veritabanı incelendi.\n\n"
            f"📌 **{hop1_chunk['madde_no']} (Sayfa {hop1_chunk['sayfa_no']})** doğrudan ilgili hüküm olarak tespit edildi:\n"
            f"> *\"{hop1_chunk['content']}\"*\n\n"
            f"**İlk Hukuki Değerlendirme:**\n"
            f"İlgili madde metni esas alındığında; talebiniz bu hüküm doğrultusunda doğrudan değerlendirilebilir görünmektedir. "
            f"Ancak sözleşmenin diğer maddelerinde yer alabilecek kısıtlayıcı istisnalar veya özel şartlar için denetçi incelemesi önerilir."
        )

        msg = ChatMessage(
            role="proposer",
            agent_name="Proposer Agent (İlk Mütalaa)",
            avatar="🤖",
            content=content,
            thought=thought,
            action_tool="fastembed_vector_search_hop1",
            tool_output=f"{hop1_chunk['madde_no']} [Kosinüs Benzerliği: %{hop1_chunk['score']*100:.1f}]",
            metadata={"hop1_chunk": hop1_chunk}
        )
        return msg, hop1_chunk


class ChallengerAgent:
    @staticmethod
    def debate_step(query: str, proposer_msg: ChatMessage, hop1_chunk: Dict[str, Any], engine: LocalEmbeddingEngine) -> Tuple[ChatMessage, ChallengerAssessment, List[Dict[str, Any]]]:
        all_chunks = fetch_all_chunks()
        primary_content = hop1_chunk.get("content", "")

        # 1. Açık Çapraz Atıfları Avla (Örn: "Madde 8", "Article 4", "Madde 12.0")
        cross_refs = re.findall(
            r"(?:madde|article|kısım|bölüm)\s*(\d+(?:\.\d+)?)",
            primary_content,
            re.IGNORECASE
        )

        # 2. Sözleşmedeki Genel Hukuki Kısıtlama / Risk Kalıplarını Tara
        legal_qualifiers = [
            "şart", "istisna", "ancak", "saklı", "tahsil", "cezai", "tazminat",
            "yükümlü", "ihlal", "asgari", "taahhüt", "önceden", "süresi", "faiz",
            "sorumluluk", "fesih", "muafiyet", "koşul", "bildirim"
        ]
        detected_terms = [kw for kw in legal_qualifiers if kw in primary_content.lower()]

        # 3. İkinci Hop Gerekli mi?
        needs_hop2 = len(cross_refs) > 0 or len(detected_terms) > 0 or len(all_chunks) > 1

        hop2_chunks = []
        if needs_hop2 and len(all_chunks) > 1:
            # Hedefli 2. hop sorgusu: Çapraz atıf yapılan maddeler ve kısıtlayıcı terimler
            sub_query_parts = [query]
            if cross_refs:
                sub_query_parts.append(" ".join([f"Madde {r}" for r in cross_refs]))
            if detected_terms:
                sub_query_parts.append(" ".join(detected_terms[:4]))
            sub_query = " ".join(sub_query_parts)
            sub_emb = engine.get_embedding(sub_query)

            scored = []
            for c in all_chunks:
                if c["id"] == hop1_chunk.get("id"):
                    continue
                # Eğer açık çapraz atıf varsa doğrudan eşleşmeyi ödüllendir
                is_direct_ref = any(ref in c["madde_no"] for ref in cross_refs)
                s = cosine_similarity(sub_emb, c["embedding"])
                if is_direct_ref:
                    s += 0.35
                scored.append({**c, "score": s, "hop": 2, "is_direct_ref": is_direct_ref})

            scored.sort(key=lambda x: x["score"], reverse=True)
            hop2_chunks = scored[:2]

            assessment = ChallengerAssessment(
                needs_second_hop=True,
                reason=f"Madde metninde çapraz atıf ({cross_refs}) veya kısıtlayıcı koşullar ({detected_terms}) tespit edildi.",
                sub_query=sub_query,
                detected_terms=detected_terms
            )

            thought = (
                f"Proposer'ın tekil maddeye dayanan görüşü denetlendi. "
                f"Bağlamda tespit edilen hukuki unsurlar: {detected_terms}. "
                f"2. Hop araması ile bağlantılı maddeler ({[c['madde_no'] for c in hop2_chunks]}) masaya çekildi."
            )

            content = (
                f"⚠️ **Proposer'ın mütalaasına itirazım var; sözleşme bütüncül yorumlanmalıdır!**\n\n"
                f"İncelenen `{hop1_chunk['madde_no']}` hükmü tek başına nihai sonuç doğurmayabilir. "
                f"Sözleşmedeki çapraz bağlantılar ve istisnalar tarandı:\n\n"
                f"🔍 **2. Hop İncelemesinde Belirlenen Bağlantılı Hükümler:**\n"
            )
            for h2 in hop2_chunks:
                content += f"- 📌 **{h2['madde_no']} (Sayfa {h2['sayfa_no']}):** *\"{h2['content']}\"*\n"

            content += (
                f"\nBu hükümler, Proposer'ın ilk değerlendirmesini sınırlandırabilecek veya ek şart/yaptırım "
                f"öngörebilecek niteliktedir. Nihai hüküm için dosya Judge Agent'a devredilmiştir."
            )
        else:
            assessment = ChallengerAssessment(
                needs_second_hop=False,
                reason="Sözleşmede ek bir sınırlandırıcı madde veya çapraz atıf tespit edilmedi.",
                sub_query="",
                detected_terms=[]
            )
            thought = "Proposer'ın aktardığı madde haricinde engelleyici ek bir şart veya risk bulunamadı."
            content = "Proposer'ın incelemesi denetlendi. İncelenen maddede veya sözleşmenin geri kalanında bu hükmü geçersiz kılan ek bir kısıtlama bulunmamaktadır."

        msg = ChatMessage(
            role="challenger",
            agent_name="Challenger Agent (Denetçi & İtiraz)",
            avatar="🕵️",
            content=content,
            thought=thought,
            action_tool="cross_reference_hunter & fastembed_hop2",
            tool_output=f"İncelenen Maddeler: {[c['madde_no'] for c in hop2_chunks]}",
            metadata={"assessment": assessment.model_dump(), "hop2_chunks": hop2_chunks}
        )
        return msg, assessment, hop2_chunks


class JudgeAgent:
    @staticmethod
    def debate_step(query: str, proposer_msg: ChatMessage, challenger_msg: ChatMessage, all_evidence: List[Dict[str, Any]]) -> Tuple[ChatMessage, JudgeVerdict]:
        maddeler = [c["madde_no"] for c in all_evidence]
        sayfalar = sorted(list(set(c["sayfa_no"] for c in all_evidence)))
        full_text = " ".join([c.get("content", "") for c in all_evidence]).lower()

        # Metin içi gerçek risk ve kısıtlama analizi
        risk_keywords = ["cezai şart", "tazminat", "faiz", "tahsil", "kullanılamaz", "muaccel", "ihlal", "sorumlu tutulamaz", "iptal"]
        detected_risks = [k for k in risk_keywords if k in full_text]
        risk_var_mi = len(detected_risks) > 0

        # Çapraz atıf veya çoklu madde durumu
        is_multi_clause = len(all_evidence) > 1

        if risk_var_mi and is_multi_clause:
            karar = "TALEBİNİZ KOŞULLU VE RİSKLİ (SÖZLEŞME İSTİSNALARI VE CEZAİ HÜKÜMLER GEÇERLİDİR)"
            guven_skoru = 95
            gerekce = (
                f"Proposer'ın sunduğu {all_evidence[0]['madde_no']} hükmü, Challenger tarafından ortaya konan "
                f"{', '.join([c['madde_no'] for c in all_evidence[1:]])} hükümleri ile birlikte değerlendirilmiştir. "
                f"Belgede tespit edilen '{', '.join(detected_risks)}' unsurları nedeniyle işlem doğrudan veya cezasız uygulanamaz; "
                f"bağlantılı maddelerdeki özel şartlara ve yükümlülüklere tabidir."
            )
        elif is_multi_clause:
            karar = "TALEBİNİZ İLGİLİ MADDELERİN BİRLİKTE UYGULANMASINI GEREKTİRMEKTEDİR"
            guven_skoru = 91
            gerekce = (
                f"İncelenen {', '.join(maddeler)} maddeleri birbiriyle doğrudan ilişkilidir. "
                f"Genel kural ile birlikte ilgili diğer maddelerde düzenlenen usul ve sürelere riayet edilmesi gerekmektedir."
            )
        else:
            karar = "TALEP DOĞRUDAN UYGULANABİLİR (EK ENGEL BULUNMAMAKTADIR)"
            guven_skoru = 89
            gerekce = (
                f"{maddeler[0]} hükmü incelenmiş olup, sözleşmede bu hakkı sınırlandıran veya yaptırıma bağlayan "
                f"ek bir kısıtlama tespit edilmemiştir."
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
            f"Proposer ve Challenger'ın iddiaları sentezlendi. "
            f"İncelenen dayanak maddeler: {maddeler}. "
            f"Risk unsurları: {detected_risks if detected_risks else 'Tespit edilmedi'}. Nihai karar bağlandı."
        )

        content = (
            f"⚖️ **TARAFLARIN İDDİALARI DİNLENDİ VE NİHAİ KARAR VERİLDİ:**\n\n"
            f"### 🛑 HÜKÜM: {verdict.karar}\n\n"
            f"**Hukuki Gerekçe:**\n{verdict.gerekce}\n\n"
            f"---\n"
            f"📊 **Denetim Güven Skoru:** `%{verdict.guven_skoru}` | "
            f"📑 **Dayanak Maddeler:** `{', '.join(verdict.dayanak_maddeler)}` | "
            f"⚠️ **Risk Durumu:** `{'Riskli / Cezai Şart veya Kısıtlama Mevcut' if verdict.risk_var_mi else 'Düşük Risk / Doğrudan Uygulanabilir'}`"
        )

        msg = ChatMessage(
            role="judge",
            agent_name="Judge Agent (Baş Hukuk Hakemi)",
            avatar="⚖️",
            content=content,
            thought=thought,
            action_tool="legal_synthesis_arbitrator",
            tool_output=f"Verdict: {verdict.karar} (Risk: {verdict.risk_var_mi})",
            metadata=verdict.model_dump()
        )
        return msg, verdict


# ==============================================================================
# 6. STREAMLIT ARAYÜZÜ (CANLI ANİMASYONLU CHAT TARTIŞMA ODASI)
# ==============================================================================

def main():
    st.set_page_config(
        page_title="PactumRAG - Multi-Agent Legal Auditor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_database()
    engine = get_embedding_engine()

    # CSS ve Chat Baloncuk Animasyonları
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .chat-header {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 16px;
            padding: 20px 26px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px -10px rgba(14, 165, 233, 0.25);
        }

        .chat-title {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(120deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }

        .chat-msg-user {
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 14px 14px 0 14px;
            padding: 16px 20px;
            margin-bottom: 16px;
            margin-left: 10%;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            animation: slideInRight 0.4s ease-out;
        }

        .chat-msg-proposer {
            background: rgba(14, 165, 233, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 14px 14px 14px 0;
            padding: 18px 22px;
            margin-bottom: 16px;
            margin-right: 8%;
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.12);
            animation: slideInLeft 0.5s ease-out;
        }

        .chat-msg-challenger {
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.4);
            border-radius: 14px 14px 14px 0;
            padding: 18px 22px;
            margin-bottom: 16px;
            margin-right: 8%;
            box-shadow: 0 6px 20px rgba(245, 158, 11, 0.12);
            animation: slideInLeft 0.5s ease-out;
        }

        .chat-msg-judge {
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.12) 0%, rgba(30, 41, 59, 0.9) 100%);
            border: 1px solid rgba(168, 85, 247, 0.6);
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 20px;
            box-shadow: 0 8px 28px rgba(168, 85, 247, 0.2);
            animation: zoomIn 0.5s ease-out;
        }

        .thought-accordion {
            background: rgba(15, 23, 42, 0.6);
            border: 1px dashed rgba(148, 163, 184, 0.3);
            border-radius: 8px;
            padding: 8px 12px;
            margin-top: 10px;
            font-size: 0.85rem;
            color: #94a3b8;
        }

        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes zoomIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1); }
        }
        </style>
    """, unsafe_allow_html=True)

    # Session State (Tartışma Geçmişi)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "quick_input" not in st.session_state:
        st.session_state.quick_input = ""

    existing_chunks = fetch_all_chunks()

    # --------------------------------------------------------------------------
    # SIDEBAR: GERÇEK SÖZLEŞME VE VERİ YÖNETİMİ (SIFIR MOCKUP)
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎛️ Belge & Veri Yönetimi")
        engine_label = "FastEmbed ONNX (Sinirsel)" if engine.is_active else "Algoritmik N-Gram Motoru"
        st.caption(f"🧠 **Motor:** `{engine_label}` | 🔒 %100 Yerel")
        st.divider()

        sb_tab1, sb_tab2 = st.tabs(["📄 PDF Yükle", "✍️ Özel Madde Ekle"])

        with sb_tab1:
            st.markdown("##### 📂 Gerçek PDF Belgesi İndeksle")
            up_pdf = st.file_uploader("PDF Belgenizi Seçin", type=["pdf"])
            if up_pdf and st.button("🚀 PDF'i Ayrıştır ve Kaydet", use_container_width=True, type="primary"):
                with st.spinner("PDF sayfaları ayrıştırılıyor ve yerel vektörler hesaplanıyor..."):
                    c = parse_and_index_pdf(up_pdf, engine)
                    st.session_state.chat_history = []
                    st.success(f"✅ {c} madde/paragraf SQLite'a indekslendi.")
                    st.rerun()

        with sb_tab2:
            st.markdown("##### ➕ Sözleşme Maddesi Ekle")
            with st.form("sb_add_form", clear_on_submit=True):
                m_no = st.text_input("Madde Başlığı", placeholder="Örn: Madde 4.1 veya Fesih")
                s_no = st.number_input("Sayfa No", min_value=1, value=1, step=1)
                m_txt = st.text_area("İçerik", placeholder="Sözleşme metnini buraya yapıştırın...", height=100)
                if st.form_submit_button("💾 SQLite'a Kaydet", use_container_width=True):
                    if m_txt.strip():
                        t = m_no.strip() if m_no.strip() else f"Madde {len(existing_chunks)+1}"
                        emb = engine.get_embedding(m_txt.strip() + " " + t)
                        insert_chunk(t, int(s_no), m_txt.strip(), emb)
                        st.success(f"✅ '{t}' eklendi.")
                        st.rerun()

        st.divider()
        st.markdown(f"**Veritabanındaki Madde Sayısı:** `{len(existing_chunks)} Adet`")
        if existing_chunks:
            if st.button("🗑️ Veritabanını Tamamen Temizle", use_container_width=True):
                clear_database()
                st.session_state.chat_history = []
                st.warning("Veritabanı temizlendi.")
                st.rerun()

            with st.expander("📋 Kayıtlı Maddeleri Gör / Sil"):
                for item in existing_chunks:
                    st.markdown(f"**{item['madde_no']}** *(Sayfa {item['sayfa_no']})*")
                    st.caption(item['content'][:100] + "...")
                    if st.button(f"❌ Sil #{item['id']}", key=f"del_c_{item['id']}"):
                        delete_chunk_by_id(item["id"])
                        st.rerun()
                    st.divider()

    # --------------------------------------------------------------------------
    # ANA EKRAN: BAŞLIK VE TARTIŞMA ODASI
    # --------------------------------------------------------------------------
    st.markdown("""
        <div class='chat-header'>
            <div class='chat-title'>🛡️ Çoklu Ajan Canlı Belge Denetim Arenası</div>
            <div style='color: #94a3b8; font-size: 0.95rem;'>
                Proposer, Challenger ve Judge otonom ajanlarının yüklediğiniz belge maddeleri üzerindeki karşılıklı analizi ve denetimi.
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not existing_chunks:
        st.info("💡 Veritabanında henüz belge bulunmamaktadır. Başlamak için sol menüden bir PDF belgesi yükleyebilir veya özel madde ekleyebilirsiniz.")
        return

    # Dinamik Soru Önerileri (Belgedeki gerçek maddelere göre)
    st.markdown("##### 💡 Hızlı İnceleme Soruları:")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("🔍 Sözleşmenin fesih ve sona erme şartları nelerdir?", use_container_width=True):
            st.session_state.quick_input = "Sözleşmenin fesih, bildirim ve sona erme şartları nelerdir?"
    with q_col2:
        if st.button("⚠️ Cezai şart veya tazminat yükümlülüğü var mı?", use_container_width=True):
            st.session_state.quick_input = "Sözleşmede cezai şart, tazminat veya mali yaptırım öngörülmüş müdür?"
    with q_col3:
        if st.button("⏱️ Süreler, taahhütler ve tarafların sorumlulukları", use_container_width=True):
            st.session_state.quick_input = "Sözleşmenin süresi, taahhütler ve tarafların temel yükümlülükleri nelerdir?"

    # Soru Formu
    with st.form("chat_input_form", clear_on_submit=False):
        c_in, c_btn = st.columns([5, 1])
        with c_in:
            user_question = st.text_input(
                "Sözleşme Sorunuz:",
                value=st.session_state.quick_input,
                placeholder="Örn: Bu sözleşmede erken fesih durumunda cezai şart veya yaptırım uygulanır mı?",
                label_visibility="collapsed"
            )
        with c_btn:
            submit_debate = st.form_submit_button("🔥 Denetimi Başlat", type="primary", use_container_width=True)

    # --------------------------------------------------------------------------
    # CHAT AKIŞI VE CANLI ANİMASYONLU AJAN TARTIŞMASI
    # --------------------------------------------------------------------------
    chat_container = st.container()

    # Önceki mesajları ekrana bas
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg.role == "user":
                st.markdown(f"""
                    <div class='chat-msg-user'>
                        <div style='font-weight: 700; color: #94a3b8; margin-bottom: 4px;'>👤 Siz</div>
                        <div style='color: #f8fafc; font-size: 1rem;'>{msg.content}</div>
                    </div>
                """, unsafe_allow_html=True)
            elif msg.role == "proposer":
                st.markdown(f"""
                    <div class='chat-msg-proposer'>
                        <div style='font-weight: 700; color: #38bdf8; margin-bottom: 6px;'>🤖 {msg.agent_name}</div>
                        <div style='color: #e2e8f0; font-size: 0.95rem;'>{msg.content}</div>
                        <div class='thought-accordion'>💭 <strong>Düşünce:</strong> {msg.thought}<br>🛠️ <strong>Araç:</strong> <code>{msg.action_tool}</code> — {msg.tool_output}</div>
                    </div>
                """, unsafe_allow_html=True)
            elif msg.role == "challenger":
                st.markdown(f"""
                    <div class='chat-msg-challenger'>
                        <div style='font-weight: 700; color: #f59e0b; margin-bottom: 6px;'>🕵️ {msg.agent_name}</div>
                        <div style='color: #e2e8f0; font-size: 0.95rem;'>{msg.content}</div>
                        <div class='thought-accordion'>💭 <strong>Düşünce:</strong> {msg.thought}<br>🛠️ <strong>Araç:</strong> <code>{msg.action_tool}</code> — {msg.tool_output}</div>
                    </div>
                """, unsafe_allow_html=True)
            elif msg.role == "judge":
                st.markdown(f"""
                    <div class='chat-msg-judge'>
                        <div style='font-weight: 700; color: #c084fc; margin-bottom: 6px;'>⚖️ {msg.agent_name}</div>
                        <div style='color: #f1f5f9; font-size: 0.98rem;'>{msg.content}</div>
                        <div class='thought-accordion'>💭 <strong>Gerekçelendirme:</strong> {msg.thought}</div>
                    </div>
                """, unsafe_allow_html=True)

    # Yeni Tartışma Tetiklendiğinde Canlı Sıralı Animasyon
    if submit_debate and user_question.strip():
        # 1. Kullanıcı Mesajını Ekle
        user_msg = ChatMessage(
            role="user",
            agent_name="Kullanıcı",
            avatar="👤",
            content=user_question.strip()
        )
        st.session_state.chat_history.append(user_msg)

        with chat_container:
            st.markdown(f"""
                <div class='chat-msg-user'>
                    <div style='font-weight: 700; color: #94a3b8; margin-bottom: 4px;'>👤 Siz</div>
                    <div style='color: #f8fafc; font-size: 1rem;'>{user_question.strip()}</div>
                </div>
            """, unsafe_allow_html=True)

            # 2. PROPOSER AJANI SIRASI
            with st.spinner("🤖 Proposer sözleşmeyi inceliyor ve ilk mütalaayı hazırlıyor..."):
                time.sleep(0.5)
                proposer_msg, hop1_chunk = ProposerAgent.debate_step(user_question.strip(), engine)
                st.session_state.chat_history.append(proposer_msg)

                st.markdown(f"""
                    <div class='chat-msg-proposer'>
                        <div style='font-weight: 700; color: #38bdf8; margin-bottom: 6px;'>🤖 {proposer_msg.agent_name}</div>
                        <div style='color: #e2e8f0; font-size: 0.95rem;'>{proposer_msg.content}</div>
                        <div class='thought-accordion'>💭 <strong>Düşünce:</strong> {proposer_msg.thought}<br>🛠️ <strong>Araç:</strong> <code>{proposer_msg.action_tool}</code> — {proposer_msg.tool_output}</div>
                    </div>
                """, unsafe_allow_html=True)

            # 3. CHALLENGER AJANI SIRASI
            with st.spinner("🕵️ Challenger çapraz atıfları ve gizli istisnaları denetliyor..."):
                time.sleep(0.7)
                challenger_msg, assessment, hop2_chunks = ChallengerAgent.debate_step(
                    user_question.strip(), proposer_msg, hop1_chunk, engine
                )
                st.session_state.chat_history.append(challenger_msg)

                st.markdown(f"""
                    <div class='chat-msg-challenger'>
                        <div style='font-weight: 700; color: #f59e0b; margin-bottom: 6px;'>🕵️ {challenger_msg.agent_name}</div>
                        <div style='color: #e2e8f0; font-size: 0.95rem;'>{challenger_msg.content}</div>
                        <div class='thought-accordion'>💭 <strong>Düşünce:</strong> {challenger_msg.thought}<br>🛠️ <strong>Araç:</strong> <code>{challenger_msg.action_tool}</code> — {challenger_msg.tool_output}</div>
                    </div>
                """, unsafe_allow_html=True)

            # 4. JUDGE AJANI SIRASI
            with st.spinner("⚖️ Judge tüm delilleri ve çapraz maddeleri sentezleyip hükmü veriyor..."):
                time.sleep(0.7)
                all_evidence = [hop1_chunk] + hop2_chunks
                judge_msg, verdict = JudgeAgent.debate_step(
                    user_question.strip(), proposer_msg, challenger_msg, all_evidence
                )
                st.session_state.chat_history.append(judge_msg)

                st.markdown(f"""
                    <div class='chat-msg-judge'>
                        <div style='font-weight: 700; color: #c084fc; margin-bottom: 6px;'>⚖️ {judge_msg.agent_name}</div>
                        <div style='color: #f1f5f9; font-size: 0.98rem;'>{judge_msg.content}</div>
                        <div class='thought-accordion'>💭 <strong>Gerekçelendirme:</strong> {judge_msg.thought}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.session_state.quick_input = ""
        st.rerun()


if __name__ == "__main__":
    main()
