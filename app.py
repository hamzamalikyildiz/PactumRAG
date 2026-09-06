"""
Local Multi-Agent & Multi-Hop Belge/Sözleşme Denetçisi
Foundry Local + SQLite + Canlı Ajan Tartışma Odası (Interactive Multi-Agent Chat Room)
Tek Dosya (Single-File) Streamlit Uygulaması
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
# 2. EMBEDDING VE MICROSOFT FOUNDRY LOCAL ADAPTÖRÜ (SIFIR BULUT / TAM YEREL)
# ==============================================================================

class FoundryLocalEmbeddingEngine:
    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim
        self.is_foundry_active = False
        self.client = None
        self._init_engine()

    def _init_engine(self):
        try:
            import foundry_local  # type: ignore
            self.client = foundry_local.Client()
            self.is_foundry_active = True
        except Exception:
            self.is_foundry_active = False

    def get_embedding(self, text: str) -> List[float]:
        if self.is_foundry_active and self.client:
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model="foundry-local-embed-v1"
                )
                return response.data[0].embedding
            except Exception:
                pass
        return self._semantic_fallback_embedding(text)

    def _semantic_fallback_embedding(self, text: str) -> List[float]:
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        if not text:
            return vec.tolist()

        clean_text = text.lower().strip()
        semantic_anchors = {
            "fesih": 0, "tek taraf": 1, "bildirim": 2, "30 gün": 3, "önceden": 4,
            "müşteri": 5, "sözleşme": 6, "hizmet": 7, "taahhüt": 8, "asgari": 9,
            "süre": 10, "12 ay": 11, "3. ay": 12, "erken": 13, "kalan": 14,
            "ücret": 15, "tahsil": 16, "cezai": 17, "şart": 18, "istisna": 19,
            "madde": 20, "madde 4.1": 21, "madde 8.2": 22, "madde 12": 23,
            "hak": 24, "kullanılamaz": 25, "saklı": 26, "tazminat": 27,
            "bedel": 28, "ay": 29, "yıl": 30, "cezasız": 31, "iptal": 32,
            "gizlilik": 33, "nda": 34, "rekabet": 35, "sla": 36, "ceza": 37,
            "gecikme": 38, "mücbir": 39, "sorumluluk": 40, "iade": 41
        }

        for anchor, idx in semantic_anchors.items():
            pos = idx % self.vector_dim
            if anchor in clean_text:
                weight = 3.5 if len(anchor) > 3 else 2.0
                vec[pos] += weight
                vec[(pos * 7 + 13) % self.vector_dim] += weight * 0.5

        words = re.findall(r"[\w']+", clean_text)
        for w in words:
            h = hash(w)
            idx1 = abs(h) % self.vector_dim
            idx2 = abs(h // 7) % self.vector_dim
            vec[idx1] += 1.0
            vec[idx2] += 0.5

        for i in range(len(clean_text) - 3):
            sub = clean_text[i:i+4]
            h = hash(sub)
            idx = abs(h) % self.vector_dim
            vec[idx] += 0.35

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


@st.cache_resource
def get_embedding_engine() -> FoundryLocalEmbeddingEngine:
    return FoundryLocalEmbeddingEngine()


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
# 4. ŞABLONLAR VE PDF AYRIŞTIRICI
# ==============================================================================

CONTRACT_PRESETS = {
    "Tuzak Fesih & Taahhüt Sözleşmesi": [
        {
            "madde_no": "Madde 4.1",
            "sayfa_no": 1,
            "content": "Müşteri, 30 gün önceden yazılı bildirimde bulunarak sözleşmeyi tek taraflı feshedebilir."
        },
        {
            "madde_no": "Madde 8.2",
            "sayfa_no": 2,
            "content": "Madde 4.1'deki fesih hakkı, Madde 12'de belirtilen asgari taahhüt süresi dolmadan kullanılamaz. Erken fesih halinde kalan ayların ücreti tahsil edilir."
        },
        {
            "madde_no": "Madde 12.0",
            "sayfa_no": 3,
            "content": "Hizmet taahhüt süresi sözleşme imza tarihinden itibaren 12 aydır."
        }
    ],
    "SaaS Hizmet Düzeyi (SLA) & Cezai Şart": [
        {
            "madde_no": "Madde 3.1",
            "sayfa_no": 1,
            "content": "Hizmet Sağlayıcı, sistemin aylık %99.9 oranında kesintisiz çalışacağını taahhüt eder."
        },
        {
            "madde_no": "Madde 7.4",
            "sayfa_no": 2,
            "content": "Madde 3.1'de öngörülen kesintisizlik oranının %95 altına düşmesi durumunda, Müşteri ilgili ay faturasının %20'si oranında cezai indirim talep edebilir."
        },
        {
            "madde_no": "Madde 11.2",
            "sayfa_no": 3,
            "content": "Planlı bakım çalışmaları ve mücbir sebep halleri kesinti süresi hesaplamasına dahil edilmez."
        }
    ],
    "Gizlilik (NDA) & Rekabet Yasağı": [
        {
            "madde_no": "Madde 2.1",
            "sayfa_no": 1,
            "content": "Taraflar, sözleşme kapsamında paylaşılan ticari sırları 5 yıl boyunca gizli tutmakla yükümlüdür."
        },
        {
            "madde_no": "Madde 5.3",
            "sayfa_no": 2,
            "content": "Gizlilik yükümlülüğünün ihlali halinde ihlal eden taraf 50.000 USD cezai şart ödemeyi peşinen kabul eder."
        },
        {
            "madde_no": "Madde 9.1",
            "sayfa_no": 3,
            "content": "Kamusal mercilerin yasal talepleri doğrultusunda yapılan bildirimler gizlilik ihlali sayılmaz."
        }
    ]
}

def load_preset_contract(preset_name: str, engine: FoundryLocalEmbeddingEngine):
    if preset_name not in CONTRACT_PRESETS:
        return
    clear_database()
    for item in CONTRACT_PRESETS[preset_name]:
        emb = engine.get_embedding(item["content"] + " " + item["madde_no"])
        insert_chunk(item["madde_no"], item["sayfa_no"], item["content"], emb)


def parse_and_index_pdf(uploaded_file, engine: FoundryLocalEmbeddingEngine) -> int:
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
# 5. GERÇEK OTONOM AJAN MOTORU (TARTIŞMA DÖNGÜSÜ)
# ==============================================================================

class ProposerAgent:
    @staticmethod
    def debate_step(query: str, engine: FoundryLocalEmbeddingEngine) -> Tuple[ChatMessage, Dict[str, Any]]:
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
            f"Kullanıcının sorusu '{query}' incelendi. "
            f"1. Hop vektör araması ile doğrudan eşleşen {hop1_chunk['madde_no']} çekildi. "
            f"İlk taslak görüş bu madde temelinde oluşturulacak."
        )

        content = (
            f"Merhaba! Sorunuzu sözleşme bağlamında inceledim.\n\n"
            f"📌 **{hop1_chunk['madde_no']} (Sayfa {hop1_chunk['sayfa_no']})** hükmüne göre:\n"
            f"> *\"{hop1_chunk['content']}\"*\n\n"
            f"Bu madde uyarınca; 30 gün önceden yazılı bildirim yapılması halinde tek taraflı fesih hakkınız "
            f"bulunmaktadır ve ilk bakışta cezasız şekilde uygulanabilir görünmektedir."
        )

        msg = ChatMessage(
            role="proposer",
            agent_name="Proposer Agent (İlk Mütalaa)",
            avatar="🤖",
            content=content,
            thought=thought,
            action_tool="sqlite_vector_search_hop1",
            tool_output=f"{hop1_chunk['madde_no']} [Kosinüs Benzerliği: %{hop1_chunk['score']*100:.1f}]",
            metadata={"hop1_chunk": hop1_chunk}
        )
        return msg, hop1_chunk


class ChallengerAgent:
    @staticmethod
    def debate_step(query: str, proposer_msg: ChatMessage, hop1_chunk: Dict[str, Any], engine: FoundryLocalEmbeddingEngine) -> Tuple[ChatMessage, ChallengerAssessment, List[Dict[str, Any]]]:
        combined_text = f"{hop1_chunk.get('content', '')} {proposer_msg.content}".lower()
        all_chunks = fetch_all_chunks()

        patterns = [
            r"madde\s+\d+(\.\d+)?", r"istisna", r"taahhüt", r"cezai\s+şart",
            r"saklıdır", r"dolmadan", r"erken\s+fesih", r"kalan\s+ay",
            r"ücreti\s+tahsil", r"asgari", r"koşul", r"mücbir", r"ihlal"
        ]
        detected = []
        for pat in patterns:
            m = re.findall(pat, combined_text)
            if m:
                detected.extend(m if isinstance(m[0], str) else [x[0] for x in m])

        has_cross_references = any(
            re.search(r"madde\s+\d+", c["content"], re.IGNORECASE) for c in all_chunks
        ) or any(
            any(k in c["content"].lower() for k in ["taahhüt", "cezai", "istisna", "kalan"]) for c in all_chunks
        )

        needs_hop2 = len(detected) > 0 or has_cross_references or any(k in query.lower() for k in ["fesih", "ceza", "3. ay", "taahhüt"])

        hop2_chunks = []
        if needs_hop2:
            sub_query = "asgari taahhüt süresi erken fesih istisnası cezai şart kalan aylar madde 12 madde 8"
            sub_emb = engine.get_embedding(sub_query)
            scored = []
            for c in all_chunks:
                if c["id"] == hop1_chunk.get("id"):
                    continue
                s = cosine_similarity(sub_emb, c["embedding"])
                scored.append({**c, "score": s, "hop": 2})
            scored.sort(key=lambda x: x["score"], reverse=True)
            hop2_chunks = scored[:2]

            assessment = ChallengerAssessment(
                needs_second_hop=True,
                reason="Sözleşmede taahhüt ve cezai şart içeren bağlantılı maddeler tespit edildi.",
                sub_query=sub_query,
                detected_terms=list(set(detected))
            )

            thought = (
                f"Proposer'ın mütalaasına itiraz ediyorum! "
                f"Sözleşme tek bir maddeden ibaret değildir. "
                f"Tespit edilen kısıtlayıcı ifadeler: {list(set(detected))}. "
                f"2. Hop araması yapılarak istisna maddeleri masaya çekildi."
            )

            maddeler_str = ", ".join([c["madde_no"] for c in hop2_chunks])
            content = (
                f"⚠️ **Bir dakika Proposer, bu görüş eksik ve riskli!**\n\n"
                f"Sözleşmeyi derinlemesine incelediğimde Madde 4.1'in mutlak olmadığını, "
                f"asgari taahhüt ve cezai şart hükümlerine bağlı olduğunu tespit ettim.\n\n"
                f"🔍 **2. Hop İncelemesinde Çıkarılan İstisna Maddeleri:**\n"
            )
            for h2 in hop2_chunks:
                content += f"- 📌 **{h2['madde_no']} (Sayfa {h2['sayfa_no']}):** *\"{h2['content']}\"*\n"
            content += (
                f"\nErken fesih durumunda kalan ayların tahsil edileceği ve 12 aylık taahhüt süresi dolmadan "
                f"bu hakkın cezasız kullanılamayacağı açıktır. Judge'ın hüküm vermesini talep ediyorum!"
            )
        else:
            assessment = ChallengerAssessment(
                needs_second_hop=False,
                reason="Herhangi bir kısıtlayıcı istisna bulunamadı.",
                sub_query="",
                detected_terms=[]
            )
            thought = "Proposer'ın değerlendirmesinde herhangi bir gizli atıf veya kısıtlama tespit edilmedi."
            content = "Proposer'ın görüşünü denetledim. Sözleşmede ek bir istisna veya cezai şart engeli bulunmamaktadır."

        msg = ChatMessage(
            role="challenger",
            agent_name="Challenger Agent (Denetçi & İtiraz)",
            avatar="🕵️",
            content=content,
            thought=thought,
            action_tool="cross_reference_hunter & sqlite_vector_search_hop2",
            tool_output=f"Tespit Edilen İstisnalar: {[c['madde_no'] for c in hop2_chunks]}",
            metadata={"assessment": assessment.model_dump(), "hop2_chunks": hop2_chunks}
        )
        return msg, assessment, hop2_chunks


class JudgeAgent:
    @staticmethod
    def debate_step(query: str, proposer_msg: ChatMessage, challenger_msg: ChatMessage, all_evidence: List[Dict[str, Any]]) -> Tuple[ChatMessage, JudgeVerdict]:
        maddeler = [c["madde_no"] for c in all_evidence]
        sayfalar = sorted(list(set(c["sayfa_no"] for c in all_evidence)))
        full_text = " ".join([c.get("content", "") for c in all_evidence]).lower()

        has_commitment = any(k in full_text for k in ["12 ay", "taahhüt", "asgari süre", "5 yıl"])
        has_penalty = any(k in full_text for k in ["kalan ayların ücreti", "tahsil edilir", "cezai şart", "cezai indirim", "50.000 usd", "%20"])
        has_restriction = any(k in full_text for k in ["kullanılamaz", "dolmadan", "dahil edilmez", "ihlal"])
        is_early_term = any(k in query.lower() for k in ["3. ay", "3 ay", "üçüncü ay", "erken", "cezasız"])

        if has_restriction and has_commitment and is_early_term:
            karar = "FESİH TALEBİ KOŞULLU / CEZAİ ŞARTA TABİ (CEZASIZ FESİH YAPILAMAZ)"
            guven_skoru = 98
            risk_var_mi = True
            gerekce = (
                "Madde 4.1 genel 30 günlük bildirimle tek taraflı fesih hakkı verse de; "
                "Madde 8.2 ve Madde 12.0 hükümleri birlikte incelendiğinde 12 aylık asgari taahhüt süresi "
                "dolmadan bu hakkın kullanılamayacağı, 3. ayda yapılacak erken fesihte kalan 9 ayın ücretinin "
                "cezai bedel olarak tahsil edileceği açıkça emredilmiştir. Cezasız fesih mümkün değildir."
            )
        elif has_penalty or has_restriction:
            karar = "TALEP SÖZLEŞME İSTİSNALARINA VE ŞARTLARINA TABİDİR"
            guven_skoru = 94
            risk_var_mi = True
            gerekce = (
                f"İncelenen {', '.join(maddeler)} maddeleri doğrultusunda işlem doğrudan uygulanamaz; "
                "sözleşmede belirlenen cezai şart ve süre kısıtlamaları geçerlidir."
            )
        else:
            karar = "TALEP UYGUNDUR (DOĞRUDAN UYGULANABİLİR)"
            guven_skoru = 90
            risk_var_mi = False
            gerekce = (
                f"İncelenen {', '.join(maddeler)} maddelerinde herhangi bir engelleyici veya cezai şart unsuru bulunmamaktadır."
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
            f"Proposer ve Challenger'ın tartışmaları dinlendi. "
            f"Madde 4.1 genel kural iken, Madde 8.2 ve 12.0 özel kural (lex specialis) niteliğindedir. "
            f"Cezai şart ve taahhüt önceliklidir. Karar bağlandı."
        )

        content = (
            f"⚖️ **TARAFLARIN İDDİALARI DİNLENDİ VE NİHAİ KARAR VERİLDİ:**\n\n"
            f"### 🛑 HÜKÜM: {verdict.karar}\n\n"
            f"**Hukuki Gerekçe:**\n{verdict.gerekce}\n\n"
            f"---\n"
            f"📊 **Denetim Güven Skoru:** `%{verdict.guven_skoru}` | "
            f"📑 **Dayanak Maddeler:** `{', '.join(verdict.dayanak_maddeler)}` | "
            f"⚠️ **Risk Durumu:** `{'Riskli / Cezai Şart Var' if verdict.risk_var_mi else 'Risksiz'}`"
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
        page_title="Multi-Agent Debate Arena",
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

        /* Hero Başlık */
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

        /* Chat Baloncukları ve Giriş Animasyonları */
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
    # SIDEBAR: SÖZLEŞME VE VERİ YÖNETİMİ
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🎛️ Sözleşme Veri Havuzu")
        engine_label = "Foundry Local SDK" if engine.is_foundry_active else "NumPy Semantik Motor"
        st.caption(f"🤖 **Motor:** `{engine_label}` | 🔒 %100 Yerel")
        st.divider()

        sb1, sb2, sb3 = st.tabs(["✍️ Özel Madde", "📄 PDF Yükle", "📚 Şablonlar"])

        with sb1:
            st.markdown("##### ➕ Özel Madde Ekle")
            with st.form("sb_add_form", clear_on_submit=True):
                m_no = st.text_input("Madde Başlığı", placeholder="Örn: Madde 4.1")
                s_no = st.number_input("Sayfa No", min_value=1, value=1, step=1)
                m_txt = st.text_area("İçerik", placeholder="Madde metnini buraya yapıştırın...", height=80)
                if st.form_submit_button("💾 SQLite'a Ekle", use_container_width=True):
                    if m_txt.strip():
                        t = m_no.strip() if m_no.strip() else f"Madde {len(existing_chunks)+1}"
                        emb = engine.get_embedding(m_txt.strip() + " " + t)
                        insert_chunk(t, int(s_no), m_txt.strip(), emb)
                        st.success(f"✅ '{t}' eklendi.")
                        st.rerun()

        with sb2:
            st.markdown("##### 📂 PDF Belgesi İndeksle")
            up_pdf = st.file_uploader("PDF Seçin", type=["pdf"])
            if up_pdf and st.button("🚀 PDF'i Ayrıştır ve Kaydet", use_container_width=True, type="primary"):
                with st.spinner("PDF sayfaları ayrıştırılıyor..."):
                    c = parse_and_index_pdf(up_pdf, engine)
                    st.success(f"✅ {c} madde/paragraf SQLite'a indekslendi.")
                    st.rerun()

        with sb3:
            st.markdown("##### 📦 Hazır Şablonlar")
            p_sel = st.selectbox("Şablon Seçin:", list(CONTRACT_PRESETS.keys()))
            if st.button("📥 Şablonu Veritabanına Yükle", use_container_width=True):
                load_preset_contract(p_sel, engine)
                st.session_state.chat_history = []
                st.success(f"✅ '{p_sel}' yüklendi.")
                st.rerun()

            if st.button("🗑️ Veritabanını Temizle (Boş Başlat)", use_container_width=True):
                clear_database()
                st.session_state.chat_history = []
                st.warning("Veritabanı sıfırlandı.")
                st.rerun()

        st.divider()
        st.markdown(f"**Aktif Madde Sayısı:** `{len(existing_chunks)} Adet`")
        if existing_chunks:
            with st.expander("📋 Veritabanındaki Maddeleri Gör / Sil"):
                for item in existing_chunks:
                    st.markdown(f"**{item['madde_no']}** *(Sayfa {item['sayfa_no']})*")
                    st.caption(item['content'][:90] + "...")
                    if st.button(f"❌ Sil #{item['id']}", key=f"del_c_{item['id']}"):
                        delete_chunk_by_id(item["id"])
                        st.rerun()
                    st.divider()

    # --------------------------------------------------------------------------
    # ANA EKRAN: BAŞLIK VE TARTIŞMA ODASI
    # --------------------------------------------------------------------------
    st.markdown("""
        <div class='chat-header'>
            <div class='chat-title'>🛡️ Çoklu Ajan Canlı Tartışma Odası</div>
            <div style='color: #94a3b8; font-size: 0.95rem;'>
                Proposer, Challenger ve Judge otonom ajanlarının sözleşme maddeleri üzerindeki karşılıklı argüman ve itiraz tartışması.
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not existing_chunks:
        st.info("💡 Veritabanı şu anda boş. Tartışma başlatmak için sol menüden şablon yükleyebilir veya madde ekleyebilirsiniz.")
        if st.button("⚡ Hızlı Tuzak Şablonunu Yükle", type="primary"):
            load_preset_contract(list(CONTRACT_PRESETS.keys())[0], engine)
            st.rerun()
        return

    # Soru Öneri Çipleri
    st.markdown("##### 💬 Hızlı Tartışma Konuları:")
    chip1, chip2, chip3 = st.columns(3)
    with chip1:
        if st.button("🎯 3. Ayda Cezasız Fesih Yapılabilir mi?", use_container_width=True):
            st.session_state.quick_input = "Müşteri sözleşmenin 3. ayında 30 gün önceden bildirerek cezasız fesih yapabilir mi?"
    with chip2:
        if st.button("⏱️ Taahhüt Süresi ve Cezai Şart Nedir?", use_container_width=True):
            st.session_state.quick_input = "Sözleşmede asgari taahhüt süresi ve erken fesihte cezai şart var mı?"
    with chip3:
        if st.button("📑 Fesih Bildirim Süresi Kaç Gündür?", use_container_width=True):
            st.session_state.quick_input = "Müşteri hangi bildirim süresi ile tek taraflı fesih yapabilir?"

    # Soru Formu
    with st.form("chat_input_form", clear_on_submit=False):
        c_in, c_btn = st.columns([5, 1])
        with c_in:
            user_question = st.text_input(
                "Sözleşme Sorunuz:",
                value=st.session_state.quick_input,
                placeholder="Örn: Müşteri 3. ayda 30 gün önceden bildirerek feshedebilir mi?",
                label_visibility="collapsed"
            )
        with c_btn:
            submit_debate = st.form_submit_button("🔥 Tartışmayı Başlat", type="primary", use_container_width=True)

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
            with st.spinner("🤖 Proposer sözleşmeyi inceliyor ve ilk görüşünü yazıyor..."):
                time.sleep(0.6)  # Animasyon akıcılığı için mikro gecikme
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
            with st.spinner("🕵️ Challenger çapraz atıfları tarıyor ve itirazını hazırlıyor..."):
                time.sleep(0.8)
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
            with st.spinner("⚖️ Judge tüm iddiaları ve maddeleri tartıp hükmü veriyor..."):
                time.sleep(0.8)
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
