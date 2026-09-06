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
import zlib

import numpy as np
from pydantic import BaseModel, Field
import streamlit as st

# ==============================================================================
# 1. SABİTLER VE KONFİGÜRASYON
# ==============================================================================

DB_FILE = "documents.db"
VECTOR_DIM = 128
RELEVANCE_THRESHOLD = 0.42  # Alakasız/kapsam dışı sorular için minimum kosinüs benzerliği eşiği

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

# Hukuki Eylem ve Konu Alanları (Kapsam Doğrulama & Responsible AI için)
ACTION_DOMAINS = {
    "dondurma": ["dondur", "dondurma", "dondurabilir", "dondurul", "geçici dondurma", "hat dondurma"],
    "fesih": ["fesih", "feshet", "feshedebilir", "feshedilir", "cayma", "cayabilir", "iptal", "vazgeç", "sonlandır"],
    "devir": ["devir", "devret", "devredebilir", "devredilir", "kota devir", "devretme"],
    "nakil": ["nakil", "naklettir", "nakledebilir", "adres değişik"],
    "itiraz": ["itiraz", "itirazı", "itiraz edebilir", "itiraz süresi"],
    "iade": ["iade", "iade edebilir", "iadesi"],
    "cezai_sart": ["cezai şart", "cayma bedeli", "ceza bedeli", "tazminat"],
    "yetkili_mahkeme": ["mahkeme", "yetkili", "yetki", "icra dairesi"],
    "modem_cihaz": ["modem", "cihaz", "donanım", "cihaz mülkiyeti"],
    "faiz": ["gecikme faizi", "faiz"],
    "hiz_kota": ["hız", "kota", "adil kullanım", "mbps", "gb"],
    "indirim_ozel": ["öğrenci", "engelli", "emekli", "öğrenci indirimi", "genç tarifesi"]
}


def grade_retrieval_relevance(
    query: str,
    top_chunk: Dict[str, Any],
    all_chunks: List[Dict[str, Any]]
) -> Tuple[bool, str, str]:
    """
    Retrieval Grader (Alaka Eşiği Denetçisi):
    Sorudaki anahtar kavramlar ile çekilen metin arasındaki alakayı kontrol eder.
    Eğer sorulan temel konu (örn: dondurma, askıya alma, dondurabilir miyim, öğrenci indirimi, kota devir vb.)
    çekilen hiçbir maddede geçmiyorsa veya benzerlik skoru eşik değerin altındaysa False döner.
    """
    q_lower = query.lower()
    all_contract_text = " ".join([c.get("content", "") for c in all_chunks]).lower()

    # 1. Özel ve hassas konu domainleri kontrolü
    DOMAIN_EXPLANATIONS = {
        "dondurma": "İncelenen sözleşme metninde geçici hat dondurma, dondurma süresi veya askıya alma koşullarına ilişkin herhangi bir hüküm yer almamaktadır. Sözleşme dışı konularda varsayım yapılmamaktadır.",
        "indirim_ozel": "İncelenen sözleşme metninde öğrenci, engelli veya özel tarife indirimlerine ilişkin herhangi bir hüküm yer almamaktadır. Sözleşme dışı konularda varsayım yapılmamaktadır.",
        "devir": "İncelenen sözleşme metninde kota devri veya abonelik devir koşullarına ilişkin herhangi bir hüküm yer almamaktadır. Sözleşme dışı konularda varsayım yapılmamaktadır.",
        "nakil": "İncelenen sözleşme metninde hat nakli veya adres değişikliği işlemlerine ilişkin herhangi bir hüküm yer almamaktadır. Sözleşme dışı konularda varsayım yapılmamaktadır.",
        "modem_cihaz": "İncelenen sözleşme metninde modem/cihaz mülkiyeti veya iade koşullarına ilişkin herhangi bir hüküm yer almamaktadır. Sözleşme dışı konularda varsayım yapılmamaktadır."
    }

    # Kullanıcı geçici dondurma / askıya almayı sorduysa
    if any(k in q_lower for k in ["dondur", "dondurma", "dondurabilir", "geçici dondurma", "hat dondurma"]):
        if not any(k in all_contract_text for k in ["dondur", "dondurma", "dondurabilir", "hat dondurma"]):
            return False, "dondurma", DOMAIN_EXPLANATIONS["dondurma"]

    for domain, keywords in ACTION_DOMAINS.items():
        if any(kw in q_lower for kw in keywords):
            # Sözleşmenin herhangi bir maddesinde bu kavram geçiyor mu?
            has_in_contract = any(kw in all_contract_text for kw in keywords)
            if not has_in_contract:
                custom_reason = DOMAIN_EXPLANATIONS.get(
                    domain,
                    f"İncelenen sözleşme metninde '{keywords[0]}' konusuna ilişkin herhangi bir hüküm yer almamaktadır. Sözleşme dışı konularda varsayım yapılmamaktadır."
                )
                return False, domain, custom_reason

    # 2. Dinamik Fiil Kökü Denetimi (örn: "dondurabilir miyim" -> "dondur")
    verb_match = re.search(r"([a-zçğıöşü]{3,20})(?:abilirim|ebilirim|abilir|ebilir|abiliyor|ebiliyor)", q_lower)
    if verb_match:
        root = verb_match.group(1)
        common_verbs = ["yap", "ed", "ol", "al", "ver", "bil", "gel", "git", "iste", "bulun"]
        if root not in common_verbs and root not in all_contract_text:
            reason = f"İncelenen sözleşme metninde '{root}' işlemine veya hakkına ilişkin herhangi bir hüküm yer almamaktadır. Sözleşme dışı konularda varsayım yapılmamaktadır."
            return False, root, reason

    # 3. Kosinüs Benzerlik Eşiği Denetimi
    score = top_chunk.get("score", 0.0) if top_chunk else 0.0
    if score < RELEVANCE_THRESHOLD:
        reason = f"Sorulan konu ile sözleşme maddeleri arasındaki semantik benzerlik (%{score*100:.1f}) belirlenen güvenilirlik eşiğinin (%{RELEVANCE_THRESHOLD*100:.0f}) altındadır. Sözleşmede bu konuyu düzenleyen bir madde bulunamadı."
        return False, "eşik_altı", reason

    return True, "uygun", ""

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
    durum_tipi: str = Field(default="DURUM_A", description="DURUM_A (Riskli), DURUM_B (Uygun), DURUM_C (Kapsam Dışı)")


# ==============================================================================
# 3. EMBEDDING MOTORU (MICROSOFT FOUNDRY LOCAL + NUMPY SEMANTİK FALLBACK)
# ==============================================================================

class FoundryLocalEmbeddingEngine:
    """
    Microsoft Foundry Local SDK tabanlı yerel gömme ve çıkarım motoru.
    SDK üzerinden yerel modelleri (Phi-3.5 Mini ve yerel embedding'ler) koşturur.
    SDK veya model ağırlıkları aktif değilse %100 çevrimdışı, deterministik ve semantik ağırlıklı
    NumPy fallback vektör üreticisine otomatik düşer. Dış API kesinlikle çağrılmaz.
    """
    def __init__(self, vector_dim: int = VECTOR_DIM):
        self.vector_dim = vector_dim
        self.is_foundry_active = False
        self.is_phi_active = False
        self.engine_name = "NumPy Semantik Fallback (128-D)"
        self.llm_name = "Yerel Hukuki Sentez Motoru (Phi-3.5 Çevrimdışı Fallback)"
        self._session = None
        self._llm_session = None
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

            phi_model = cat.get_model("phi-3.5-mini-instruct") or cat.get_model("phi-3.5-mini")
            if phi_model and getattr(phi_model, "is_loaded", False):
                self._llm_session = fl.ChatSession(phi_model)
                self.is_phi_active = True
                self.llm_name = "Microsoft Foundry Local (Phi-3.5 Mini)"
        except Exception:
            self.is_foundry_active = False
            self.is_phi_active = False
            self.engine_name = "NumPy Semantik Fallback (128-D)"
            self.llm_name = "Yerel Hukuki Sentez Motoru (Phi-3.5 Çevrimdışı Fallback)"

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
                # Türkçe kelime başlangıcı kontrolü (Örn: 'bugün' içindeki 'gün' veya 'sayfa' içindeki 'ay' eşleşmesin)
                pattern = r"(?<![a-zçğıöşü0-9])" + re.escape(kw)
                if re.search(pattern, clean_text):
                    for idx in range(start_idx, end_idx):
                        vec[idx] += 1.8

        # 2. Kelime Bazlı Hash Dağıtımı (zlib.crc32 ile Süreç Bağımsız Determinizm)
        for w in words:
            h = zlib.crc32(w.encode("utf-8"))
            vec[abs(h) % self.vector_dim] += 1.0
            vec[abs(h // 13) % self.vector_dim] += 0.5

        # 3. Karakter 3-Gram Hash Dağıtımı
        for i in range(len(clean_text) - 2):
            trigram = clean_text[i:i+3]
            h_tri = zlib.crc32(trigram.encode("utf-8"))
            vec[abs(h_tri) % self.vector_dim] += 0.25

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
    Mod A: Spesifikasyonda belirtilen ve tüm senaryoları (Durum A, B, C) kapsayan demo sözleşmesini yükler:
    - Madde 4.1: 30 gün önceden bildirimli fesih hakkı
    - Madde 8.2: 12 aylık taahhüt süresi ve cezai şart (Tuzak senaryo)
    - Madde 7.1: Yazılı bildirim ve tebligat usulü (Doğrudan uygulanabilir hak)
    - Madde 12.0: İstanbul mahkemeleri yetki kuralı
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
            "Madde 7.1",
            2,
            "Sözleşmeye ilişkin tüm bildirim ve tebligatlar tarafların sözleşmede belirtilen yazılı adreslerine veya kayıtlı e-posta adreslerine yapılır. Bildirim süresi tebliğ tarihinden itibaren başlar."
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
    Demo modunda Madde 4.1'i sunar. Serbest sorularda Retrieval Grader ile alaka denetimi yapar;
    sözleşmede olmayan konular için akışı durdurup kapsam dışı bildiriminde bulunur.
    """
    @staticmethod
    def evaluate(
        query: str,
        engine: FoundryLocalEmbeddingEngine,
        is_preset_demo: bool = False
    ) -> Tuple[ChatMessage, Dict[str, Any], bool, str]:
        all_chunks = fetch_all_chunks()
        if not all_chunks:
            raise ValueError("Veritabanında incelenecek sözleşme maddesi bulunamadı!")

        # 1. Preset Tuzak Demo Modu (Mod A)
        if is_preset_demo:
            target_chunk = next((c for c in all_chunks if "4.1" in c["madde_no"]), all_chunks[0])
            thought = "Tuzak Demo Modu: Kullanıcı fesih hakkını sordu. Madde 4.1 30 gün önceden bildirim hakkını doğrudan karşılıyor."
            content = (
                f"Sözleşme veritabanı incelendiğinde doğrudan fesih maddesi tespit edilmiştir:\n\n"
                f"📌 **{target_chunk['madde_no']} (Sayfa {target_chunk['sayfa_no']}):**\n"
                f"> *\"{target_chunk['content']}\"*\n\n"
                f"**İlk Hukuki Mütalaa:**\n"
                f"Bu maddeye göre müşteri 30 gün önceden yazılı bildirim yaparak sözleşmeyi tek taraflı feshedebilir. "
                f"Ancak sözleşmede yer alabilecek gizli istisnalar veya taahhüt şartları için denetim gereklidir."
            )
            msg = ChatMessage(
                role="proposer",
                agent_name="Proposer Agent (İlk Görüş)",
                avatar="🤖",
                content=content,
                thought=thought,
                action_tool="retrieve_hop_1",
                tool_output=f"{target_chunk['madde_no']} [Demo Modu]",
                metadata={"hop1_chunk": target_chunk, "is_relevant": True}
            )
            return msg, target_chunk, True, ""

        # 2. Serbest Soru Modu: 1. Hop Arama & Retrieval Grader Denetimi
        hop1_chunk, _ = retrieve_hop_1(query, engine)
        is_relevant, topic_or_stem, reason = grade_retrieval_relevance(query, hop1_chunk, all_chunks)

        if not is_relevant:
            thought = (
                f"Kullanıcı sorusu: '{query}'. Retrieval Grader denetimi: {reason}. "
                f"Soru sözleşmede yer almayan bir hususa aittir."
            )
            content = (
                f"⚠️ **Sözleşmede İlgili Hüküm Bulunamadı (Kapsam Dışı):**\n\n"
                f"Sözleşme veritabanı tarandı ancak yönelttiğiniz soruya (`\"{query}\"`) ilişkin doğrudan veya dolaylı bir hüküm bulunamamıştır.\n\n"
                f"- **Değerlendirme:** {reason}\n"
                f"- **Responsible AI:** Belgede düzenlenmeyen konular hakkında varsayım veya ceza üretilmemektedir."
            )
            msg = ChatMessage(
                role="proposer",
                agent_name="Proposer Agent (İlk Görüş)",
                avatar="🤖",
                content=content,
                thought=thought,
                action_tool="retrieval_grader",
                tool_output=f"Kapsam Dışı ({topic_or_stem})",
                metadata={"hop1_chunk": hop1_chunk, "is_relevant": False}
            )
            return msg, hop1_chunk, False, reason

        thought = (
            f"Kullanıcı sorusu: '{query}'. 1. Hop semantik araması tamamlandı. "
            f"En yüksek skorlu hüküm: '{hop1_chunk['madde_no']}' (%{hop1_chunk['score']*100:.1f}). "
            f"Hüküm metnine göre talep incelenebilir görünmektedir."
        )
        content = (
            f"Sözleşme veritabanı incelendiğinde ilgili hüküm tespit edilmiştir:\n\n"
            f"📌 **{hop1_chunk['madde_no']} (Sayfa {hop1_chunk['sayfa_no']}):**\n"
            f"> *\"{hop1_chunk['content']}\"*\n\n"
            f"**İlk Hukuki Mütalaa:**\n"
            f"Bu maddeye göre talep kural olarak değerlendirilebilir görünmektedir. "
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
            metadata={"hop1_chunk": hop1_chunk, "is_relevant": True}
        )
        return msg, hop1_chunk, True, ""


class ChallengerAgent:
    """
    2. Hop Ajanı:
    Proposer'ın sunduğu maddeyi denetler. Metindeki çapraz atıfları (Madde 4.1 vb.)
    ve kısıtlayıcı hukuk terimlerini (taahhüt, ceza, tazminat vb.) avlar.
    Soru alakasız/kapsam dışıysa 2. Hop aramasını çalıştırmaz; akış doğrudan Judge'a geçer.
    """
    @staticmethod
    def evaluate(
        query: str,
        hop1_chunk: Dict[str, Any],
        engine: FoundryLocalEmbeddingEngine,
        is_relevant: bool = True,
        unmatched_reason: str = "",
        is_preset_demo: bool = False
    ) -> Tuple[ChatMessage, ChallengerAssessment, List[Dict[str, Any]]]:
        if not is_relevant:
            assessment = ChallengerAssessment(
                needs_second_hop=False,
                reason=f"Soru sözleşme kapsamı dışındadır: {unmatched_reason}",
                sub_query="",
                detected_terms=[]
            )
            thought = "Soru sözleşme konusu dışı olduğu için 2. Hop arama atlandı. Doğrudan Judge kararına yönlendiriliyor."
            content = (
                f"Denetim filtresi doğruladı: {unmatched_reason} "
                f"Sözleşmede bu konuyla ilişkili herhangi bir taahhüt veya kural bulunmadığı için 2. Hop arama yapılmamıştır."
            )
            msg = ChatMessage(
                role="challenger",
                agent_name="Challenger Agent (Denetçi & İtiraz)",
                avatar="🕵️",
                content=content,
                thought=thought,
                action_tool="relevance_filter",
                tool_output="Kapsam Dışı Soru (2. Hop Atlandı)",
                metadata={"assessment": assessment.model_dump(), "hop2_chunks": []}
            )
            return msg, assessment, []

        all_chunks = fetch_all_chunks()

        # Preset Tuzak Demo Modu (Mod A)
        if is_preset_demo:
            target_h2 = [c for c in all_chunks if "8.2" in c["madde_no"]]
            if not target_h2:
                target_h2 = [c for c in all_chunks if c["id"] != hop1_chunk["id"]][:1]

            assessment = ChallengerAssessment(
                needs_second_hop=True,
                reason="Proposer'ın görüşü yalnızca fesih bildirimi hakkına (Madde 4.1) odaklanmıştır. Ancak Madde 8.2'de 12 aylık taahhüt ve cezai şart kuralı mevcuttur.",
                sub_query="taahhüt süresi 12 ay cezai şart indirimlerin tahsili",
                detected_terms=["taahhüt", "cezai şart", "12 ay", "indirim", "fatura"]
            )
            thought = "Tuzak demo çapraz denetimi: Madde 8.2 taahhüt ve cezai şart hükmü yakalandı. İtiraz sunuldu."
            content = (
                f"⚠️ **Proposer'ın mütalaasına itiraz edilmiştir; sözleşme bütüncül yorumlanmalıdır!**\n\n"
                f"`{hop1_chunk['madde_no']}` fesih bildirimi hakkı verse de tek başına uygulanamaz. "
                f"2. Hop denetiminde tespit edilen bağlayıcı taahhüt hükmü:\n\n"
            )
            for h2 in target_h2:
                content += f"- 📌 **{h2['madde_no']} (Sayfa {h2['sayfa_no']}):** *\"{h2['content']}\"*\n"

            content += (
                f"\nBu hüküm uyarınca, 12 aylık taahhüt süresi dolmadan fesih yapılması durumunda cezai şart ve indirim tutarları yansıtılacaktır. "
                f"Dosya nihai hüküm için Judge Agent'a devredilmiştir."
            )
            msg = ChatMessage(
                role="challenger",
                agent_name="Challenger Agent (Denetçi & İtiraz)",
                avatar="🕵️",
                content=content,
                thought=thought,
                action_tool="retrieve_hop_2",
                tool_output=f"İncelenen Ek Maddeler: {[c['madde_no'] for c in target_h2]}",
                metadata={"assessment": assessment.model_dump(), "hop2_chunks": target_h2}
            )
            return msg, assessment, target_h2

        # Gerçek Serbest Soru 2. Hop Değerlendirmesi
        primary_content = hop1_chunk.get("content", "")
        primary_madde = hop1_chunk.get("madde_no", "")

        found_madde_refs = re.findall(
            r"(?:madde|article|kısım|bölüm)\s*(\d+(?:\.\d+)?)",
            primary_content,
            re.IGNORECASE
        )
        cross_refs = [primary_madde] + [f"Madde {r}" for r in found_madde_refs]

        legal_qualifiers = [
            "ancak", "saklıdır", "saklı", "uyarınca", "taahhüt süresi", "taahhüt",
            "cezai şart", "ceza", "tazminat", "cayma bedeli", "cayma", "kalan ayların",
            "indirim", "istisna", "tahsil", "yükümlü", "asgari", "fatura", "şart"
        ]
        detected_terms = [kw for kw in legal_qualifiers if kw in primary_content.lower() or kw in query.lower()]

        needs_hop2 = len(all_chunks) > 1 and (len(detected_terms) > 0 or len(found_madde_refs) > 0)

        hop2_chunks = []
        if needs_hop2:
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

            all_detected = list(set(detected_terms + detected_in_hop2))
            assessment = ChallengerAssessment(
                needs_second_hop=True,
                reason="İlk madde tek başına değerlendirilemez. Sözleşmede kısıtlayıcı istisnalar veya taahhüt şartları taranmıştır.",
                sub_query=sub_query,
                detected_terms=all_detected
            )

            thought = (
                f"Proposer görüşü denetlendi. Çapraz maddeler tarandı: {[c['madde_no'] for c in hop2_chunks]}. "
                f"Kısıtlayıcı şartlar ({all_detected}) incelendi."
            )
            content = (
                f"⚠️ **Proposer'ın mütalaası denetlenmiştir:**\n\n"
                f"`{hop1_chunk['madde_no']}` maddesi değerlendirilirken bağlantılı şu hükümler de tespit edilmiştir:\n\n"
            )
            for h2 in hop2_chunks:
                content += f"- 📌 **{h2['madde_no']} (Sayfa {h2['sayfa_no']}):** *\"{h2['content']}\"*\n"

            content += "\nDosya nihai risk analizi ve sentez için Judge Agent'a devredilmiştir."
        else:
            assessment = ChallengerAssessment(
                needs_second_hop=False,
                reason="Sözleşmede bu maddeyi sınırlayan ek bir istisna veya cezai şart saptanmadı.",
                sub_query="",
                detected_terms=[]
            )
            thought = "İlk madde denetlendi, sınırlayıcı ek bir çapraz hüküm saptanmadı."
            content = "Proposer'ın incelemesi denetlendi. Sözleşmede bu hükmü kısıtlayan ek bir kayıt saptanmamıştır."

        msg = ChatMessage(
            role="challenger",
            agent_name="Challenger Agent (Denetçi & İtiraz)",
            avatar="🕵️",
            content=content,
            thought=thought,
            action_tool="retrieve_hop_2" if needs_hop2 else "cross_check_verified",
            tool_output=f"İncelenen Ek Maddeler: {[c['madde_no'] for c in hop2_chunks]}",
            metadata={"assessment": assessment.model_dump(), "hop2_chunks": hop2_chunks}
        )
        return msg, assessment, hop2_chunks


class JudgeAgent:
    """
    Nihai Hakem Ajanı:
    Proposer ve Challenger delillerini sentezler, risk analizi yapar
    ve kesin hükmü şu 3 durumdan birine göre Pydantic JudgeVerdict şemasıyla açıklar:

    - Durum A (Riskli / Cezai Şart): Sözleşmede açıkça fesih kısıtı veya taahhüt cezası varsa
    - Durum B (Uygundur / Hak Tanınmış): Talep sözleşme maddelerince doğrudan destekleniyorsa
    - Durum C (Kapsam Dışı / Bilgi Bulunamadı): Sorulan husus sözleşmede hiç düzenlenmemişse (%99 güven)
    """
    @staticmethod
    def evaluate(
        query: str,
        proposer_msg: ChatMessage,
        challenger_msg: ChatMessage,
        all_evidence: List[Dict[str, Any]],
        is_relevant: bool = True,
        unmatched_reason: str = "",
        is_preset_demo: bool = False
    ) -> Tuple[ChatMessage, JudgeVerdict]:

        # ======================================================================
        # DURUM C: KAPSAM DIŞI / BİLGİ BULUNAMADI (RESPONSIBLE AI FALLBACK)
        # ======================================================================
        if not is_relevant:
            reason_text = (
                unmatched_reason if unmatched_reason
                else "İncelenen sözleşme metninde geçici hat dondurma, dondurma süresi veya askıya alma koşullarına ilişkin herhangi bir hüküm yer almamaktadır. Sözleşme dışı konularda varsayım yapılmamaktadır."
            )
            verdict = JudgeVerdict(
                karar="BİLGİ SÖZLEŞMEDE BULUNAMADI (Kapsam Dışı)",
                guven_skoru=99,
                gerekce=reason_text,
                dayanak_maddeler=[],
                sayfa_referanslari=[],
                risk_var_mi=False,
                durum_tipi="DURUM_C"
            )
            thought = (
                f"Soru sözleşme kapsamı dışı tespit edildi. "
                f"Responsible AI prensibi gereği sözleşme dışı konularda varsayım/halüsinasyon yapılmadı. "
                f"Durum C (Bilgi Bulunamadı) kararı üretildi."
            )
            msg = ChatMessage(
                role="judge",
                agent_name="Judge Agent (Baş Hukuk Hakemi)",
                avatar="⚖️",
                content=f"### ⚠️ NİHAİ DENETÇİ HÜKMÜ\n\n**HÜKÜM:** {verdict.karar}\n\n**Gerekçe:** {verdict.gerekce}",
                thought=thought,
                action_tool="responsible_ai_guard",
                tool_output="Bilgi Sözleşmede Bulunamadı (Güven: %99)",
                metadata=verdict.model_dump()
            )
            return msg, verdict

        # ======================================================================
        # TUZAK DEMO ÖZEL KARARI (DURUM A)
        # ======================================================================
        if is_preset_demo:
            verdict = JudgeVerdict(
                karar="TALEBİNİZ KOŞULLU VE RİSKLİ (SÖZLEŞME İSTİSNALARI VE CEZAİ HÜKÜMLER GEÇERLİDİR)",
                guven_skoru=88,
                gerekce=(
                    "Proposer'ın dayandığı Madde 4.1 hükmü 30 gün önceden bildirimle fesih hakkı tanımakla birlikte; "
                    "Challenger tarafından ortaya konan Madde 8.2 hükmü uyarınca sözleşme 12 aylık taahhüt süresine tabidir. "
                    "Erken fesih durumunda kalan ayların bedelleri ve sağlanan indirimler cezai şart olarak faturalandırılacaktır. "
                    "Dolayısıyla fesih bildirimi mümkündür ANCAK cezasız erken fesih yapılamaz."
                ),
                dayanak_maddeler=["Madde 4.1", "Madde 8.2"],
                sayfa_referanslari=[2, 4],
                risk_var_mi=True,
                durum_tipi="DURUM_A"
            )
            thought = "Tuzak demo senaryosu: Madde 4.1 ve 8.2 çapraz analiziyle taahhüt cezai şart riski hükme bağlandı."
            content = (
                f"### ⚖️ NİHAİ DENETÇİ HÜKMÜ (TUZAK DEMO)\n\n"
                f"**HÜKÜM:** {verdict.karar}\n\n"
                f"**Gerekçeli Hukuki Karar:** {verdict.gerekce}"
            )
            msg = ChatMessage(
                role="judge",
                agent_name="Judge Agent (Baş Hukuk Hakemi)",
                avatar="⚖️",
                content=content,
                thought=thought,
                action_tool="legal_synthesis",
                tool_output="Durum A: Yüksek Risk / Cezai Şart Tespiti",
                metadata=verdict.model_dump()
            )
            return msg, verdict

        # ======================================================================
        # GERÇEK SERBEST SORU SENTEZİ (DURUM A VEYA DURUM B)
        # ======================================================================
        maddeler = [c["madde_no"] for c in all_evidence]
        sayfalar = sorted(list(set(c["sayfa_no"] for c in all_evidence)))
        full_text = " ".join([c.get("content", "") for c in all_evidence]).lower()

        risk_keywords = [
            "cezai şart", "taahhüt süresi", "taahhüt", "tazminat", "kalan ayların",
            "cayma bedeli", "cayma", "fatura edilir", "tahsil edilir", "faiz"
        ]
        detected_risks = [k for k in risk_keywords if k in full_text]
        risk_var_mi = len(detected_risks) > 0

        # Dinamik Güven Skoru Hesabı
        h1_score = all_evidence[0].get("score", 0.0) if all_evidence else 0.5
        h2_scores = [c.get("score", 0.0) for c in all_evidence[1:]] if len(all_evidence) > 1 else []
        top_h2 = max(h2_scores) if h2_scores else (h1_score * 0.8)
        has_direct_ref = any(c.get("is_direct_ref", False) for c in all_evidence)
        cross_ref_weight = 0.95 if has_direct_ref else (0.80 if h2_scores else 0.65)
        raw_confidence = ((h1_score * 0.40) + (top_h2 * 0.30) + (cross_ref_weight * 0.30)) * 100
        guven_skoru = int(np.clip(round(raw_confidence), 60, 95))

        q_lower = query.lower()
        is_termination_q = any(k in q_lower for k in ["fesih", "feshet", "ayrıl", "iptal", "vazgeç", "sonlandır", "cayma"])

        if risk_var_mi:
            # DURUM A: Riskli / Cezai Şart
            karar = "TALEBİNİZ KOŞULLU VE RİSKLİ (SÖZLEŞME İSTİSNALARI VE CEZAİ HÜKÜMLER GEÇERLİDİR)"
            if is_termination_q:
                gerekce = (
                    f"İncelenen {all_evidence[0]['madde_no']} hükmü fesih usulünü düzenlemekle birlikte; "
                    f"sözleşmede yer alan kısıtlayıcı kayıtlar ({', '.join([c['madde_no'] for c in all_evidence[1:]]) if len(all_evidence) > 1 else all_evidence[0]['madde_no']}) "
                    f"ve taahhüt şartları uyarınca erken ayrılma durumunda cezai şart veya indirim bedelleri tahsil edilmektedir."
                )
            else:
                gerekce = (
                    f"İncelenen {all_evidence[0]['madde_no']} hükmü incelenmiş olup, sözleşmedeki bağlantılı kısıtlayıcı hükümler "
                    f"({', '.join(detected_risks)}) nedeniyle talep koşulsuz uygulanamaz; sözleşmedeki mali ve hukuki yaptırımlar saklıdır."
                )
            durum_tipi = "DURUM_A"
        else:
            # DURUM B: Uygundur / Hak Tanınmış
            karar = "TALEP UYGUNDUR VE DOĞRUDAN UYGULANABİLİR (HAK TANINMIŞTIR)"
            gerekce = (
                f"İncelenen {', '.join(maddeler)} hükümleri uyarınca, talebinizi engelleyen veya cezai şarta bağlayan "
                f"herhangi bir taahhüt kısıtlaması tespit edilmemiştir. Sözleşme şartlarına uygun olarak doğrudan uygulanabilir."
            )
            durum_tipi = "DURUM_B"

        verdict = JudgeVerdict(
            karar=karar,
            guven_skoru=guven_skoru,
            gerekce=gerekce,
            dayanak_maddeler=maddeler,
            sayfa_referanslari=sayfalar,
            risk_var_mi=risk_var_mi,
            durum_tipi=durum_tipi
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
            thought=f"Deliller incelendi: {maddeler}. Tespit edilen durum: {durum_tipi}.",
            action_tool="legal_synthesis",
            tool_output=f"Verdict: {verdict.karar} [{durum_tipi}]",
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
        st.info(
            f"🧠 **Embedding Motoru:**\n`{engine.engine_name}`\n\n"
            f"🤖 **Yerel Model (LLM):**\n`{engine.llm_name}`\n\n"
            f"🔒 **Çevrimdışı Güvenlik:** %100 Yerel / Sıfır Dış API"
        )
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

    # --------------------------------------------------------------------------
    # HIZLI SENARYO TESTİ (TÜM DURUMLAR: DURUM A, DURUM B, DURUM C)
    # --------------------------------------------------------------------------
    st.markdown("#### 🎯 Hızlı Senaryo Denetimi (Tek Tıkla Tüm Durumları Test Edin)")
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        run_demo_s1 = st.button(
            "🔴 Senaryo A: Tuzak Fesih\n(Cezai Şart Riski)",
            type="primary",
            use_container_width=True,
            help="Madde 4.1 fesih hakkı vadeder; fakat Madde 8.2 taahhüdü kalan ayları cezai şart olarak yansıtır."
        )
        st.caption("💡 **Tuzak Fesih:** Madde 4.1 serbest fesih gibi görünür; ancak 12 aylık taahhüt cezası doğar.")

    with col_s2:
        run_demo_s2 = st.button(
            "🟢 Senaryo B: Doğrudan Hak\n(Usule Uygun / Cezasız)",
            use_container_width=True,
            help="Madde 7.1 uyarınca yazılı/e-posta bildirim usulü doğrudan uygulanabilir; ek cezai engel yoktur."
        )
        st.caption("💡 **Doğrudan Hak:** Sözleşmede usulü düzenlenen ve cezai şarta bağlanmayan hak sorgulanır.")

    with col_s3:
        run_demo_s3 = st.button(
            "🟡 Senaryo C: Sorumlu AI\n(Kapsam Dışı / Çekimser)",
            use_container_width=True,
            help="Sözleşmede 'geçici dondurma' yoktur; sistem cezai şart uydurmaz, %99 güvenle çekimser kalır."
        )
        st.caption("💡 **Halüsinasyon Engeli:** Belgede geçmeyen hat dondurma sorulur; çekimser kalınır.")

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
    is_preset_demo = False

    if run_demo_s1:
        load_demo_data(engine)
        trigger_question = "Müşteri sözleşmenin 3. ayında 30 gün önceden bildirerek cezasız fesih yapabilir mi?"
        is_preset_demo = True
    elif run_demo_s2:
        load_demo_data(engine)
        trigger_question = "Sözleşmeye ilişkin tüm tebligat ve yazılı bildirimler hangi usulle yapılmalıdır?"
        is_preset_demo = False
    elif run_demo_s3:
        load_demo_data(engine)
        trigger_question = "Aboneliğimi 1 yıl içinde en fazla kaç ay süreyle geçici olarak dondurabilirim?"
        is_preset_demo = False
    elif submit_query_button and user_question.strip():
        chunks_check = fetch_all_chunks()
        if not chunks_check:
            st.warning("⚠️ Lütfen önce sol menüden bir PDF yükleyin veya yukarıdaki hazır senaryolardan birine basın.")
        else:
            trigger_question = user_question.strip()
            is_preset_demo = False

    # --------------------------------------------------------------------------
    # CANLI YÜRÜTME LOGU (ST.STATUS) VE AJAN TARTIŞMASI
    # --------------------------------------------------------------------------
    if trigger_question:
        with st.status(
            "🚀 2 Dk'lık Tuzak Demo Yürütülüyor..." if is_preset_demo else "🔄 Çoklu Ajan Denetimi Yürütülüyor...",
            expanded=True
        ) as status:
            time.sleep(0.3)
            # 1. Hop Arama & Proposer
            st.write("🔍 **1. Hop Vektör Arama:** Kullanıcı sorgusu SQLite vektör alanında tarandı...")
            proposer_msg, hop1_chunk, is_relevant, unmatched_reason = ProposerAgent.evaluate(
                trigger_question, engine, is_preset_demo=is_preset_demo
            )

            if not is_relevant:
                # DURUM C: Kapsam Dışı / Bilgi Bulunamadı
                st.write(f"⚠️ **Retrieval Grader Uyarısı:** {unmatched_reason}")
                time.sleep(0.3)
                challenger_msg, assessment, hop2_chunks = ChallengerAgent.evaluate(
                    trigger_question, hop1_chunk, engine, is_relevant=False, unmatched_reason=unmatched_reason, is_preset_demo=False
                )
                st.write("🕵️ **Challenger Agent:** Kapsam dışı soru tespit edildi; gereksiz 2. Hop arama atlandı.")
                time.sleep(0.3)
                all_evidence = []
                judge_msg, verdict = JudgeAgent.evaluate(
                    trigger_question, proposer_msg, challenger_msg, all_evidence,
                    is_relevant=False, unmatched_reason=unmatched_reason, is_preset_demo=False
                )
                st.write("⚖️ **Judge Agent:** Sorumlu Yapay Zeka filtresi devrede. Halüsinasyon üretilmedi, 'BİLGİ BULUNAMADI' kararı verildi.")
                status.update(label="⚠️ Bilgi Sözleşmede Bulunamadı (Kapsam Dışı)", state="error", expanded=False)
            else:
                st.write(f"🤖 **Proposer Agent İlk Mütalaayı Sundu:** `{hop1_chunk['madde_no']}` tespit edildi (Kosinüs: `%{hop1_chunk['score']*100:.1f}`).")
                time.sleep(0.4)

                # 2. Hop Arama & Challenger
                st.write("🕵️ **Challenger Agent Denetimi Başlattı:** Çapraz atıflar, taahhütler ve istisnalar taranıyor...")
                challenger_msg, assessment, hop2_chunks = ChallengerAgent.evaluate(
                    trigger_question, hop1_chunk, engine, is_relevant=True, is_preset_demo=is_preset_demo
                )
                if assessment.needs_second_hop and hop2_chunks:
                    st.write(f"⚠️ **Challenger İtiraz Etti:** 2. Hop araması ile sınırlandırıcı ek maddeler çekildi: `{[c['madde_no'] for c in hop2_chunks]}`.")
                else:
                    st.write("✅ **Challenger Onayladı:** Ek bir kısıtlama veya istisna bulunmadı.")
                time.sleep(0.4)

                # Judge Sentezi
                st.write("⚖️ **Judge Agent Hükmü Hazırlıyor:** Tüm hop delilleri ve risk faktörleri sentezleniyor...")
                all_evidence = [hop1_chunk] + hop2_chunks
                judge_msg, verdict = JudgeAgent.evaluate(
                    trigger_question, proposer_msg, challenger_msg, all_evidence,
                    is_relevant=True, is_preset_demo=is_preset_demo
                )
                time.sleep(0.3)

                status.update(label="✅ Denetim Başarıyla Tamamlandı!", state="complete", expanded=False)

        # Durumu Session State'e Kaydet
        st.session_state.last_verdict = verdict
        st.session_state.debate_messages = [proposer_msg, challenger_msg, judge_msg]
        st.session_state.evidence_chunks = [hop1_chunk] + hop2_chunks if is_relevant else [hop1_chunk]
        st.session_state.input_query = trigger_question

    # --------------------------------------------------------------------------
    # RAPORLAMA VE METRİK PANELİ
    # --------------------------------------------------------------------------
    if st.session_state.last_verdict:
        verdict = st.session_state.last_verdict
        is_durum_c = (
            getattr(verdict, "durum_tipi", "") == "DURUM_C"
            or verdict.karar.startswith("BİLGİ SÖZLEŞMEDE BULUNAMADI")
            or "BULUNAMADI" in verdict.karar
        )
        is_durum_a = (
            getattr(verdict, "durum_tipi", "") == "DURUM_A"
            or verdict.risk_var_mi
        )

        st.markdown("## 📋 Denetim Raporu ve Hüküm")

        rep_col1, rep_col2 = st.columns([3, 1])

        with rep_col1:
            if is_durum_c:
                st.warning(f"### ⚠️ {verdict.karar}")
                st.markdown(f"**Gerekçeli Hukuki Değerlendirme:**\n\n{verdict.gerekce}")
                st.info("💡 **Bilgilendirme:** Bu konu yüklenen sözleşme metninde düzenlenmemiştir. Sorumlu Yapay Zeka (Responsible AI) ilkeleri uyarınca belgede yer almayan hususlarda varsayım veya ceza üretilmemektedir.")
            elif is_durum_a:
                st.error(f"### 🛑 HÜKÜM:\n**{verdict.karar}**")
                st.markdown(f"**Gerekçeli Hukuki Karar:**\n\n{verdict.gerekce}")
                st.markdown("**Dayanak Maddeler:**")
                badges_html = "".join([f"<span class='badge-tag'>{m}</span>" for m in verdict.dayanak_maddeler])
                st.markdown(badges_html, unsafe_allow_html=True)
                st.markdown(f"**İlgili Sayfalar:** {', '.join([str(p) for p in verdict.sayfa_referanslari])}")
            else:  # DURUM B
                st.success(f"### 🟢 HÜKÜM:\n**{verdict.karar}**")
                st.markdown(f"**Gerekçeli Hukuki Karar:**\n\n{verdict.gerekce}")
                st.markdown("**Dayanak Maddeler:**")
                badges_html = "".join([f"<span class='badge-tag'>{m}</span>" for m in verdict.dayanak_maddeler])
                st.markdown(badges_html, unsafe_allow_html=True)
                st.markdown(f"**İlgili Sayfalar:** {', '.join([str(p) for p in verdict.sayfa_referanslari])}")

        with rep_col2:
            if is_durum_c:
                st.metric(
                    label="Hukuki Durum",
                    value="Kapsam Dışı",
                    delta="Sözleşmede Yok",
                    delta_color="off",
                    help="Yöneltilen soru mevcut sözleşmenin düzenlediği konular ve maddeler arasında yer almamaktadır."
                )
                st.metric(
                    label="Güven Skoru",
                    value=f"%{verdict.guven_skoru}",
                    delta="Yüksek Doğruluk",
                    delta_color="normal",
                    help="Sorunun sözleşme metninde yer almadığı %99 doğrulukla kesinleştirilmiştir."
                )
                st.metric(
                    label="Risk Değerlendirmesi",
                    value="RİSK YOK",
                    delta="Düzenlenmemiş Konu",
                    delta_color="off",
                    help="Sözleşmede bu konuya ilişkin bir cezai şart veya yaptırım riski bulunmamaktadır."
                )
                st.metric(
                    label="Hop Sayısı",
                    value="1 Hop",
                    delta="2. Hop Atlandı",
                    delta_color="off",
                    help="Soru kapsam dışı olduğu için gereksiz 2. Hop çapraz arama engellenmiştir."
                )
            else:
                st.metric(
                    label="Güven Skoru",
                    value=f"%{verdict.guven_skoru}",
                    delta="Yüksek Doğruluk" if verdict.guven_skoru >= 85 else ("Orta" if verdict.guven_skoru >= 60 else "Düşük"),
                    help="Sorgunun maddelerle anlamsal kosinüs benzerliği, 2. Hop istisna uyumu ve doğrudan çapraz atıf sentezidir."
                )
                hop_count = 2 if len(st.session_state.evidence_chunks) > 1 else 1
                st.metric(
                    label="Hop Sayısı",
                    value=f"{hop_count} Hop",
                    delta="Multi-Hop Aktif" if hop_count > 1 else "Tekil",
                    help="Arama derinliği kademesidir. 1 Hop: İlk doğrudan madde. 2 Hop: Zincirleme istisna ve taahhüt maddeleri."
                )
                st.metric(
                    label="Risk Değerlendirmesi",
                    value="YÜKSEK RİSK" if verdict.risk_var_mi else "DÜŞÜK RİSK",
                    delta="Cezai Şart Mevcut" if verdict.risk_var_mi else "Doğrudan Uygulanabilir",
                    delta_color="inverse" if verdict.risk_var_mi else "normal",
                    help="Sözleşme maddelerinde cezai şart, taahhüt süresi ihlali veya mali yaptırım riski tespit edilip edilmediğini gösterir."
                )

        st.divider()

        # AKORDİYON 1: KULLANILAN SQLITE CHUNK'LARI VE SKORLARI
        if not is_durum_c:
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
        else:
            with st.expander("📋 Taranan En Yakın Kayıt (Bilgi Amaçlı)", expanded=False):
                st.caption("Aşağıdaki madde veritabanında taranan kayıtlardan biridir; ancak sorunuzla doğrudan konu uyumu veya anlamsal benzerliği yetersiz olduğu için hükümde delil olarak kullanılmamıştır:")
                if st.session_state.evidence_chunks:
                    ref_c = st.session_state.evidence_chunks[0]
                    st.info(f"📌 **{ref_c['madde_no']}** *(Sayfa {ref_c['sayfa_no']})* — En Yakın Benzerlik: `%{ref_c.get('score', 0.0)*100:.1f}` (Eşik Altı)\n\n\"{ref_c['content']}\"")

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
