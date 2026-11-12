"""분자 구조 유사도 탐색기 - DOS 텍스트 화면 버전.

사용법:
  python -m streamlit run app.py

분자 구조 그림만 이미지로 출력하고, 나머지는 텍스트로 출력합니다.
기준 분자와 검색 1위 분자의 구조·용도·작용기를 나란히 비교합니다.
주의: 구조 유사성 검색이지, 약효/독성 예측 AI가 아닙니다.
"""

import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors


# 학습·시연용 구조식 목록: 외부 인터넷 접속 불필요
MOLECULES = {
    "아스피린": "CC(=O)Oc1ccccc1C(=O)O",
    "살리실산": "O=C(O)c1ccccc1O",
    "메틸 살리실레이트": "COC(=O)c1ccccc1O",
    "벤조산": "O=C(O)c1ccccc1",
    "아세트아미노펜": "CC(=O)Nc1ccc(O)cc1",
    "이부프로펜": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "나프록센": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "카페인": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "페놀": "Oc1ccccc1",
    "에탄올": "CCO",
    "다이메틸 에터": "COC",
    "아세트산": "CC(=O)O",
    "에틸 아세테이트": "CCOC(=O)C",
}


# 이미 알려진 사실만 출력하도록 수동으로 정리한 교육용 설명.
# 절대 자카드 유사도로 약효를 추정하거나 생성하지 않습니다.
# 물질명이 아니라 표준화된 SMILES를 키로 사용하므로 직접 입력해도
# 예제에 들어 있는 물질과 동일하다면 같은 설명을 보여줄 수 있습니다.
COMPOUND_INFO = {
    "아스피린": {
        "kind": "약물",
        "feature": "통증 완화, 해열, 항염 작용이 알려져 있음. 일부 의료 상황에서는 혈소판 응집 억제에 사용됨.",
        "source": "MedlinePlus - Aspirin",
    },
    "살리실산": {
        "kind": "피부 외용 성분 / 화학물질",
        "feature": "각질을 부드럽게 하고 제거하는 성질을 이용하여 여드름·사마귀 관련 피부 외용제에 사용됨.",
        "source": "PubChem - Salicylic Acid",
    },
    "메틸 살리실레이트": {
        "kind": "외용제 성분 / 향료 성분",
        "feature": "일부 근육·관절 통증 완화용 바르는 제제에 쓰임. 윈터그린 향이 특징이며 과량 노출은 위험할 수 있음.",
        "source": "PubChem - Methyl Salicylate",
    },
    "벤조산": {
        "kind": "보존제 / 화학원료",
        "feature": "산성 조건에서 미생물 생장을 억제하는 특성을 이용해 보존제 및 화학 합성 원료로 사용됨.",
        "source": "PubChem - Benzoic Acid",
    },
    "아세트아미노펜": {
        "kind": "약물",
        "feature": "통증 완화와 해열에 사용됨. 일반적인 비스테로이드성 소염진통제(NSAID)로 분류하지 않음.",
        "source": "MedlinePlus - Acetaminophen",
    },
    "이부프로펜": {
        "kind": "약물",
        "feature": "비스테로이드성 소염진통제(NSAID)로 통증·발열·염증을 줄이는 데 사용됨.",
        "source": "MedlinePlus - Ibuprofen",
    },
    "나프록센": {
        "kind": "약물",
        "feature": "비스테로이드성 소염진통제(NSAID)로 통증·발열·염증 완화에 사용됨.",
        "source": "MedlinePlus - Naproxen",
    },
    "카페인": {
        "kind": "생리활성 물질",
        "feature": "중추신경계를 자극하여 각성도를 높일 수 있으며, 커피·차 등에 들어 있는 물질임.",
        "source": "PubChem - Caffeine",
    },
    "페놀": {
        "kind": "산업용 화학물질",
        "feature": "페놀수지 등 화학제품의 원료. 부식성과 유해성이 있으므로 취급에 주의해야 함.",
        "source": "PubChem - Phenol",
    },
    "에탄올": {
        "kind": "알코올 / 용매",
        "feature": "물과 잘 섞이는 휘발성 알코올. 용매 및 일부 소독제품의 성분으로 사용되며 인화성이 있음.",
        "source": "PubChem - Ethanol",
    },
    "다이메틸 에터": {
        "kind": "에터 / 산업용 화학물질",
        "feature": "상온에서 기체인 가연성 물질로, 에어로졸 분사제나 연료 등으로 사용됨.",
        "source": "PubChem - Dimethyl Ether",
    },
    "아세트산": {
        "kind": "유기산 / 화학원료",
        "feature": "식초의 산미를 내는 대표 성분이며, 다양한 화학제품의 합성 원료로 쓰임.",
        "source": "PubChem - Acetic Acid",
    },
    "에틸 아세테이트": {
        "kind": "에스터 / 유기용매",
        "feature": "과일향이 나는 휘발성 액체로, 추출·세척 등에서 유기용매로 사용됨.",
        "source": "PubChem - Ethyl Acetate",
    },
}


# SMARTS는 부분 구조를 찾기 위한 RDKit 패턴 언어.
# 카복실기/에스터기/아마이드기처럼 중첩되는 C=O는 중복 표시하지 않음.
FUNCTIONAL_GROUPS = [
    ("카복실기(-COOH)", "[CX3](=O)[OX2H1]"),
    ("에스터기(-COO-)", "[CX3](=O)[OX2H0][#6]"),
    ("아마이드기(-CONH- 등)", "[NX3][CX3](=[OX1])"),
    ("페놀성 하이드록실기(Ar-OH)", "[OX2H][c]"),
    ("알코올성 하이드록실기(-OH)", "[CX4][OX2H]"),
    ("에터기(-O-)", "[OD2]([#6;!$(C=O)])[#6;!$(C=O)]"),
]
FUNCTIONAL_SMARTS = [(name, Chem.MolFromSmarts(query)) for name, query in FUNCTIONAL_GROUPS]
CARBONYL = Chem.MolFromSmarts("[#6]=[OX1]")
BENZENE = Chem.MolFromSmarts("c1ccccc1")
AROMATIC_N = Chem.MolFromSmarts("[n;r]")


def describe_functional_groups(molecule):
    """입력된 화학구조에서 주요 작용기와 고리 특징을 찾아 문자열로 반환."""
    found = []
    for name, pattern in FUNCTIONAL_SMARTS:
        count = len(molecule.GetSubstructMatches(pattern))
        if count:
            found.append(f"{name} {count}개")

    # 분류되지 않은 카보닐(C=O)이 남은 경우(예: 카페인) 표시.
    total_carbonyl = len(molecule.GetSubstructMatches(CARBONYL))
    classified_carbonyl = sum(
        len(molecule.GetSubstructMatches(pattern))
        for name, pattern in FUNCTIONAL_SMARTS
        if name.startswith(("카복실기", "에스터기", "아마이드기"))
    )
    remaining_carbonyl = total_carbonyl - classified_carbonyl
    if remaining_carbonyl > 0:
        found.append(f"그 밖의 카보닐기(C=O) {remaining_carbonyl}개")

    structures = []
    if molecule.HasSubstructMatch(BENZENE):
        structures.append("벤젠 고리(방향족)")
    if molecule.HasSubstructMatch(AROMATIC_N):
        structures.append("질소 함유 방향족 헤테로고리")
    return (", ".join(found) if found else "등록된 대표 작용기 없음",
            ", ".join(structures) if structures else "별도 표시할 고리 특징 없음")


def known_information(molecule):
    canonical = Chem.MolToSmiles(molecule)
    for name, smiles in MOLECULES.items():
        if Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == canonical:
            return COMPOUND_INFO.get(name)
    return None


@st.cache_resource
def morgan_generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def make_fingerprint(molecule, method):
    if method == "MACCS Keys (167비트)":
        return MACCSkeys.GenMACCSKeys(molecule)
    return morgan_generator().GetFingerprint(molecule)


def calculate_jaccard(fp_a, fp_b):
    """이진 지문에서 1인 비트의 집합으로 자카드 유사도를 직접 계산."""
    bits_a = set(fp_a.GetOnBits())
    bits_b = set(fp_b.GetOnBits())
    intersection = len(bits_a & bits_b)
    union = len(bits_a | bits_b)
    score = intersection / union if union else 1.0

    # 계산식과 RDKit 타니모토 계수의 값이 동일한지 검산
    reference = DataStructs.TanimotoSimilarity(fp_a, fp_b)
    if abs(score - reference) > 1e-10:
        raise RuntimeError("자카드 유사도와 타니모토 검산 결과가 다릅니다.")
    return intersection, union, score


st.set_page_config(page_title="MOLECULE SEARCH / DOS", page_icon="💻", layout="centered")

# 텍스트 중심의 DOS 외관. 구조식 2개를 첫 화면에서 나란히 비교할 수 있도록 배치.
st.markdown(
    """
    <style>
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
          background-color: #050805 !important;
      }
      .stApp * {
          font-family: Consolas, "Courier New", "Malgun Gothic", monospace !important;
          color: #a6f59b;
      }
      [data-testid="stMainBlockContainer"] {
          max-width: 950px;
          padding-top: 0.5rem;
          padding-bottom: 1.5rem;
      }
      [data-testid="stVerticalBlock"] { gap: 0.45rem; }
      [data-testid="stHorizontalBlock"] { gap: 1.0rem; }
      [data-testid="stText"], [data-testid="stMarkdownContainer"] p, label {
          white-space: pre-wrap;
          font-size: 0.88rem !important;
          line-height: 1.3 !important;
      }
      [data-testid="stSelectbox"] [data-baseweb="select"] > div,
      [data-testid="stTextInput"] input {
          background: #050805 !important;
          border: 1px solid #588855 !important;
          border-radius: 0 !important;
          box-shadow: none !important;
      }
      [data-testid="stSelectbox"] [data-baseweb="select"] * {
          color: #a6f59b !important;
      }
      /* 흰 배경의 선택 항목은 검정 글씨로 고정. */
      [data-baseweb="popover"], [data-baseweb="menu"],
      [role="listbox"], [role="option"] {
          background: #ffffff !important;
          color: #111111 !important;
      }
      [data-baseweb="popover"] *, [data-baseweb="menu"] *,
      [role="listbox"] *, [role="option"] * {
          color: #111111 !important;
          -webkit-text-fill-color: #111111 !important;
      }
      [role="option"]:hover, [role="option"][aria-selected="true"] {
          background: #e5eee5 !important;
      }
      /* 긴 설명은 가로 스크롤 대신 줄바꿈하여 보여준다. */
      [data-testid="stText"] pre {
          white-space: pre-wrap !important;
          overflow-wrap: anywhere !important;
          line-height: 1.35 !important;
          font-size: 0.84rem !important;
      }
      [data-testid="stImage"] img {
          border: 1px solid #588855;
          border-radius: 0 !important;
      }
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
          display: none !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

LINE = "=" * 56
st.text("C:\\CHEM> MOLECULAR_SIMILARITY.EXE / TOP-1")
st.text(LINE)

# 입력부를 한 줄로 압축. 직접 입력을 고르면 그 아래에 SMILES 입력창만 하나 추가.
col_mode, col_name, col_fp = st.columns([1.05, 1.30, 1.35])
with col_mode:
    input_mode = st.selectbox("입력 방식", ["예제 물질 선택", "SMILES 직접 입력"])
with col_name:
    if input_mode == "예제 물질 선택":
        query_name = st.selectbox("기준 물질", list(MOLECULES.keys()))
        query_smiles = MOLECULES[query_name]
    else:
        query_name = "직접 입력"
        st.text("기준 물질: 사용자 입력")
        query_smiles = ""
with col_fp:
    method = st.selectbox("분자 지문", ["Morgan (2048비트, 반경 2)", "MACCS Keys (167비트)"])
if input_mode == "SMILES 직접 입력":
    query_smiles = st.text_input("SMILES 구조식", value=MOLECULES["아스피린"])

query_mol = Chem.MolFromSmiles(query_smiles.strip()) if query_smiles.strip() else None
if query_mol is None:
    st.text("ERROR: 올바른 SMILES를 입력해야 합니다. 예: CCO")
    st.stop()

query_fp = make_fingerprint(query_mol, method)
query_canonical = Chem.MolToSmiles(query_mol)

# 같은 물질을 그대로 찾는 무의미한 1.0 결과를 제외하고, 항상 최고 유사도 1종만 계산·표시.
best = None
for name, smiles in MOLECULES.items():
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or Chem.MolToSmiles(molecule) == query_canonical:
        continue
    fp = make_fingerprint(molecule, method)
    intersection, union, similarity = calculate_jaccard(query_fp, fp)
    candidate = {
        "name": name,
        "smiles": smiles,
        "mol": molecule,
        "fp": fp,
        "intersection": intersection,
        "union": union,
        "score": similarity,
    }
    if best is None or similarity > best["score"] or (
        similarity == best["score"] and name < best["name"]
    ):
        best = candidate

if best is None:
    st.text("NO MATCH: 비교할 다른 구조가 없습니다.")
    st.stop()

# 기준 분자와 최고 유사도 분자의 그림/설명을 두 칸에 나란히 표시.
# 약효/특징은 준비된 13개 물질에 한해 알려진 사실을 보여주며,
# 직접 입력한 미등록 분자에 대해서는 유사도만으로 약효를 추론하지 않는다.
def show_molecule_info(molecule, name, smiles, label, score=None):
    heading = f"{label} {name}"
    if score is not None:
        heading += f" / J={score:.4f}"
    st.text(heading)
    st.image(Draw.MolToImage(molecule, size=(410, 210)), width=350)
    st.text(f"분자식 : {rdMolDescriptors.CalcMolFormula(molecule)}")
    st.text(f"SMILES : {smiles}")

    information = known_information(molecule)
    if information:
        st.text(f"구분      : {information['kind']}")
        st.text(f"용도/특징 : {information['feature']}")
        st.text(f"정보 출처 : {information['source']} (사전 정리)")
    else:
        st.text("용도/특징 : 등록된 정보 없음 (유사도만으로 약효 예측 불가)")

    groups, skeleton = describe_functional_groups(molecule)
    st.text(f"작용기    : {groups}")
    st.text(f"구조 특징 : {skeleton}")


st.text("[01] STRUCTURE + INFO / 기준 물질 vs 유사도 1위")
left, right = st.columns(2)
with left:
    show_molecule_info(query_mol, query_name, query_smiles, "[A] 기준:")
with right:
    show_molecule_info(best["mol"], best["name"], best["smiles"], "[B] 1위:", score=best["score"])

st.text("[02] JACCARD CALCULATION")
if best["union"]:
    st.text(
        f"J(A,B) = 교집합 / 합집합 = {best['intersection']} / {best['union']}"
        f" = {best['score']:.4f}"
    )
else:
    st.text("1인 비트가 양쪽 모두 없음: J(A,B) = 1.0000 (프로그램 정의)")
st.text(LINE)
st.text("C:\\CHEM> DONE / 예제 13종만 비교 / 구조 유사도 ≠ 약효 유사도")
