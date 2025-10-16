import streamlit as st
import pandas as pd
import re, math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from io import BytesIO

# ==================== Page Config ====================
st.set_page_config(
    page_title="bewild V-R100 Tools",
    page_icon="🧾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
  .main{background:linear-gradient(135deg,#e3f2fd 0%,#fff 100%)}
  [data-testid="stSidebar"]{background:linear-gradient(180deg,#bbdefb 0%,#e3f2fd 100%)}
  .stButton>button{background:linear-gradient(90deg,#42a5f5 0%,#2196f3 100%);color:#fff;border:none;border-radius:8px;
    padding:.5rem 1.5rem;font-weight:500;transition:.3s;box-shadow:0 2px 5px rgba(33,150,243,.3)}
  .stButton>button:hover{transform:translateY(-2px)}
  .stDownloadButton>button{background:linear-gradient(90deg,#26c6da 0%,#00acc1 100%);color:#fff;border:none;border-radius:8px;
    padding:.5rem 1.5rem;font-weight:500;box-shadow:0 2px 5px rgba(0,172,193,.3)}
  .stDownloadButton>button:hover{transform:translateY(-2px)}
  .stForm{background:#fff;padding:1.5rem;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.05);border:1px solid #e3f2fd}
  h1,h2,h3{color:#1565c0}
  .stDataFrame{border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.05)}
</style>
""", unsafe_allow_html=True)

# ==================== Header ====================
st.markdown("<h2 style='text-align:center'>Casio V-R100 Tools</h2>", unsafe_allow_html=True)

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("### ⚙️ การตั้งค่า")
    vr100_enc = st.selectbox("Encoding ไฟล์ SQL", ["UTF-8 (ปกติ)", "UTF-8 with BOM (UTF-8-SIG)"], index=1)
    st.caption("อัตโนมัติ: จับชีท + หัวตาราง + คอลัมน์เอง • ใช้เฉพาะ 'ราคาขาย' + โปรคงที่")
    st.caption("โปร: 3ชิ้น100→50฿, 4ชิ้น100→35฿, 50/2ชิ้น100→80฿ (ต่อชิ้น)")

# ==================== Helpers ====================
def export_bytes(sql_text: str) -> bytes:
    fixed = "\r\n".join(ln.rstrip("\r\n") for ln in sql_text.splitlines())
    enc = "utf-8-sig" if vr100_enc.endswith("SIG") else "utf-8"
    return fixed.encode(enc, errors="ignore")

def export_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig")

def now_casio_fmt() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")

def sql_str(s) -> str:
    return "" if s is None else str(s).replace("'", "''")

def nz_str(v: object) -> str:
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = "" if v is None else str(v).strip()
    return "" if s.lower() == "nan" else s

def to_int0(v, default=0) -> int:
    try:
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default

def to_satang(x) -> int:
    if x is None:
        return 0
    try:
        if (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
            return 0
    except Exception:
        pass
    s = str(x).strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return 0
    d = Decimal(s).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(d * 100)

# ---------- อ่าน Excel อัตโนมัติ (SMART HEADER) ----------
CAND_HEADERS = {
    "รหัสสินค้า","ITEM CODE","ITEMCODE","SAPID","MATERIAL","MATERIAL ID",
    "ชื่อสินค้า","ITEMNAME","NAME ITEM","NAMEITEM","รายการสินค้า","SKU DESCRIPTION",
    "บาร์โค้ด","BARCODE","UNIT BARCODE","SCANCODE1",
    "UNITQTY","QTY","PACK","ชิ้นต่อแพ็ค","รวมชิ้นต่อแพ็ค","หน่วยต่อแพ็ค",
    "ราคาขาย","PRICE","UNIT PRICE","RETAIL PRICE","ราคาต่อหน่วย"
}

def _canon(s: str) -> str:
    return re.sub(r"[\s_\-\+\.\(\)\[\]\{\}/\\]+", "", s.strip().upper())


def read_excel_smart(file_obj) -> pd.DataFrame:
    """เลือกชีทที่ใช่ + หาแถวหัวตารางจริงโดยสแกน 10 แถวแรก"""
    data = file_obj.read()
    xls = pd.ExcelFile(BytesIO(data))
    # เลือกชีท
    if "ยอดขาย" in xls.sheet_names:
        target_sheet = "ยอดขาย"
    else:
        target_sheet = None
        for sh in xls.sheet_names:
            probe = pd.read_excel(BytesIO(data), sheet_name=sh, nrows=1, header=None)
            if probe.astype(str).apply(lambda s: s.str.contains("ราคา", na=False)).any(axis=None):
                target_sheet = sh
                break
        if not target_sheet:
            target_sheet = xls.sheet_names[0]
    # หา header row
    df_probe = pd.read_excel(BytesIO(data), sheet_name=target_sheet, header=None, dtype=str)
    best_row, best_score = 0, -1
    cand_set = {_canon(h) for h in CAND_HEADERS}
    for i in range(min(10, len(df_probe))):
        row = [str(x) if pd.notna(x) else "" for x in df_probe.iloc[i].tolist()]
        score = sum(1 for v in row if _canon(v) in cand_set)
        if score > best_score:
            best_score, best_row = score, i
    # อ่านจริงด้วย header=best_row
    return pd.read_excel(BytesIO(data), sheet_name=target_sheet, header=best_row, dtype=str)

# ==================== CIA001 SQL ====================
def make_row_sql_cia001(row: pd.Series, ts: str) -> str:
    raw_code   = nz_str(row.get("ITEMCODE", ""))
    itemcode   = raw_code.zfill(12) if raw_code else ""
    scancode1  = nz_str(row.get("SCANCODE1", ""))
    itemname   = nz_str(row.get("ITEMNAME", ""))

    dept       = "bewild"
    parm       = nz_str(row.get("ITEMPARMCODE", "000001"))
    taxcode_1  = nz_str(row.get("TAXCODE_1", "01"))

    unitweight = to_int0(row.get("UNITWEIGHT", 0), 0)
    unitqty    = to_int0(row.get("UNITQTY",   1), 1)
    unitprice  = to_int0(row.get("UNITPRICE", 0), 0)

    delete_sql = f"DELETE FROM CIA001 WHERE ITEMCODE='{sql_str(itemcode)}';"
    insert_sql = (
        "INSERT INTO CIA001 (ITEMCODE, SCANCODE1, ITEMNAME, ITEMDEPTCODE, ITEMPARMCODE, "
        "UNITWEIGHT, UNITQTY, UNITPRICE, TAXCODE_1, CREATEDATETIME, UPDATEDATETIME) VALUES "
        f"('{sql_str(itemcode)}','{sql_str(scancode1)}','{sql_str(itemname)}','{dept}','{sql_str(parm)}',"
        f"{unitweight},{unitqty},{unitprice},'{sql_str(taxcode_1)}','{ts}','{ts}');"
    )
    return delete_sql + "\n" + insert_sql


def build_sql_cia001(df: pd.DataFrame) -> str:
    ts = now_casio_fmt()
    lines = ["BEGIN TRANSACTION;"]
    for _, r in df.iterrows():
        lines.append(make_row_sql_cia001(r, ts))
    lines.append("COMMIT;")
    return "\n".join(lines)

# ==================== Normalizer (อัตโนมัติ) ====================

def normalize_uploaded_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    ใช้ 'ราคาขาย' (เฉพาะเลขล้วน) + override โปร:
      - 3 ชิ้น 100      -> 50 บาท/ชิ้น
      - 4 ชิ้น 100      -> 35 บาท/ชิ้น
      - 50 / 2 ชิ้น 100 -> 80 บาท/ชิ้น
    แล้วค่อยแปลงเป็นสตางค์
    """
    def canon(s: str) -> str:
        if s is None: return ""
        s = str(s).strip().upper()
        return re.sub(r"[\s_\-\+\.\(\)\[\]\{\}/\\]+", "", s)

    cols = list(df_raw.columns)
    cmap = {canon(c): c for c in cols}

    def pick_exact(names):
        for n in names:
            k = canon(n)
            if k in cmap:
                return cmap[k]
        return None

    col_itemcode = pick_exact(["รหัสสินค้า","ITEM CODE","ITEMCODE","SAPID","MATERIAL","MATERIAL ID"])
    col_itemname = pick_exact(["ชื่อสินค้า","ITEMNAME","NAMEITEM","NAME ITEM","SKU DESCRIPTION","รายการสินค้า"])
    col_barcode  = pick_exact(["บาร์โค้ด","BARCODE","UNIT BARCODE","SCANCODE1"])
    col_price    = pick_exact(["ราคาขาย","PRICE","UNIT PRICE","RETAIL PRICE","ราคาต่อหน่วย"])
    col_unitqty  = pick_exact(["UNITQTY","QTY","PACK","ชิ้นต่อแพ็ค","รวมชิ้นต่อแพ็ค","หน่วยต่อแพ็ค"])

    out = pd.DataFrame()
    out["ITEMCODE"]  = df_raw[col_itemcode] if col_itemcode else ""
    out["ITEMNAME"]  = df_raw[col_itemname] if col_itemname else ""
    out["SCANCODE1"] = df_raw[col_barcode]  if col_barcode  else ""
    out["UNITQTY"]   = pd.to_numeric(df_raw[col_unitqty], errors="coerce").fillna(1).astype(int) if col_unitqty else 1

    # 1) ใช้ 'ราคาขาย' เฉพาะที่เป็นเลขล้วน
    if col_price:
        rp = df_raw[col_price].astype(str).str.strip()
        is_num = rp.str.fullmatch(r"[+-]?\d+(?:[.,]\d+)?")
        base_baht = pd.to_numeric(
            rp.str.replace(",", "", regex=False).str.replace("฿","", regex=False),
            errors="coerce"
        ).where(is_num)
    else:
        base_baht = pd.Series([None]*len(df_raw), index=df_raw.index, dtype="float")
    out["UNITPRICE"] = base_baht

    # 2) Override โปรข้อความ (ค่าคงที่)
    trans = str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789")
    row_texts = df_raw.apply(lambda r: " ".join([str(v) for v in r.values if pd.notna(v)]).translate(trans).lower(), axis=1)
    promo_rules = [
        (re.compile(r"3\s*ชิ้น\s*100"), 50.0),
        (re.compile(r"4\s*ชิ้น\s*100"), 35.0),
        (re.compile(r"50\s*/\s*2\s*ชิ้น\s*100"), 80.0),
    ]
    override = pd.Series([None]*len(df_raw), index=df_raw.index, dtype="object")
    for pat, baht in promo_rules:
        override.loc[row_texts.str.contains(pat, regex=True, na=False)] = baht
    out.loc[override.notna(), "UNITPRICE"] = override[override.notna()].astype(float)

    # 3) ค่าคงที่เครื่อง
    out["ITEMPARMCODE"] = "000001"
    out["UNITWEIGHT"]   = 0
    out["TAXCODE_1"]    = "01"

    # 4) ทำความสะอาด + แปลง บาท→สตางค์
    for c in ["ITEMCODE","ITEMNAME","SCANCODE1","ITEMPARMCODE","TAXCODE_1"]:
        out[c] = out[c].astype("string").fillna("").astype(str).str.strip()
    out["UNITPRICE"] = out["UNITPRICE"].fillna(0).apply(to_satang)

    # 5) ตัดแถวว่าง
    out = out[~((out["ITEMCODE"] == "") & (out["ITEMNAME"] == ""))].reset_index(drop=True)
    return out

# ==================== EJ Parsing ====================
EJ_ENCODINGS = ["utf-8-sig", "utf-8", "cp874", "tis-620", "utf-16le"]
NON_ITEM_KEYWORDS = ("รวม","ยอดสุทธิ","เงินสด","ทอน","บัตร","รับชำระ","ชำระ","ส่วนลด","คูปอง","VAT","ภาษี","หัวบิล","ท้ายบิล","ยกเลิก","VOID")
# ตัวอย่างบรรทัดสินค้าใน EJ มักเป็น: "2  Product Name      140.00" หรือมีช่องว่าง/ลบ
PAT_LINE_ITEM = re.compile(r"^\s*(?P<qty>\d+)\s+(?P<name>.+?)\s+(?P<amt>-?[\d\.,\(\)]+)\s*$")


def read_text_try(b: bytes) -> str:
    for enc in EJ_ENCODINGS:
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")


def num_from_text(s: str) -> float:
    s = s.replace(",", "").replace("฿", "").strip()
    s = s.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    try:
        return float(s)
    except Exception:
        return 0.0


def df_to_excel_bytes(df: pd.DataFrame, sheet_name="สรุปตามสินค้า") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.getvalue()


def parse_ej_text(txt: str):
    """คืนค่า (receipts, items) โดย items = list of dict(name, qty, amount).
    ปรับให้ทนทานขึ้นกับไฟล์ที่ต่างกัน: ไม่ต้องพึ่ง Ureceipt, รองรับ CRLF/CR, และบรรทัดยกเลิก/ส่วนลดหลายแบบ"""
    # ทำให้ขึ้นบรรทัดด้วย 
 เสมอ
    txt = txt.replace("
", "
").replace("
", "
")

    receipts = []
    items = []

    # ตัดเป็นบล็อกด้วย S ... E (เริ่มด้วย S ที่ต้นบรรทัด)
    blocks = re.split(r"
(?=S
)", "
" + txt)
    for blk in blocks:
        if not blk.strip().startswith("S
"):
            continue
        mode = None
        price_total = None
        canceled = False

        # เก็บบรรทัดที่ขึ้นต้นด้วย B ภายในบล็อก
        b_lines = []
        for raw in blk.splitlines():
            if raw.startswith("HMODE="):
                mode = raw.split("=",1)[1].strip()
            elif raw.startswith("HPRICE="):
                price_total = raw.split("=",1)[1].strip()
            elif raw.startswith("B"):
                t = raw[1:].strip()
                if any(k in t for k in ["ยกเลิก","VOID","Cancel","CANCEL"]):
                    canceled = True
                b_lines.append(t)

        # เฉพาะบิลขาย (REG หรือไม่มี HMODE ก็ยอมรับ) + ไม่ถูกยกเลิก
        if mode not in (None, "REG", "REG "):
            continue
        if canceled:
            continue

        # แปลงรายการสินค้า
        for t in b_lines:
            # ข้ามบรรทัดสรุป/ส่วนลด/ภาษี/ชำระเงิน
            if any(k in t for k in NON_ITEM_KEYWORDS + ("ส่วนลดพิเศษ","คูปองส่วนลด","เงินทอน","รวมทั้งสิ้น","สุทธิ")):
                continue
            m = PAT_LINE_ITEM.match(t)
            if not m:
                continue
            name = m.group("name").strip()
            qty = int(m.group("qty"))
            amt_text = m.group("amt").strip()
            # รองรับรูปแบบ (140.00) เป็นเลขลบ
            if amt_text.startswith("(") and amt_text.endswith(")"):
                amt_text = "-" + amt_text[1:-1]
            amt = num_from_text(amt_text)
            items.append({"name": name, "qty": qty, "amount": amt})

        if price_total and price_total.strip():
            receipts.append({"amount": num_from_text(price_total)})

    return pd.DataFrame(receipts), pd.DataFrame(items)


def summarize_items(df_items: pd.DataFrame) -> pd.DataFrame:
    if df_items.empty:
        return pd.DataFrame(columns=["สินค้า", "จำนวนชิ้น", "ยอดเงิน"])
    g = (
        df_items
        .groupby("name", as_index=False)
        .agg(qty=("qty","sum"), amount=("amount","sum"))
        .rename(columns={"name":"สินค้า", "qty":"จำนวนชิ้น", "amount":"ยอดเงิน"})
        .sort_values(["จำนวนชิ้น","ยอดเงิน"], ascending=[False, False])
    )
    return g

# ==================== Tabs ====================
tab_prod, tab_sales = st.tabs(["🏷️ สินค้า (CIA001)", "📊 ยอดขาย (EJ)"])

# ===== TAB 1: CIA001 =====
with tab_prod:
    st.markdown("### อัปโหลด Excel/CSV หลายรายการ")
    st.caption("อัตโนมัติ: จับชีท + หัวตาราง + คอลัมน์เอง • ใช้เฉพาะ 'ราคาขาย' + โปรคงที่")

    uploaded = st.file_uploader("เลือกไฟล์ Excel หรือ CSV", type=["xlsx", "csv"], key="up_prod")
    if uploaded is not None:
        with st.spinner("🔄 กำลังประมวลผลไฟล์..."):
            if uploaded.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
            else:
                df_raw = read_excel_smart(uploaded)
            df = normalize_uploaded_df(df_raw)

        st.success(f"✅ นำเข้าสำเร็จ {len(df):,} รายการ")
        with st.expander("👀 ดูข้อมูลที่นำเข้า (30 รายการแรก)", expanded=True):
            st.dataframe(df.head(30), use_container_width=True, hide_index=True)

        sql_text = build_sql_cia001(df)
        with st.expander("📄 ดู SQL (ตัวอย่าง 50 บรรทัดแรก)"):
            st.code("\n".join(sql_text.splitlines()[:50]) + "\n...", language="sql")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ ดาวน์โหลด SQL (เลือก Encoding ด้านซ้าย)", export_bytes(sql_text),
                               file_name="CIA001_bulk_import.sql", mime="text/plain", use_container_width=True)
        with c2:
            st.download_button("📊 Export CSV (สำรองข้อมูล)", export_csv_bytes(df),
                               file_name="CIA001_data_backup.csv", mime="text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("### เพิ่มสินค้าทีละรายการ")
    if "one_sql" not in st.session_state: st.session_state.one_sql = ""
    with st.form("one_item"):
        c1, c2 = st.columns(2)
        with c1:
            itemcode = st.text_input("🔢 รหัสสินค้า (SKU)", "")
            itemname = st.text_input("📝 ชื่อสินค้า", "")
        with c2:
            scancode1 = st.text_input("📱 บาร์โค้ด (ชิ้น)", "")
            unitqty   = st.number_input("📦 จำนวนต่อหน่วย", min_value=1, step=1, value=1)
        price_baht = st.text_input("💰 ราคาต่อชิ้น (บาท)", "", placeholder="179.00")
        unitprice  = to_satang(price_baht)
        submitted_one = st.form_submit_button("✨ สร้าง SQL", use_container_width=True)

    if submitted_one:
        row = {
            "ITEMCODE": itemcode, "SCANCODE1": scancode1, "ITEMNAME": itemname,
            "UNITQTY": unitqty, "UNITPRICE": unitprice,
            "ITEMPARMCODE": "000001", "UNITWEIGHT": 0, "TAXCODE_1": "01"
        }
        st.session_state.one_sql = "BEGIN TRANSACTION;\n" + make_row_sql_cia001(pd.Series(row), now_casio_fmt()) + "\nCOMMIT;"
        st.success("✅ สร้าง SQL สำเร็จ!")
    if st.session_state.one_sql:
        with st.expander("📄 ดู SQL ที่สร้าง", expanded=True):
            st.code(st.session_state.one_sql, language="sql")
        st.download_button("⬇️ ดาวน์โหลด SQL (รายการเดียว)", export_bytes(st.session_state.one_sql),
                           file_name="CIA001_single_item.sql", mime="text/plain", use_container_width=True)

# ===== TAB 2: EJ =====
with tab_sales:
    st.markdown("### วิเคราะห์ยอดขายจากไฟล์ EJ")
    st.caption("อัปโหลด log_YYYYMMDD.txt จากเครื่อง V-R100 (อัปได้หลายไฟล์) — สรุปยอดขายตามบิลและตามสินค้า")

    files = st.file_uploader("เลือกไฟล์ EJ (*.txt)", type=["txt"], accept_multiple_files=True, key="up_ej_logs")
    if files:
        receipts_all, items_all = [], []
        with st.spinner("🔄 กำลังประมวลผลไฟล์..."):
            for f in files:
                b = f.read()
                txt = read_text_try(b)
                r, it = parse_ej_text(txt)
                if not r.empty:
                    receipts_all.append(r)
                if not it.empty:
                    items_all.append(it)

        df_receipts = pd.concat(receipts_all, ignore_index=True) if receipts_all else pd.DataFrame(columns=["amount"])\
                        .astype({"amount":"float"})
        df_items    = pd.concat(items_all,    ignore_index=True) if items_all    else pd.DataFrame(columns=["name","qty","amount"])

        # KPI
        total_receipts = len(df_receipts)
        total_amount = float(df_receipts["amount"].sum()) if total_receipts else float(df_items["amount"].sum())
        total_qty = int(df_items["qty"].sum()) if not df_items.empty else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนบิล (สำเร็จ)", f"{total_receipts:,}")
        c2.metric("จำนวนชิ้น (รวม)", f"{total_qty:,}")
        c3.metric("ยอดขายรวม", f"{total_amount:,.2f}")

        st.markdown("#### 📦 สรุปยอดตามสินค้า")
        df_sum = summarize_items(df_items)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Export CSV — สรุปตามสินค้า", export_csv_bytes(df_sum),
                               file_name="EJ_items_summary.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("⬇️ Export Excel — สรุปตามสินค้า", df_to_excel_bytes(df_sum),
                               file_name="EJ_items_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

# ==================== Footer ====================
st.markdown("---")
st.caption("💾 อย่าลืม Restart App หลังนำเข้า SQL • โปร: 3ชิ้น100→50฿, 4ชิ้น100→35฿, 50/2ชิ้น100→80฿ • ITEMPARMCODE=000001 • TAXCODE_1=01")
