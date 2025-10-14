# Casio V-R100 Tools: CIA001 SQL + EJ Viewer (เพิ่มสรุปยอดขายตามสินค้า)
# run:  streamlit run Codeapp.py
# pip:  pip install streamlit pandas openpyxl pillow xlsxwriter

import streamlit as st
import pandas as pd
import re, math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from pathlib import Path
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
  .stButton>button:hover{background:linear-gradient(90deg,#2196f3 0%,#1976d2 100%);transform:translateY(-2px)}
  .stDownloadButton>button{background:linear-gradient(90deg,#26c6da 0%,#00acc1 100%);color:#fff;border:none;border-radius:8px;
    padding:.5rem 1.5rem;font-weight:500;box-shadow:0 2px 5px rgba(0,172,193,.3)}
  .stDownloadButton>button:hover{background:linear-gradient(90deg,#00acc1 0%,#0097a7 100%);transform:translateY(-2px)}
  .stForm{background:#fff;padding:1.5rem;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.05);border:1px solid #e3f2fd}
  .stTabs [data-baseweb="tab-list"]{gap:8px;background:#fff;padding:.5rem;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
  .stTabs [data-baseweb="tab"]{border-radius:8px;color:#42a5f5;font-weight:500}
  .stTabs [aria-selected="true"]{background:linear-gradient(90deg,#42a5f5 0%,#2196f3 100%);color:#fff}
  h1,h2,h3{color:#1565c0}
  .stDataFrame{border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.05)}
  [data-testid="stFileUploader"]{background:#fff;padding:1rem;border-radius:10px;border:2px dashed #90caf9}
</style>
""", unsafe_allow_html=True)

# ==================== Header ====================
def show_header():
    st.markdown("""
    <div style='text-align:center;margin:.75rem 0 1.5rem 0;'>
      <div style='font-size:42px;font-weight:700;background:linear-gradient(135deg,#42a5f5 0%,#2196f3 100%);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>bewild</div>
      <div style='color:#64b5f6'>Casio V-R100 Tools · แปลง Excel/CSV เป็น SQL (CIA001) · วิเคราะห์ยอดขาย EJ</div>
    </div>
    """, unsafe_allow_html=True)

show_header()

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("### ⚙️ การตั้งค่า")
    st.markdown("---")
    vr100_enc = st.selectbox("Encoding ไฟล์ SQL", ["UTF-8 (ปกติ)", "UTF-8 with BOM (UTF-8-SIG)"], index=0)
    st.caption("• ราคาใช้เฉพาะคอลัมน์ 'ราคาขาย' • แถวที่มี '3 ชิ้น 100' หรือ '4 ชิ้น 100' จะตั้งราคาเป็น 0 ก่อน")

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

def is_numeric_series(sr: pd.Series, sample=50, min_ratio=0.6) -> bool:
    vals = sr.dropna().astype(str)
    if vals.empty:
        return False
    vals = vals.head(sample)
    ok = 0
    for v in vals:
        v = v.strip().replace(",", "")
        try:
            float(v)
            ok += 1
        except Exception:
            pass
    return ok / len(vals) >= min_ratio

# บาท → สตางค์ (robust)
def to_satang(x) -> int:
    if x is None:
        return 0
    try:
        if (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
            return 0
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return 0
    s = re.sub(r"[^\d\.\-]", "", s)  # keep digits/dot/minus
    if s in {"", ".", "-", "-.", ".-"}:
        return 0
    try:
        d = Decimal(s).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return int(d * 100)
    except Exception:
        return 0

# อ่าน Excel เดาหัวตารางอัตโนมัติ
def read_excel_smart(file_obj) -> pd.DataFrame:
    data = file_obj.read()
    buf = BytesIO(data)
    probe = pd.read_excel(buf, header=None, dtype=str)
    header_row = 0
    candidates = {
        "SKU CODE", "MATERIAL ID", "MATERIAL",
        "SKU DESCRIPTION", "NAMEITEM", "NAME+ITEM", "ชื่อสินค้า",
        "UNIT BARCODE", "บาร์โค้ด (ชิ้น)", "บาร์โค้ด", "ราคาขาย"
    }
    for i in range(min(10, len(probe))):
        row = [str(x).strip() if pd.notna(x) else "" for x in probe.iloc[i].tolist()]
        if candidates & {v.upper() for v in row if v}:
            header_row = i
            break
    buf2 = BytesIO(data)
    return pd.read_excel(buf2, header=header_row, dtype=str)

# ==================== CIA001 SQL ====================
def make_row_sql_cia001(row: pd.Series, ts: str) -> str:
    itemcode   = str(row.get("ITEMCODE", "")).strip().zfill(12)
    scancode1  = str(row.get("SCANCODE1", "")).strip()
    itemname   = str(row.get("ITEMNAME", "")).strip()
    dept       = "bewild"
    parm       = str(row.get("ITEMPARMCODE", "000001")).strip()
    unitweight = float(pd.to_numeric(row.get("UNITWEIGHT", 0), errors="coerce") or 0)
    unitqty    = int(pd.to_numeric(row.get("UNITQTY", 1), errors="coerce") or 1)
    unitprice  = int(pd.to_numeric(row.get("UNITPRICE", 0), errors="coerce") or 0)
    taxcode_1  = str(row.get("TAXCODE_1", "01")).strip()

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

# ==================== Normalizer ====================
def normalize_uploaded_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    แมปคอลัมน์จากไฟล์ผู้ใช้ -> CIA001:
      - ราคาใช้เฉพาะคอลัมน์ 'ราคาขาย'
      - ถ้าเจอข้อความ '3 ชิ้น 100' หรือ '4 ชิ้น 100' ในแถวนั้น (คอลัมน์ใดก็ได้) → UNITPRICE = 0
      - บาร์โค้ดว่างให้ว่าง ไม่เติมเอง
    """
    def canon(s: str) -> str:
        if s is None:
            return ""
        s = str(s).strip().upper()
        return re.sub(r"[\s_\-\+\.\(\)\[\]\{\}\/\\]+", "", s)

    cols = list(df_raw.columns)
    cmap = {canon(c): c for c in cols}

    def pick_exact(names: list[str]) -> str | None:
        for n in names:
            k = canon(n)
            if k in cmap:
                return cmap[k]
        return None

    col_itemcode = pick_exact(["SKU CODE", "MATERIAL ID", "MATERIAL", "ITEM CODE", "รหัสสินค้า"])
    col_itemname = pick_exact(["nameItem", "NAMEITEM", "SKU DESCRIPTION", "NAME+ITEM", "ITEMNAME", "ชื่อสินค้า", "รายการสินค้า"])

    barcode_candidates = [c for c in cols if any(k in canon(c) for k in ["BARCODE", "บาร์โค้ด"])]
    col_barcode = barcode_candidates[0] if barcode_candidates else None

    qty_candidates = [c for c in cols if any(k in canon(c) for k in ["UNITQTY", "QTY", "PACK", "ชิ้นต่อแพ็ค", "รวมชิ้นต่อแพ็ค", "ในกล่อง", "หน่วย"])]
    col_unitqty = qty_candidates[0] if qty_candidates else None

    col_price = pick_exact(["ราคาขาย"])
    col_date = pick_exact(["DATE", "วันที่"])

    out = pd.DataFrame()
    if col_date:
        out["DATE"] = df_raw[col_date]

    out["ITEMCODE"]  = df_raw[col_itemcode] if col_itemcode else ""
    out["ITEMNAME"]  = df_raw[col_itemname] if col_itemname else ""
    out["SCANCODE1"] = df_raw[col_barcode]  if col_barcode  else ""
    out["UNITQTY"]   = df_raw[col_unitqty]  if col_unitqty  else 1

    price_series = df_raw[col_price] if col_price else pd.Series([0] * len(df_raw))

    promo_mask = pd.Series(False, index=df_raw.index)
    patterns = [r"3\s*ชิ้น\s*100", r"4\s*ชิ้น\s*100"]
    for c in cols:
        s = df_raw[c].astype(str).fillna("")
        for pat in patterns:
            promo_mask |= s.str.contains(pat, flags=re.IGNORECASE, regex=True)

    out["UNITPRICE"] = price_series
    out.loc[promo_mask, "UNITPRICE"] = 0

    out["ITEMPARMCODE"] = "000001"
    out["UNITWEIGHT"]   = 0
    out["TAXCODE_1"]    = "01"

    for c in ["ITEMCODE", "SCANCODE1", "ITEMNAME", "ITEMPARMCODE", "TAXCODE_1"]:
        out[c] = out[c].astype(str).str.strip()

    out["UNITQTY"]   = pd.to_numeric(out.get("UNITQTY", 1), errors="coerce").fillna(1).astype(int)
    out["UNITPRICE"] = out["UNITPRICE"].fillna(0).apply(to_satang)

    out = out[~((out["ITEMCODE"] == "") & (out["ITEMNAME"] == ""))].reset_index(drop=True)

    if "DATE" in out.columns:
        def fmt_date(s):
            s = str(s).strip()
            if re.fullmatch(r"\d{8}", s):
                return f"{s[0:4]}/{s[4:6]}/{s[6:8]}"
            m = re.match(r"(\d{1,2})\.(\d{1,2})\.(\d{4})$")
            if m:
                d, mth, y = m.groups()
                return f"{y}/{int(mth):02d}/{int(d):02d}"
            return s
        out["DATE"] = out["DATE"].apply(fmt_date)

    return out

# ==================== EJ Parsing (เพิ่มสรุปรายการสินค้า) ====================
EJ_ENCODINGS = ["utf-8-sig", "utf-8", "cp874", "tis-620", "utf-16le"]
NON_ITEM_KEYWORDS = (
    "รวม", "ยอดสุทธิ", "เงินสด", "ทอน", "บัตร", "รับชำระ", "ชำระ",
    "ส่วนลด", "คูปอง", "VAT", "ภาษี", "หัวบิล", "ท้ายบิล"
)
PAT_DATE_TIME_LINE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?:\s+\d+)?$")
PAT_QTYxPRICE = re.compile(r"^(?P<name>.+?)\s+(?P<qty>\d+)\s*[x×]\s*(?P<unit>[\d\.,]+)\s*=\s*(?P<amt>[\d\.,]+)$")
PAT_DATE_TIME_ANY = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}")
PAT_NAME_HAS_LETTER = re.compile(r"[A-Za-zก-๙]")
PAT_TAIL_AMOUNT = re.compile(r"^(?P<name>.+?)\s+(?P<amt>[\d\.,]+)$")
PAT_RECEIPT_NO = re.compile(r"(ใบเสร็จ|เลขที่บิล|RECEIPT|HINVOICENUMBER).*?(\d+)", re.IGNORECASE)

def read_text_try(b: bytes) -> str:
    for enc in EJ_ENCODINGS:
        try:
            return b.decode(enc)
        except:
            continue
    return b.decode("utf-8", errors="ignore")

def num_from_text(s: str) -> float:
    s = s.replace(",", "").replace("฿", "").strip()
    s = s.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    try:
        return float(s)
    except:
        return 0.0

def df_to_excel_bytes(df: pd.DataFrame, sheet_name="สรุปตามสินค้า") -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.getvalue()


def looks_like_discount(name: str) -> bool:
    n = name.upper()
    return n.startswith("DISC") or "ส่วนลด" in name

def parse_blocks_from_lines(lines: list[str]):
    blocks, curr = [], []
    for ln in lines:
        if ln.strip() == "S":
            curr = [ln]
        elif ln.strip() == "E":
            curr.append(ln); blocks.append(curr); curr = []
        elif curr:
            curr.append(ln)
    return blocks

def parse_receipts_and_items_from_ej_bytes(b: bytes):
    """คืนค่า (df_receipts, df_item_rows)"""
    txt = read_text_try(b)
    lines = [ln.rstrip("\r\n") for ln in txt.splitlines()]
    blocks = parse_blocks_from_lines(lines)

    receipts, item_rows = [], []
    for blk in blocks:
        hdr, body = {}, []
        for ln in blk:
            if ln.startswith("H") and "=" in ln:
                k, v = ln.split("=", 1); hdr[k.strip()] = v.strip()
            if ln.startswith("B"):
                body.append(ln[1:].rstrip())

        # เฉพาะบิลขายปกติ
        if hdr.get("HMODE","") == "REG" and hdr.get("HINVOICENUMBER","") != "000000":
            # รายการบิลรวม
            amt = hdr.get("HPRICE","0").replace(",","")
            try: amt = float(amt)
            except: amt = 0.0
            receipts.append({
                "DATE": hdr.get("HBIZDATE",""),
                "TIME": hdr.get("HINVOICETIME",""),
                "INVOICE": hdr.get("HINVOICENUMBER",""),
                "AMOUNT": amt
            })

            # รายการสินค้าในบิล (จากบรรทัด B...)
            curr_receipt = hdr.get("HINVOICENUMBER","")
            for raw in body:
                line = " ".join(raw.strip().split())
                if not line:
                    continue
                # ข้ามบรรทัดวันที่/เวลา/บิล เช่น 14/10/2025 9:31 หรือมีเลขบิลต่อท้าย
                if PAT_DATE_TIME_LINE.match(line) or PAT_DATE_TIME_ANY.search(line):
                    continue
                if any(k in line for k in NON_ITEM_KEYWORDS):
                    continue
                m1 = PAT_QTYxPRICE.match(line)
                if m1:
                    name = m1.group("name").strip()
                    name = re.sub(r'^\d+\s*', '', name)  # ลบเลขขึ้นต้นชื่อ (เฉพาะจุดนี้)
                    if looks_like_discount(name):  # ตัดส่วนลด/บรรทัดไม่ใช่สินค้า
                        continue
                    qty = int(m1.group("qty"))
                    amt = num_from_text(m1.group("amt"))
                    item_rows.append({"receipt": curr_receipt, "name": name, "qty": qty, "amount": amt})
                    continue
                m2 = PAT_TAIL_AMOUNT.match(line)
                if m2:
                    name = m2.group("name").strip()
                    name = re.sub(r'^\d+\s*', '', name)  # ลบเลขขึ้นต้นชื่อ (เฉพาะจุดนี้)
                    if not PAT_NAME_HAS_LETTER.search(name):
                        continue
                    if looks_like_discount(name) or any(k in name for k in NON_ITEM_KEYWORDS):
                        continue
                    amt = num_from_text(m2.group("amt"))
                    if amt == 0:
                        continue
                    item_rows.append({"receipt": curr_receipt, "name": name, "qty": 1, "amount": amt})
                    continue

    df_r = pd.DataFrame(receipts) if receipts else pd.DataFrame(columns=["DATE","TIME","INVOICE","AMOUNT"])
    df_i = pd.DataFrame(item_rows) if item_rows else pd.DataFrame(columns=["receipt","name","qty","amount"])
    return df_r, df_i

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
    st.caption("ต้องมีคอลัมน์: **SKU CODE (หรือ MATERIAL ID), ชื่อสินค้า (เช่น nameItem/SKU DESCRIPTION), บาร์โค้ด (ถ้ามี), จำนวนต่อหน่วย (ถ้ามี), และ 'ราคาขาย'** • แถวที่มีข้อความ **3 ชิ้น 100/4 ชิ้น 100** จะตั้งราคาเป็น 0 ชั่วคราว")

    uploaded = st.file_uploader("เลือกไฟล์ Excel หรือ CSV", type=["xlsx", "csv"], key="up_prod",
                                help="รองรับ .xlsx และ .csv • หากไฟล์มีบรรทัด 'Update:' อยู่เหนือหัวตาราง ระบบจะจับหัวให้เอง")
    if uploaded is not None:
        with st.spinner("🔄 กำลังประมวลผลไฟล์..."):
            if uploaded.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
            else:
                df_raw = read_excel_smart(uploaded)
            df = normalize_uploaded_df(df_raw)

        st.success(f"✅ นำเข้าสำเร็จ **{len(df)}** รายการ")
        with st.expander("👀 ดูข้อมูลที่นำเข้า (20 รายการแรก)", expanded=True):
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)

        sql_text = build_sql_cia001(df)
        with st.expander("📄 ดู SQL Code (50 บรรทัดแรก)"):
            st.code("\n".join(sql_text.split("\n")[:50]) + "\n\n... (ดูเพิ่มเติมในไฟล์ที่ดาวน์โหลด)", language="sql")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(f"⬇️ ดาวน์โหลด SQL ({len(df)} รายการ)", export_bytes(sql_text),
                               file_name="CIA001_bulk_import.sql", mime="text/plain", use_container_width=True)
        with c2:
            st.download_button("📊 Export ข้อมูลเป็น CSV", export_csv_bytes(df),
                               file_name="CIA001_data_backup.csv", mime="text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("### เพิ่มสินค้าทีละรายการ")
    if "one_sql" not in st.session_state:
        st.session_state.one_sql = ""
    with st.form("one_item", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            itemcode = st.text_input("🔢 รหัสสินค้า (SKU)", placeholder="12345")
            itemname = st.text_input("📝 ชื่อสินค้า", placeholder="เช่น กาแฟเย็น")
        with c2:
            scancode1 = st.text_input("📱 บาร์โค้ด (ชิ้น)", placeholder="8850123456789")
            unitqty   = st.number_input("📦 จำนวนต่อหน่วย", min_value=1, step=1, value=1)
        price_baht = st.text_input("💰 ราคาต่อชิ้น (บาท)", value="", placeholder="179.00")
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
        with st.expander("📄 ดู SQL Code", expanded=True):
            st.code(st.session_state.one_sql, language="sql")
        st.download_button("⬇️ ดาวน์โหลด SQL", export_bytes(st.session_state.one_sql),
                           file_name="CIA001_single_item.sql", mime="text/plain", use_container_width=True)

# ===== TAB 2: EJ =====
with tab_sales:
    st.markdown("### วิเคราะห์ยอดขายจากไฟล์ EJ")
    st.caption("อัปโหลด `log_YYYYMMDD.txt` จากเครื่อง V-R100 (อัปได้หลายไฟล์)")

    files = st.file_uploader("เลือกไฟล์ EJ (*.txt)", type=["txt"], accept_multiple_files=True,
                             key="up_ej_logs", help="ระบบจะรวมหลายวันให้อัตโนมัติ")
    if files:
        receipts_all, items_all = [], []
        with st.spinner("🔄 กำลังประมวลผลไฟล์..."):
            for f in files:
                b = f.read()
                df_r, df_i = parse_receipts_and_items_from_ej_bytes(b)
                if not df_r.empty:
                    receipts_all.append(df_r)
                if not df_i.empty:
                    items_all.append(df_i)

        # ----- ภาพรวมต่อวัน -----
        df_receipts = pd.concat(receipts_all, ignore_index=True) if receipts_all else pd.DataFrame(columns=["DATE","TIME","INVOICE","AMOUNT"])
        if df_receipts.empty:
            st.warning("⚠️ ไม่พบข้อมูลขายในไฟล์ที่อัปโหลด")
        else:
            dates = sorted(df_receipts["DATE"].unique())
            st.success(f"✅ โหลดข้อมูลสำเร็จ {len(dates)} วัน")
            sel_date = st.selectbox("📅 เลือกวันที่ต้องการดู", options=dates, index=len(dates)-1)
            sub = df_receipts[df_receipts["DATE"] == sel_date]
            col1, col2 = st.columns(2)
            col1.metric("🧾 จำนวนบิล", f"{len(sub):,}")
            col2.metric("💰 ยอดรวมทั้งวัน", f"{sub['AMOUNT'].sum():,.2f} ฿")
            st.dataframe(sub.rename(columns={"TIME":"เวลา","INVOICE":"เลขที่บิล","AMOUNT":"ยอดเงิน (฿)"}),
                         use_container_width=True, hide_index=True)

        # ----- สรุปตามสินค้า -----
        st.markdown("---")
        st.markdown("### 📦 สรุปรายการขายตามสินค้า")
        df_items = pd.concat(items_all, ignore_index=True) if items_all else pd.DataFrame(columns=["receipt","name","qty","amount"])
        if df_items.empty:
            st.info("ยังไม่พบรูปแบบบรรทัดรายการในไฟล์ EJ — หาก EJ ของคุณใช้รูปแบบพิเศษ ส่งตัวอย่าง 4–5 บรรทัดช่วงรายการมาได้เลย ผมจะปรับตัวจับให้ครับ")
        else:
            df_sum = summarize_items(df_items)
            st.metric("จำนวนรายการสินค้า (ไม่รวมส่วนลด)", len(df_sum))
            st.dataframe(df_sum, use_container_width=True, hide_index=True)

            xlsx = df_to_excel_bytes(df_sum)
            st.download_button("⬇️ ดาวน์โหลดสรุปรายการสินค้า (Excel)",
                               data=xlsx, file_name="ej_item_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

            with st.expander("ดูบรรทัดรายการที่ดึงได้ (debug)"):
                st.dataframe(df_items, use_container_width=True, hide_index=True)

# ==================== Footer ====================
st.markdown("---")
st.caption("💾 อย่าลืม Restart App หลังนำเข้า SQL • ราคาใช้เฉพาะคอลัมน์ 'ราคาขาย' • แถวโปร (3/4 ชิ้น 100) จะถูกตั้งราคาเป็น 0 ไว้ก่อน")
