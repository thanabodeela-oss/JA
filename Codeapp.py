import streamlit as st
import pandas as pd
import re
import math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from io import BytesIO

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="bewild V-R100 Tools",
    page_icon="🧾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
  .main { background: linear-gradient(135deg, #e3f2fd 0%, #fff 100%); }
  [data-testid="stSidebar"] { background: linear-gradient(180deg, #bbdefb 0%, #e3f2fd 100%); }
  .stButton>button { background: linear-gradient(90deg, #42a5f5 0%, #2196f3 100%); color:#fff; border:none; border-radius:8px; padding:.5rem 1.5rem; font-weight:500; transition:.3s; box-shadow:0 2px 5px rgba(33,150,243,.3); }
  .stButton>button:hover { transform: translateY(-2px); }
  .stDownloadButton>button { background: linear-gradient(90deg, #26c6da 0%, #00acc1 100%); color:#fff; border:none; border-radius:8px; padding:.5rem 1.5rem; font-weight:500; box-shadow:0 2px 5px rgba(0,172,193,.3); }
  .stDownloadButton>button:hover { transform: translateY(-2px); }
  .stForm { background:#fff; padding:1.5rem; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,.05); border:1px solid #e3f2fd; }
  h1,h2,h3 { color:#1565c0; }
  .stDataFrame { border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.05); }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
CANDIDATE_HEADERS = {
    "รหัสสินค้า", "ITEM CODE", "ITEMCODE", "SAPID", "MATERIAL", "MATERIAL ID",
    "ชื่อสินค้า", "ITEMNAME", "NAME ITEM", "NAMEITEM", "รายการสินค้า", "SKU DESCRIPTION",
    "บาร์โค้ด", "BARCODE", "UNIT BARCODE", "SCANCODE1",
    "UNITQTY", "QTY", "PACK", "ชิ้นต่อแพ็ค", "รวมชิ้นต่อแพ็ค", "หน่วยต่อแพ็ค",
    "PRICE", "UNIT PRICE", "RETAIL PRICE", "ราคาต่อหน่วย",
    "ราคาต่อชิ้น"
}
EJ_ENCODINGS = ["utf-8-sig", "utf-8", "cp874", "tis-620", "utf-16le"]

NON_ITEM_KEYWORDS = (
    "รวม", "ยอดสุทธิ", "เงินสด", "ทอน", "บัตร", "รับชำระ", "ชำระ",
    "ส่วนลด", "คูปอง", "VAT", "ภาษี", "หัวบิล", "ท้ายบิล", "ยกเลิก", "VOID",
    "No", "คน", "Qty change"  # กันบรรทัดหลอก
)
DISCOUNT_KEYWORDS = ("ส่วนลด", "คูปอง", "Coupon", "DISCOUNT", "โปร", "Promotion", "โปรฯ")

# ---------- Regex (รองรับจำนวนติดลบ) ----------
PAT_LINE_ITEM      = re.compile(r"^\s*(?P<qty>[+-]?\d+)\s+(?P<name>.+?)\s+(?P<amt>-?[\d\.,\(\)]+)\s*$")
PAT_DISCOUNT       = re.compile(r"^\s*(?:(?P<qty>[+-]?\d+)\s+)?(?P<name>.+?)\s+(?P<amt>-?\(?[\d\.,]+\)?)\s*$")
# 2 บรรทัด: (จำนวน+ชื่อ) ตามด้วย (ราคาอย่างเดียว)
PAT_QTY_NAME_ONLY  = re.compile(r"^\s*(?P<qty>[+-]?\d+)\s+(?P<name>.+?)\s*$")
PAT_AMOUNT_ONLY    = re.compile(r"^\s*(?P<amt>-?[\d\.,\(\)]+)\s*$")

# ==================== UTILS ====================
def canonicalize_text(text: str) -> str:
    return re.sub(r"[\s_\-\+\.\(\)\[\]\{\}/\\]+", "", text.strip().upper())

def normalize_string(value) -> str:
    try:
        if pd.isna(value): return ""
    except Exception:
        pass
    s = "" if value is None else str(value).strip()
    return "" if s.lower() == "nan" else s

def to_int_safe(value, default=0) -> int:
    try:
        x = pd.to_numeric(value, errors="coerce")
        if pd.isna(x): return default
        return int(float(x))
    except Exception:
        return default

def to_satang(value) -> int:
    if value is None: return 0
    try:
        if (isinstance(value, float) and math.isnan(value)) or pd.isna(value): return 0
    except Exception:
        pass
    s = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s): return 0
    decimal_value = Decimal(s).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(decimal_value * 100)

def sql_escape_string(s) -> str:
    return "" if s is None else str(s).replace("'", "''")

def get_casio_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")

def export_to_bytes(sql_text: str, encoding_choice: str) -> bytes:
    fixed = "\r\n".join(line.rstrip("\r\n") for line in sql_text.splitlines())
    enc = "utf-8-sig" if encoding_choice.endswith("SIG") else "utf-8"
    return fixed.encode(enc, errors="ignore")

def export_csv_to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig")

def export_excel_to_bytes(df: pd.DataFrame, sheet_name="สรุปตามสินค้า") -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer.getvalue()

def export_excel_bills_bytes(df_bills_summary: pd.DataFrame,
                             df_bills_items: pd.DataFrame,
                             df_bills_discounts: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_bills_summary.to_excel(writer, index=False, sheet_name="Bills")
        df_bills_items.to_excel(writer, index=False, sheet_name="Bill Items")
        df_bills_discounts.to_excel(writer, index=False, sheet_name="Bill Discounts")
    buffer.seek(0)
    return buffer.getvalue()

# ==================== EXCEL READING ====================
def read_excel_smart(file_obj, manual_sheet: str | None = None) -> tuple[pd.DataFrame, str, int]:
    data = file_obj.read()
    excel_file = pd.ExcelFile(BytesIO(data))
    target_sheets = [manual_sheet] if manual_sheet else excel_file.sheet_names

    best_sheet, best_row, best_score = None, 0, -1
    candidate_set = {canonicalize_text(h) for h in CANDIDATE_HEADERS}

    for sheet_name in target_sheets:
        df_probe = pd.read_excel(BytesIO(data), sheet_name=sheet_name, header=None, dtype=str)
        limit = min(20, len(df_probe))
        local_best_row, local_best_score = 0, -1
        for i in range(limit):
            row = [str(x) if pd.notna(x) else "" for x in df_probe.iloc[i].tolist()]
            score = sum(1 for v in row if canonicalize_text(v) in candidate_set)
            if any("ราคา" in str(v) for v in row): score += 2
            non_empty_cols = sum(1 for v in row if str(v).strip() != "")
            score += min(non_empty_cols, 3) * 0.1
            if score > local_best_score:
                local_best_score, local_best_row = score, i
        if local_best_score > best_score:
            best_sheet, best_row, best_score = sheet_name, local_best_row, local_best_score

    if best_sheet is None:
        best_sheet, best_row = target_sheets[0], 0

    df = pd.read_excel(BytesIO(data), sheet_name=best_sheet, header=best_row, dtype=str)
    return df, best_sheet, best_row

# ==================== NORMALIZE PRODUCT DF ====================
def normalize_uploaded_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    columns = list(df_raw.columns)
    column_map = {canonicalize_text(c): c for c in columns}

    def pick_column(names):
        for name in names:
            key = canonicalize_text(name)
            if key in column_map:
                return column_map[key]
        return None

    col_itemcode = pick_column(["รหัสสินค้า","ITEM CODE","ITEMCODE","SAPID","MATERIAL","MATERIAL ID"])
    col_itemname = pick_column(["ชื่อสินค้า","ITEMNAME","NAMEITEM","NAME ITEM","SKU DESCRIPTION","รายการสินค้า"])
    col_barcode  = pick_column(["บาร์โค้ด","BARCODE","UNIT BARCODE","SCANCODE1"])
    col_unitqty  = pick_column(["UNITQTY","QTY","PACK","ชิ้นต่อแพ็ค","รวมชิ้นต่อแพ็ค","หน่วยต่อแพ็ค"])

    col_price_piece = pick_column(["ราคาต่อชิ้น","ราคาขายต่อชิ้น"])
    col_price_fallback = pick_column(["ราคาต่อหน่วย","UNIT PRICE","RETAIL PRICE","PRICE","ราคาขาย"])

    out = pd.DataFrame()
    out["ITEMCODE"]  = df_raw[col_itemcode] if col_itemcode else ""
    out["ITEMNAME"]  = df_raw[col_itemname] if col_itemname else ""
    out["SCANCODE1"] = df_raw[col_barcode]  if col_barcode  else ""
    out["UNITQTY"]   = pd.to_numeric(df_raw[col_unitqty], errors="coerce").fillna(1).astype(int) if col_unitqty else 1

    if col_price_piece:
        raw_price = df_raw[col_price_piece].astype(str).str.strip()
        base_baht = pd.to_numeric(raw_price.str.replace(",", "", regex=False).str.replace("฿", "", regex=False), errors="coerce")
    elif col_price_fallback:
        raw_price = df_raw[col_price_fallback].astype(str).str.strip()
        is_numeric = raw_price.str.fullmatch(r"[+-]?\d+(?:[.,]\d+)?")
        base_baht = pd.to_numeric(raw_price.str.replace(",", "", regex=False).str.replace("฿", "", regex=False), errors="coerce").where(is_numeric)

        thai_to_arabic = str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789")
        row_texts = df_raw.apply(lambda r: " ".join([str(v) for v in r.values if pd.notna(v)]).translate(thai_to_arabic).lower(), axis=1)
        promo_rules = [
            (re.compile(r"3\s*ชิ้น\s*100"), 50.0),
            (re.compile(r"4\s*ชิ้น\s*100"), 35.0),
            (re.compile(r"50\s*/\s*2\s*ชิ้น\s*100"), 80.0),
        ]
        override = pd.Series([None]*len(df_raw), index=df_raw.index, dtype="object")
        for pattern, baht in promo_rules:
            override.loc[row_texts.str.contains(pattern, regex=True, na=False)] = baht
        base_baht = base_baht.astype("float")
        base_baht.loc[override.notna()] = override[override.notna()].astype(float)
    else:
        base_baht = pd.Series([None]*len(df_raw), index=df_raw.index, dtype="float")

    out["UNITPRICE"] = base_baht
    out["ITEMPARMCODE"] = "000001"
    out["UNITWEIGHT"]   = 0
    out["TAXCODE_1"]    = "01"

    for c in ["ITEMCODE","ITEMNAME","SCANCODE1","ITEMPARMCODE","TAXCODE_1"]:
        out[c] = out[c].astype("string").fillna("").astype(str).str.strip()

    out["UNITPRICE"] = out["UNITPRICE"].fillna(0).apply(to_satang)
    out = out[~((out["ITEMCODE"] == "") & (out["ITEMNAME"] == ""))].reset_index(drop=True)
    return out

# ==================== SQL GENERATION ====================
def generate_row_sql_cia001(row: pd.Series, timestamp: str) -> str:
    raw_code = normalize_string(row.get("ITEMCODE", ""))
    itemcode = raw_code.zfill(12) if raw_code else ""
    scancode1 = normalize_string(row.get("SCANCODE1", ""))
    itemname = normalize_string(row.get("ITEMNAME", ""))

    dept      = "bewild"
    parm      = normalize_string(row.get("ITEMPARMCODE", "000001"))
    taxcode_1 = normalize_string(row.get("TAXCODE_1", "01"))

    unitweight = to_int_safe(row.get("UNITWEIGHT", 0), 0)
    unitqty    = to_int_safe(row.get("UNITQTY",   1), 1)
    unitprice  = to_int_safe(row.get("UNITPRICE", 0), 0)

    delete_sql = f"DELETE FROM CIA001 WHERE ITEMCODE='{sql_escape_string(itemcode)}';"
    insert_sql = (
        "INSERT INTO CIA001 (ITEMCODE, SCANCODE1, ITEMNAME, ITEMDEPTCODE, ITEMPARMCODE, "
        "UNITWEIGHT, UNITQTY, UNITPRICE, TAXCODE_1, CREATEDATETIME, UPDATEDATETIME) VALUES "
        f"('{sql_escape_string(itemcode)}','{sql_escape_string(scancode1)}','{sql_escape_string(itemname)}','{dept}','{sql_escape_string(parm)}',"
        f"{unitweight},{unitqty},{unitprice},'{sql_escape_string(taxcode_1)}','{timestamp}','{timestamp}');"
    )
    return delete_sql + "\n" + insert_sql

def build_sql_cia001(df: pd.DataFrame) -> str:
    timestamp = get_casio_timestamp()
    lines = ["BEGIN TRANSACTION;"]
    for _, row in df.iterrows():
        lines.append(generate_row_sql_cia001(row, timestamp))
    lines.append("COMMIT;")
    return "\n".join(lines)

# ==================== EJ PARSING ====================
def read_text_with_encoding(data: bytes) -> str:
    for encoding in EJ_ENCODINGS:
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")

def extract_number_from_text(text: str) -> float:
    text = text.replace(",", "").replace("฿", "").strip()
    text = text.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))
    try:
        return float(text)
    except Exception:
        return 0.0

def clean_time_token(tok: str | None) -> str:
    if not tok: return ""
    s = re.sub(r"\D", "", str(tok).strip())
    if len(s) == 4:   # HHMM
        return f"{s[:2]}:{s[2:]}"
    if len(s) == 6:   # HHMMSS
        return f"{s[:2]}:{s[2:4]}:{s[4:]}"
    return str(tok).strip()

def clean_date_token(tok: str | None) -> str:
    if not tok: return ""
    s = str(tok).strip()
    if re.fullmatch(r"\d{8}", s):  # YYYYMMDD
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)  # DD/MM/YYYY
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}"
    return s

def format_datetime_label(d: str, t: str) -> str:
    t = clean_time_token(t or "")
    hhmm = t[:5] if ":" in t else (t[:2] + ":" + t[2:4] if len(t) >= 4 else t)
    return (d or "").strip() + (" " + hhmm if hhmm else "")

# ✅ เกณฑ์ว่าราคา “สมเหตุผล” กันเลข 01/02/03 ฯลฯ ที่เป็นส่วนหนึ่งของชื่อ
def _is_plausible_price(raw: str) -> bool:
    v = abs(extract_number_from_text(raw))
    return (v >= 5) or ("." in raw) or ("(" in raw and ")" in raw)

def parse_ej_text(text: str):
    """Parse EJ and return (receipts, items, discounts) with invoice/date/time."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    receipts, items, discounts = [], [], []

    pat_b_header = re.compile(
        r"^\s*(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s+(?P<inv>\d{3,})\s*$"
    )

    blocks = re.split(r"\n(?=S\n)", "\n" + text)
    for block in blocks:
        if not block.strip().startswith("S\n"):
            continue

        mode, price_total, canceled = None, None, False
        body_lines = []
        inv_date_raw = inv_time_raw = inv_no = None
        block_items_total = 0.0

        for raw_line in block.splitlines():
            if raw_line.startswith("HINVOICEDATE="):
                inv_date_raw = raw_line.split("=", 1)[1].strip()
            elif raw_line.startswith("HINVOICETIME="):
                inv_time_raw = raw_line.split("=", 1)[1].strip()
            elif raw_line.startswith("HINVOICENUMBER="):
                inv_no = raw_line.split("=", 1)[1].strip()
            elif raw_line.startswith("HMODE="):
                mode = raw_line.split("=", 1)[1].strip()
            elif raw_line.startswith("HPRICE="):
                price_total = raw_line.split("=", 1)[1].strip()
            elif raw_line.startswith("B"):
                text_line = raw_line[1:].strip()
                if inv_no is None:
                    mhead = pat_b_header.match(text_line)
                    if mhead:
                        inv_date_raw = mhead.group("date")
                        inv_time_raw = mhead.group("time")
                        inv_no       = mhead.group("inv")
                if any(k in text_line for k in ("ยกเลิก","VOID","Cancel","CANCEL")):
                    canceled = True
                body_lines.append(text_line)

        if mode not in (None, "REG", "REG "):
            continue
        if canceled:
            continue

        inv_date = clean_date_token(inv_date_raw) if inv_date_raw else ""
        inv_time = clean_time_token(inv_time_raw) if inv_time_raw else ""

        i = 0
        while i < len(body_lines):
            text_line = body_lines[i]

            # ส่วนลด
            if any(k in text_line for k in DISCOUNT_KEYWORDS):
                m2 = PAT_DISCOUNT.match(text_line)
                if m2:
                    qty_txt = m2.group("qty")
                    times = int(qty_txt) if qty_txt else 1
                    discount_name = m2.group("name").strip()
                    amount_text = m2.group("amt").strip()
                    if amount_text.startswith("(") and amount_text.endswith(")"):
                        amount_text = "-" + amount_text[1:-1]
                    discounts.append({
                        "discount": discount_name,
                        "amount": extract_number_from_text(amount_text),
                        "times": times,
                        "date": inv_date,
                        "time": inv_time,
                        "invoice": inv_no,
                    })
                i += 1
                continue

            # ข้ามสรุป/การชำระ/บรรทัดไม่ใช่สินค้า
            if any(k in text_line for k in NON_ITEM_KEYWORDS):
                i += 1
                continue

            handled = False

            # ---- ลองแบบ 2 บรรทัดก่อน (จำนวน+ชื่อ → ราคา) ----
            m_head = PAT_QTY_NAME_ONLY.match(text_line)
            if m_head and (i + 1) < len(body_lines):
                next_line = body_lines[i + 1]
                if not any(k in next_line for k in NON_ITEM_KEYWORDS):
                    m_amt = PAT_AMOUNT_ONLY.match(next_line)
                    if m_amt and _is_plausible_price(m_amt.group("amt")):
                        item_name = m_head.group("name").strip()
                        amount_text = m_amt.group("amt").strip()
                        if amount_text.startswith("(") and amount_text.endswith(")"):
                            amount_text = "-" + amount_text[1:-1]

                        name_compact = (item_name.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789"))
                                                   .replace(",", "").replace(".", "").replace(" ", "")
                                                   .replace("฿","").replace("-",""))
                        if not (name_compact.isdigit() or item_name in {".","","-"}):
                            qty_val = int(m_head.group("qty"))
                            amt_f = extract_number_from_text(amount_text)
                            items.append({
                                "name": item_name,
                                "qty": qty_val,
                                "amount": amt_f,
                                "date": inv_date,
                                "time": inv_time,
                                "invoice": inv_no,
                            })
                            block_items_total += amt_f
                            i += 2
                            handled = True

            if handled:
                continue

            # ---- ลองแบบ 1 บรรทัด (พร้อมตรวจ fallback ไปบรรทัดถัดไปถ้าราคาแปลก) ----
            m = PAT_LINE_ITEM.match(text_line)
            if m:
                item_name = m.group("name").strip()
                amount_text = m.group("amt").strip()

                # ถ้าราคาไม่สมเหตุผล (เช่น 02) → ลองใช้บรรทัดถัดไปเป็นราคา
                if not _is_plausible_price(amount_text) and (i + 1) < len(body_lines):
                    next_line = body_lines[i + 1]
                    if not any(k in next_line for k in NON_ITEM_KEYWORDS):
                        m_amt = PAT_AMOUNT_ONLY.match(next_line)
                        if m_amt and _is_plausible_price(m_amt.group("amt")):
                            amount_text = m_amt.group("amt").strip()
                            i_advance = 2
                        else:
                            i_advance = 1
                    else:
                        i_advance = 1
                else:
                    i_advance = 1

                if amount_text.startswith("(") and amount_text.endswith(")"):
                    amount_text = "-" + amount_text[1:-1]

                name_compact = (item_name.translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙","0123456789"))
                                           .replace(",", "").replace(".", "").replace(" ", "")
                                           .replace("฿","").replace("-",""))
                if not (name_compact.isdigit() or item_name in {".","","-"}):
                    qty_val = int(m.group("qty"))
                    if _is_plausible_price(amount_text):
                        amt_f = extract_number_from_text(amount_text)
                        items.append({
                            "name": item_name,
                            "qty": qty_val,
                            "amount": amt_f,
                            "date": inv_date,
                            "time": inv_time,
                            "invoice": inv_no,
                        })
                        block_items_total += amt_f

                i += i_advance
                continue

            i += 1

        # รวมยอดบิล
        amount_final = extract_number_from_text(price_total) if (price_total and price_total.strip()) else block_items_total
        if amount_final != 0 or inv_no or inv_date or inv_time:
            receipts.append({
                "amount": amount_final,
                "date": inv_date,
                "time": inv_time,
                "invoice": inv_no,
            })

    return pd.DataFrame(receipts), pd.DataFrame(items), pd.DataFrame(discounts)

def summarize_items(df_items: pd.DataFrame) -> pd.DataFrame:
    if df_items.empty:
        return pd.DataFrame(columns=["สินค้า", "จำนวนชิ้น", "ยอดเงิน"])
    return (
        df_items.groupby("name", as_index=False)
        .agg(qty=("qty","sum"), amount=("amount","sum"))
        .rename(columns={"name":"สินค้า","qty":"จำนวนชิ้น","amount":"ยอดเงิน"})
        .sort_values(["จำนวนชิ้น","ยอดเงิน"], ascending=[False, False])
    )

# ==================== HEADER ====================
st.markdown("<h2 style='text-align:center'>Casio V-R100 Tools</h2>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ การตั้งค่า")
    vr100_encoding = st.selectbox("Encoding ไฟล์ SQL", ["UTF-8 (ปกติ)", "UTF-8 with BOM (UTF-8-SIG)"], index=1)
    st.caption("อัตโนมัติ: จับชีท + หัวตาราง + คอลัมน์เอง • ใช้เฉพาะ 'ราคาขาย' + โปรคงที่")
    st.caption("โปร: 3ชิ้น100→50฿, 4ชิ้น100→35฿, 50/2ชิ้น100→80฿ (ต่อชิ้น)")

# ==================== TABS ====================
tab_product, tab_sales = st.tabs(["🏷️ สินค้า (CIA001)", "📊 ยอดขาย (EJ)"])

# ==================== TAB 1: PRODUCT ====================
with tab_product:
    st.markdown("### อัปโหลด Excel/CSV หลายรายการ")
    st.caption("อัตโนมัติ: จับชีท + หัวตาราง + คอลัมน์เอง • ใช้เฉพาะ 'ราคาขาย' + โปรคงที่")

    uploaded_file = st.file_uploader("เลือกไฟล์ Excel หรือ CSV", type=["xlsx", "csv"], key="upload_product")
    if uploaded_file is not None:
        with st.spinner("🔄 กำลังประมวลผลไฟล์..."):
            manual_sheet = None
            if uploaded_file.name.lower().endswith(".xlsx"):
                data_first = uploaded_file.getvalue()
                xls = pd.ExcelFile(BytesIO(data_first))
                with st.expander("🗂️ เลือกชีทเอง (ไม่บังคับ)"):
                    manual_sheet = st.selectbox("ชีทที่ต้องการอ่าน", [None] + xls.sheet_names, index=0, format_func=lambda x: "อัตโนมัติ" if x is None else x)
                uploaded_file = BytesIO(data_first)

            if str(getattr(uploaded_file, "name", "")).lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
                chosen_sheet, header_row = "CSV", 0
            else:
                df_raw, chosen_sheet, header_row = read_excel_smart(uploaded_file, manual_sheet=manual_sheet)

            df_normalized = normalize_uploaded_dataframe(df_raw)

        st.success(f"✅ นำเข้าสำเร็จ {len(df_normalized):,} รายการ • ใช้ชีท: {chosen_sheet} • แถวหัวตาราง: {header_row}")
        with st.expander("👀 ดูข้อมูลที่นำเข้า (30 รายการแรก)", expanded=True):
            st.dataframe(df_normalized.head(30), use_container_width=True, hide_index=True)

        sql_text = build_sql_cia001(df_normalized)
        with st.expander("📄 ดู SQL (ตัวอย่าง 50 บรรทัดแรก)"):
            st.code("\n".join(sql_text.splitlines()[:50]) + "\n...", language="sql")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ ดาวน์โหลด SQL (เลือก Encoding ด้านซ้าย)", export_to_bytes(sql_text, vr100_encoding), file_name="CIA001_bulk_import.sql", mime="text/plain", use_container_width=True)
        with c2:
            st.download_button("📊 Export CSV (สำรองข้อมูล)", export_csv_to_bytes(df_normalized), file_name="CIA001_data_backup.csv", mime="text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("### เพิ่มสินค้าทีละรายการ")
    if "single_item_sql" not in st.session_state:
        st.session_state.single_item_sql = ""

    with st.form("single_item_form"):
        c1, c2 = st.columns(2)
        with c1:
            itemcode = st.text_input("🔢 รหัสสินค้า (SKU)", "")
            itemname = st.text_input("📝 ชื่อสินค้า", "")
        with c2:
            scancode1 = st.text_input("📱 บาร์โค้ด (ชิ้น)", "")
            unitqty   = st.number_input("📦 จำนวนต่อหน่วย", min_value=1, step=1, value=1)
        price_baht = st.text_input("💰 ราคาต่อชิ้น (บาท)", "", placeholder="179.00")
        unitprice  = to_satang(price_baht)
        submitted  = st.form_submit_button("✨ สร้าง SQL", use_container_width=True)

    if submitted:
        row_data = {"ITEMCODE": itemcode, "SCANCODE1": scancode1, "ITEMNAME": itemname, "UNITQTY": unitqty, "UNITPRICE": unitprice, "ITEMPARMCODE": "000001", "UNITWEIGHT": 0, "TAXCODE_1": "01"}
        timestamp = get_casio_timestamp()
        sql = generate_row_sql_cia001(pd.Series(row_data), timestamp)
        st.session_state.single_item_sql = f"BEGIN TRANSACTION;\n{sql}\nCOMMIT;"
        st.success("✅ สร้าง SQL สำเร็จ!")

    if st.session_state.single_item_sql:
        with st.expander("📄 ดู SQL ที่สร้าง", expanded=True):
            st.code(st.session_state.single_item_sql, language="sql")
        st.download_button("⬇️ ดาวน์โหลด SQL (รายการเดียว)", export_to_bytes(st.session_state.single_item_sql, vr100_encoding), file_name="CIA001_single_item.sql", mime="text/plain", use_container_width=True)

# ==================== TAB 2: SALES (EJ) ====================
with tab_sales:
    st.markdown("### วิเคราะห์ยอดขายจากไฟล์ EJ")
    st.caption("อัปโหลด log_YYYYMMDD.txt จากเครื่อง V-R100 (อัปได้หลายไฟล์) — สรุปยอดขายตามบิลและตามสินค้า")

    ej_files = st.file_uploader("เลือกไฟล์ EJ (*.txt)", type=["txt"], accept_multiple_files=True, key="upload_ej_logs")
    if ej_files:
        all_receipts, all_items, all_discounts = [], [], []
        with st.spinner("🔄 กำลังประมวลผลไฟล์..."):
            for file in ej_files:
                text = read_text_with_encoding(file.read())
                receipts, items, disc = parse_ej_text(text)
                if not receipts.empty: all_receipts.append(receipts)
                if not items.empty:    all_items.append(items)
                if not disc.empty:     all_discounts.append(disc)

        df_receipts = pd.concat(all_receipts, ignore_index=True) if all_receipts else pd.DataFrame(columns=["amount","date","time","invoice"]).astype({"amount":"float"})
        df_items    = pd.concat(all_items,    ignore_index=True) if all_items    else pd.DataFrame(columns=["name","qty","amount","date","time","invoice"])
        df_discounts= pd.concat(all_discounts,ignore_index=True) if all_discounts else pd.DataFrame(columns=["discount","amount","times","date","time","invoice"])

        total_receipts = len(df_receipts)
        total_amount = float(df_receipts["amount"].sum()) if total_receipts else float(df_items["amount"].sum())
        total_qty = int(df_items["qty"].sum()) if not df_items.empty else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนบิล (สำเร็จ)", f"{total_receipts:,}")
        c2.metric("จำนวนชิ้น (รวม)", f"{total_qty:,}")
        c3.metric("ยอดขายรวม", f"{total_amount:,.2f}")

        # ---------- ตารางบิลสำหรับหน้า UI ----------
        if not df_receipts.empty:
            df_receipts_pretty = (
                df_receipts.copy()
                .assign(วันที่=lambda d: d["date"].fillna(""),
                        เวลา=lambda d: d["time"].fillna(""),
                        บิล=lambda d: d["invoice"].fillna(""))
                .rename(columns={"amount": "ยอดเงิน"})
                [["บิล","วันที่","เวลา","ยอดเงิน"]]
                .sort_values(["วันที่","เวลา","บิล"])
            )
        else:
            if not df_items.empty:
                def _fix_time(s): return clean_time_token(s) if isinstance(s, str) else s
                df_receipts_pretty = (
                    df_items.groupby(["date","time","invoice"], as_index=False)["amount"].sum()
                    .assign(วันที่=lambda d: d["date"].fillna(""),
                            เวลา=lambda d: d["time"].apply(_fix_time),
                            บิล=lambda d: d["invoice"].fillna(""))
                    .rename(columns={"amount": "ยอดเงิน"})
                    [["บิล","วันที่","เวลา","ยอดเงิน"]]
                    .sort_values(["วันที่","เวลา","บิล"])
                )
            else:
                df_receipts_pretty = pd.DataFrame(columns=["บิล","วันที่","เวลา","ยอดเงิน"])

        df_receipts_display = (
            df_receipts_pretty.assign(**{"วันที่-เวลา": lambda d: d.apply(lambda r: format_datetime_label(r["วันที่"], r["เวลา"]), axis=1)})
            [["บิล","วันที่-เวลา","ยอดเงิน"]]
            .sort_values(["วันที่-เวลา","บิล"])
            .reset_index(drop=True)
        ) if not df_receipts_pretty.empty else pd.DataFrame(columns=["บิล","วันที่-เวลา","ยอดเงิน"])

        with st.expander("🧾 ดูบิลทั้งหมด (มีวัน–เวลา)", expanded=False):
            st.dataframe(df_receipts_display, use_container_width=True, hide_index=True)

        # ---------- รายละเอียดตามบิล สำหรับ Export ----------
        # (1) รวมรายการสินค้า (สุทธิ และกรอง qty=0 ออกในสตริงสินค้า)
        if not df_items.empty:
            items_by_inv = (
                df_items
                .groupby(["invoice","date","time"], as_index=False)
                .agg(items_qty=("qty","sum"), items_amount=("amount","sum"))
            )
            def _items_str(g):
                sums = g.groupby("name")["qty"].sum()
                parts = [f"{name} x{int(q)}" for name, q in sums.items() if int(q) != 0]
                return ", ".join(parts)
            items_name_list = (
                df_items.groupby(["invoice","date","time"]).apply(_items_str).reset_index(name="สินค้า")
            )
            items_by_inv = items_by_inv.merge(items_name_list, on=["invoice","date","time"], how="left")
        else:
            items_by_inv = pd.DataFrame(columns=["invoice","date","time","items_qty","items_amount","สินค้า"])

        # (2) รวมส่วนลดต่อบิล + สตริงส่วนลด
        if not df_discounts.empty:
            disc_by_inv = (
                df_discounts
                .groupby(["invoice","date","time"], as_index=False)
                .agg(discount_times=("times","sum"), discount_amount=("amount","sum"))
            )
            disc_list = (
                df_discounts.groupby(["invoice","date","time"])
                .apply(lambda g: ", ".join(
                    f"{name} x{int(times)}"
                    for name, times in g.groupby("discount")["times"].sum().items()
                ))
                .reset_index(name="ส่วนลด")
            )
            disc_by_inv = disc_by_inv.merge(disc_list, on=["invoice","date","time"], how="left")
        else:
            disc_by_inv = pd.DataFrame(columns=["invoice","date","time","discount_times","discount_amount","ส่วนลด"])

        # (3) ฐานคีย์: ยูเนียน receipts/items/discounts เพื่อไม่ให้บิลตกหล่น
        keys_parts = []
        if not df_receipts.empty:
            keys_parts.append(df_receipts[["invoice","date","time"]])
        if not items_by_inv.empty:
            keys_parts.append(items_by_inv[["invoice","date","time"]])
        if not disc_by_inv.empty:
            keys_parts.append(disc_by_inv[["invoice","date","time"]])
        base_keys = pd.concat(keys_parts, ignore_index=True).drop_duplicates() if keys_parts else pd.DataFrame(columns=["invoice","date","time"])

        # แนบ HPRICE ถ้ามี
        if not df_receipts.empty:
            df_rec_merge = (
                df_receipts.groupby(["invoice","date","time"], as_index=False)
                .agg(hprice_amount=("amount","max"))
            )
            base = base_keys.merge(df_rec_merge, on=["invoice","date","time"], how="left")
        else:
            base = base_keys.copy()
            base["hprice_amount"] = pd.NA

        # แนบ items + discounts
        base = base.merge(items_by_inv, on=["invoice","date","time"], how="left") \
                   .merge(disc_by_inv, on=["invoice","date","time"], how="left") \
                   .fillna({"items_qty":0, "items_amount":0.0, "discount_times":0, "discount_amount":0.0, "สินค้า":"", "ส่วนลด":""})

        # คำนวณยอดบิล (ใช้ HPRICE หากมี)
        bills_summary = (
            base.assign(
                Invoice=lambda d: d["invoice"].astype(str).str.zfill(6),
                Date=lambda d: d["date"].astype(str),
                Time=lambda d: d["time"].astype(str),
                Amount=lambda d: d.apply(
                    lambda r: (r["hprice_amount"] if pd.notna(r["hprice_amount"]) else (float(r["items_amount"]) + float(r["discount_amount"]))),
                    axis=1
                )
            )[
                ["Invoice","Date","Time","สินค้า","ส่วนลด","Amount"]
            ].rename(columns={"Amount":"ยอดเงิน"})
            .sort_values(["Date","Time","Invoice"])
        )

        # (4) Bill Items (แยกรายการ)
        bills_items = (
            df_items
            .assign(
                Invoice=lambda d: d["invoice"].astype(str).str.zfill(6),
                Date=lambda d: d["date"].astype(str),
                Time=lambda d: d["time"].astype(str),
                Item=lambda d: d["name"],
                Qty=lambda d: d["qty"].astype(int),
                Amount=lambda d: d["amount"].astype(float),
            )[["Invoice","Date","Time","Item","Qty","Amount"]]
            .sort_values(["Date","Time","Invoice","Item"])
        )

        # (5) Bill Discounts (แยกรายการ)
        bills_discounts = (
            (df_discounts[["invoice","date","time","discount","times","amount"]]
             if not df_discounts.empty else
             pd.DataFrame(columns=["invoice","date","time","discount","times","amount"]))
            .assign(
                Invoice=lambda d: d["invoice"].astype(str).str.zfill(6) if len(d)>0 else d["invoice"],
                Date=lambda d: d["date"].astype(str) if len(d)>0 else d["date"],
                Time=lambda d: d["time"].astype(str) if len(d)>0 else d["time"],
                DiscountName=lambda d: d["discount"] if len(d)>0 else d["discount"],
                Times=lambda d: d["times"].astype(int) if len(d)>0 else d["times"],
                Amount=lambda d: d["amount"].astype(float) if len(d)>0 else d["amount"],
            )
        )
        if not bills_discounts.empty:
            bills_discounts = bills_discounts[["Invoice","Date","Time","DiscountName","Times","Amount"]] \
                                             .sort_values(["Date","Time","Invoice","DiscountName"])
        else:
            bills_discounts = pd.DataFrame(columns=["Invoice","Date","Time","DiscountName","Times","Amount"])

        # (6) ปุ่มดาวน์โหลด Excel
        st.markdown("#### ⬇️ ดาวน์โหลด Excel — รายละเอียดตามบิล (ทั้งวัน)")
        excel_bytes = export_excel_bills_bytes(bills_summary, bills_items, bills_discounts)
        st.download_button(
            "📥 Export Excel — Bills / Bill Items / Bill Discounts",
            excel_bytes,
            file_name="EJ_bills_detail.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        with st.expander("👀 ตัวอย่างชีท Bills (Top 20)", expanded=False):
            st.dataframe(bills_summary.head(20), use_container_width=True, hide_index=True)

        # ---------- สรุปตามสินค้า ----------
        st.markdown("#### 📦 สรุปยอดตามสินค้า")
        df_summary = summarize_items(df_items)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # ---------- ส่วนลดรวม ----------
        st.markdown("#### 🧾 ส่วนลด/คูปองที่ใช้")
        if df_discounts.empty:
            st.info("ไม่มีการใช้ส่วนลดในไฟล์ที่อัปโหลด")
        else:
            df_discount_summary = (
                df_discounts.groupby("discount", as_index=False).agg({"times": "sum", "amount": "sum"})
                .rename(columns={"discount": "ส่วนลด", "times": "จำนวนครั้ง", "amount": "มูลค่ารวมลด"})
                .sort_values(["จำนวนครั้ง", "มูลค่ารวมลด"], ascending=[False, True])
            )
            st.dataframe(df_discount_summary, use_container_width=True, hide_index=True)

        # ---------- Export สรุปตามสินค้า ----------
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Export CSV — สรุปตามสินค้า", export_csv_to_bytes(df_summary), file_name="EJ_items_summary.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("⬇️ Export Excel — สรุปตามสินค้า", export_excel_to_bytes(df_summary), file_name="EJ_items_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        # ---------- สรุปตามวัน ----------
        if not df_receipts_pretty.empty:
            df_by_date = (
                df_receipts_pretty.groupby("วันที่", as_index=False)
                .agg(จำนวนบิล=("บิล","nunique"), ยอดขายรวม=("ยอดเงิน","sum"))
                .sort_values("วันที่")
            )
            st.markdown("#### 🗓️ สรุปยอดตามวัน")
            st.dataframe(df_by_date, use_container_width=True, hide_index=True)

        # ---------- สรุปตามชั่วโมง ----------
        def _pick_hour(s):
            s = (s or "").strip()
            return s.split(":")[0] if ":" in s else (s[:2] if len(s) >= 2 else "")
        if not df_receipts_pretty.empty:
            df_by_hour = (
                df_receipts_pretty.assign(ชั่วโมง=lambda d: d["เวลา"].apply(_pick_hour))
                .groupby("ชั่วโมง", as_index=False)
                .agg(จำนวนบิล=("บิล","nunique"), ยอดขายรวม=("ยอดเงิน","sum"))
                .sort_values("ชั่วโมง")
            )
            st.markdown("#### ⏰ สรุปยอดตามชั่วโมง")
            st.dataframe(df_by_hour, use_container_width=True, hide_index=True)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("💾 อย่าลืม Restart App หลังนำเข้า SQL • โปร: 3ชิ้น100→50฿, 4ชิ้น100→35฿, 50/2ชิ้น100→80฿ • ITEMPARMCODE=000001 • TAXCODE_1=01")
