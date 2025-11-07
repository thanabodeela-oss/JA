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
    "No", "คน", "Qty change"
)
DISCOUNT_KEYWORDS = ("ส่วนลด", "คูปอง", "Coupon", "DISCOUNT", "โปร", "Promotion", "โปรฯ")

# ---------- Regex ----------
PAT_LINE_ITEM      = re.compile(r"^\s*(?P<qty>[+-]?\d+)\s+(?P<name>.+?)\s+(?P<amt>-?[\d\.,\(\)]+)\s*$")
PAT_DISCOUNT       = re.compile(r"^\s*(?:(?P<qty>[+-]?\d+)\s+)?(?P<name>.+?)\s+(?P<amt>-?\(?[\d\.,]+\)?)\s*$")
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
                             df_bills_discounts: pd.DataFrame,
                             df_manager: pd.DataFrame | None = None) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_bills_summary.to_excel(writer, index=False, sheet_name="Bills")
        df_bills_items.to_excel(writer, index=False, sheet_name="Bill Items")
        df_bills_discounts.to_excel(writer, index=False, sheet_name="Bill Discounts")
        if df_manager is not None and len(df_manager) > 0:
            df_manager.to_excel(writer, index=False, sheet_name="Manager Adjustments")
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
    if len(s) == 3:
        return f"0{s[0]}:{s[1:]}"
    if len(s) == 4:
        return f"{s[:2]}:{s[2:]}"
    if len(s) == 6:
        return f"{s[:2]}:{s[2:4]}:{s[4:]}"
    return str(tok).strip()

def clean_date_token(tok: str | None) -> str:
    if not tok: return ""
    s = str(tok).strip()
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}"
    return s

def format_datetime_label(d: str, t: str) -> str:
    t = clean_time_token(t or "")
    hhmm = t[:5] if ":" in t else (t[:2] + ":" + t[2:4] if len(t) >= 4 else t)
    return (d or "").strip() + (" " + hhmm if hhmm else "")

def _is_plausible_price(raw: str) -> bool:
    v = abs(extract_number_from_text(raw))
    return (v >= 5) or ("." in raw) or ("(" in raw and ")" in raw)

def parse_ej_text(text: str):
    """Parse EJ -> (receipts_df, items_df, discounts_df, manager_df).
       - รองรับบิลผู้จัดการ
       - เลือกยอดสุทธิจาก HPRICE / คำว่า 'รวมเงินทั้งหมด|ยอดสุทธิ|สุทธิ'
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    receipts, items, discounts, managers = [], [], [], []

    pat_b_header = re.compile(r"^\s*(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s+(?P<inv>\d{3,})\s*$")
    re_consec   = re.compile(r"Consec\s*(?:Number|No\.?)\s*[:\-]?\s*(?P<no>\d+)", re.I)
    re_sum_qty  = re.compile(r"ยอดขายรวม\s*จำนวน\s*(?P<qty>\d+)")
    re_sum_amt  = re.compile(r"ยอดขายรวม\s*รวมเงิน\s*(?P<amt>[\d\.,\(\)\-]+)")
    re_manager  = re.compile(r"\bmanager\b", re.I)
    re_total    = re.compile(r"(รวมเงินทั้งหมด|ยอดสุทธิ|สุทธิ)\s+(?P<amt>[\d\.,]+)")

    blocks = re.split(r"\n(?=S\n)", "\n" + text)
    for block in blocks:
        if not block.strip().startswith("S\n"):
            continue

        mode, price_total, canceled = None, None, False
        body_lines = []
        inv_date_raw = inv_time_raw = inv_no = None
        block_items_total = 0.0
        last_total_text = None

        is_manager = False
        consec_link = None
        mgr_qty = None
        mgr_amt_text = None

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
                if re_manager.search(text_line): is_manager = True
                mcs = re_consec.search(text_line)
                if mcs: consec_link = mcs.group("no").zfill(6)
                mqty = re_sum_qty.search(text_line)
                if mqty:
                    try: mgr_qty = int(mqty.group("qty"))
                    except: pass
                mamt = re_sum_amt.search(text_line)
                if mamt: mgr_amt_text = mamt.group("amt").strip()

                mt = re_total.search(text_line)
                if mt: last_total_text = mt.group("amt").strip()

                if inv_no is None:
                    mhead = pat_b_header.match(text_line)
                    if mhead:
                        inv_date_raw = mhead.group("date")
                        inv_time_raw = mhead.group("time")
                        inv_no       = mhead.group("inv")
                if any(k in text_line for k in ("ยกเลิก","VOID","Cancel","CANCEL")):
                    canceled = True
                body_lines.append(text_line)

        inv_date = clean_date_token(inv_date_raw) if inv_date_raw else ""
        inv_time = clean_time_token(inv_time_raw) if inv_time_raw else ""

        # Manager slip -> หักลบให้บิลต้นทาง
        if is_manager and consec_link and mgr_amt_text:
            amt = extract_number_from_text("-" + mgr_amt_text[1:-1] if mgr_amt_text.startswith("(") and mgr_amt_text.endswith(")") else mgr_amt_text)
            if amt != 0:
                managers.append({
                    "ManagerInvoice": str(inv_no).zfill(6) if inv_no else "",
                    "Date": inv_date,
                    "Time": inv_time,
                    "LinkedInvoice": str(consec_link).zfill(6),
                    "Count": mgr_qty if mgr_qty is not None else 0,
                    "Amount": abs(amt),
                    "ToInvoiceAmount": abs(amt)
                })
                receipts.append({
                    "amount": -abs(amt),
                    "date": inv_date,
                    "time": inv_time,
                    "invoice": consec_link,
                    "src": "mgr"
                })
            continue

        if mode not in (None, "REG", "REG "):
            continue
        if canceled:
            continue

        # ---------- parse normal receipt body ----------
        i = 0
        while i < len(body_lines):
            text_line = body_lines[i]

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

            if any(k in text_line for k in NON_ITEM_KEYWORDS):
                i += 1
                continue

            handled = False
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

            m = PAT_LINE_ITEM.match(text_line)
            if m:
                item_name = m.group("name").strip()
                amount_text = m.group("amt").strip()
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

        # --- เลือกยอดสุทธิของบิล ---
        if price_total and price_total.strip():
            amount_final = extract_number_from_text(price_total)
        elif last_total_text:
            amount_final = extract_number_from_text(last_total_text)
        else:
            amount_final = block_items_total

        if amount_final != 0 or inv_no or inv_date or inv_time:
            receipts.append({
                "amount": amount_final,
                "date": inv_date,
                "time": inv_time,
                "invoice": inv_no,
                "src": "sale"
            })

    return pd.DataFrame(receipts), pd.DataFrame(items), pd.DataFrame(discounts), pd.DataFrame(managers)

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
        all_receipts, all_items, all_discounts, all_mgr = [], [], [], []
        with st.spinner("🔄 กำลังประมวลผลไฟล์..."):
            for file in ej_files:
                text = read_text_with_encoding(file.read())
                receipts, items, disc, mgr = parse_ej_text(text)
                if not receipts.empty: all_receipts.append(receipts)
                if not items.empty:    all_items.append(items)
                if not disc.empty:     all_discounts.append(disc)
                if not mgr.empty:      all_mgr.append(mgr)

        df_receipts_raw = pd.concat(all_receipts, ignore_index=True) if all_receipts else pd.DataFrame(columns=["amount","date","time","invoice","src"]).astype({"amount":"float"})
        df_items        = pd.concat(all_items,    ignore_index=True) if all_items    else pd.DataFrame(columns=["name","qty","amount","date","time","invoice"])
        df_discounts    = pd.concat(all_discounts,ignore_index=True) if all_discounts else pd.DataFrame(columns=["discount","amount","times","date","time","invoice"])
        df_manager      = pd.concat(all_mgr,      ignore_index=True) if all_mgr      else pd.DataFrame(columns=["ManagerInvoice","Date","Time","LinkedInvoice","Count","Amount","ToInvoiceAmount"])

        # ---------- map date per invoice (กันเลขบิลซ้ำข้ามวัน) ----------
        date_from_items = df_items.groupby("invoice")["date"].min() if not df_items.empty else pd.Series(dtype="object")
        date_from_disc  = df_discounts.groupby("invoice")["date"].min() if not df_discounts.empty else pd.Series(dtype="object")
        date_from_sale  = df_receipts_raw[df_receipts_raw.get("src","sale")=="sale"].groupby("invoice")["date"].min() if not df_receipts_raw.empty else pd.Series(dtype="object")

        def pick_date(inv, original):
            for s in (date_from_items, date_from_disc, date_from_sale):
                if inv in s.index:
                    val = s.loc[inv]
                    if isinstance(val, pd.Series): val = val.iloc[0]
                    if pd.notna(val) and str(val).strip() != "":
                        return val
            return original

        if not df_receipts_raw.empty:
            df_receipts_raw = df_receipts_raw.assign(date_key=lambda d: d.apply(lambda r: pick_date(r["invoice"], r["date"]), axis=1))
        else:
            df_receipts_raw = pd.DataFrame(columns=["amount","date","time","invoice","src","date_key"])

        # ✅ รวมใบเสร็จต่อ (วันที่+บิล)
        df_receipts = (
            df_receipts_raw
            .groupby(["invoice","date_key"], as_index=False)
            .agg(amount=("amount","sum"),
                 date=("date_key","min"),
                 time=("time","min"))
        )

        total_receipts = (df_receipts[["invoice","date"]].drop_duplicates().shape[0] if not df_receipts.empty else 0)
        total_amount = float(df_receipts["amount"].sum()) if total_receipts else float(df_items["amount"].sum() if not df_items.empty else 0.0)
        total_qty = int(df_items["qty"].sum()) if not df_items.empty else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนบิล (สำเร็จ)", f"{total_receipts:,}")
        c2.metric("จำนวนชิ้น (รวม)", f"{total_qty:,}")
        c3.metric("ยอดขายรวม", f"{total_amount:,.2f}")

        # ---------- ตารางบิลสำหรับหน้า UI ----------
        df_receipts_pretty = (
            df_receipts.copy()
            .assign(วันที่=lambda d: d["date"].fillna(""),
                    เวลา=lambda d: d["time"].fillna(""),
                    บิล=lambda d: d["invoice"].fillna(""))
            .rename(columns={"amount": "ยอดเงิน"})
            [["บิล","วันที่","เวลา","ยอดเงิน"]]
            .sort_values(["วันที่","เวลา","บิล"])
        ) if not df_receipts.empty else pd.DataFrame(columns=["บิล","วันที่","เวลา","ยอดเงิน"])

        df_receipts_display = (
            df_receipts_pretty.assign(**{"วันที่-เวลา": lambda d: d.apply(lambda r: format_datetime_label(r["วันที่"], r["เวลา"]), axis=1)})
            [["บิล","วันที่-เวลา","ยอดเงิน"]]
            .sort_values(["วันที่-เวลา","บิล"])
            .reset_index(drop=True)
        ) if not df_receipts_pretty.empty else pd.DataFrame(columns=["บิล","วันที่-เวลา","ยอดเงิน"])

        with st.expander("🧾 ดูบิลทั้งหมด (มีวัน–เวลา)", expanded=False):
            st.dataframe(df_receipts_display, use_container_width=True, hide_index=True)

        # ---------- รายละเอียดตามบิลสำหรับ Export ----------
        # (1) สินค้าต่อ (วันที่+บิล)
        if not df_items.empty:
    items_by_inv = (
        df_items
        .groupby(["invoice","date"], as_index=False)
        .agg(Time=("time","min"),
             items_qty=("qty","sum"),
             items_amount=("amount","sum"))
    )

    # รวมจำนวนรายชื่อสินค้าแล้ว "กรองรายการที่จำนวนสุทธิ == 0"
    name_agg = (
        df_items.groupby(["invoice","date","name"])["qty"]
        .sum()
        .reset_index()
    )
    # บางที dtype จะเป็น float จากการรวม ให้ปัดเป็น int แล้วกรอง 0
    name_agg["qty"] = name_agg["qty"].astype(int)
    name_agg = name_agg[name_agg["qty"] != 0]

    name_agg["txt"] = name_agg.apply(lambda r: f"{r['name']} x{int(r['qty'])}", axis=1)
    items_name_list = (
        name_agg.groupby(["invoice","date"])["txt"]
        .apply(lambda s: ", ".join(s.tolist()))
        .reset_index(name="สินค้า")
    )

    items_by_inv = items_by_inv.merge(items_name_list, on=["invoice","date"], how="left")
else:
    items_by_inv = pd.DataFrame(columns=["invoice","date","Time","items_qty","items_amount","สินค้า"])
        # (2) ส่วนลดต่อ (วันที่+บิล)
        if not df_discounts.empty:
            disc_by_inv = (
                df_discounts
                .groupby(["invoice","date"], as_index=False)
                .agg(Time_d=("time","min"),
                     discount_times=("times","sum"),
                     discount_amount=("amount","sum"))
            )
            disc_base = df_discounts.groupby(["invoice","date","discount"])["times"].sum().reset_index()
            disc_base["txt"] = disc_base.apply(lambda r: f"{r['discount']} x{int(r['times'])}", axis=1)
            disc_names = disc_base.groupby(["invoice","date"])["txt"].apply(lambda s: ", ".join(s.tolist())).reset_index(name="ส่วนลด")
            disc_by_inv = disc_by_inv.merge(disc_names, on=["invoice","date"], how="left")
        else:
            disc_by_inv = pd.DataFrame(columns=["invoice","date","Time_d","discount_times","discount_amount","ส่วนลด"])

        # (3) รวมฐานด้วย (วันที่+บิล) — เปลี่ยนชื่อจาก receipts เป็น Date_r/Time_r กันชนกัน
        base_keys = (
            pd.concat([
                df_receipts[["invoice","date"]],
                items_by_inv[["invoice","date"]],
                disc_by_inv[["invoice","date"]],
            ], ignore_index=True).drop_duplicates()
        )

        receipts_for_merge = df_receipts.rename(columns={"invoice":"Invoice","date":"Date_r","time":"Time_r"})
        base = (
            base_keys
            .merge(receipts_for_merge, left_on=["invoice","date"], right_on=["Invoice","Date_r"], how="left")
            .merge(items_by_inv, on=["invoice","date"], how="left")
            .merge(disc_by_inv, on=["invoice","date"], how="left")
        )

        # เวลา: เลือกดีที่สุดจาก items > receipts > discounts
        def _pick_time(r):
            for k in ("Time", "Time_r", "Time_d"):
                v = r.get(k, "")
                if pd.notna(v) and str(v).strip() != "":
                    return v
            return ""

        base["TimeFinal"] = base.apply(_pick_time, axis=1)

        # ทำความสะอาดคอลัมน์ซ้ำ
        base = base.fillna({
            "สินค้า":"", "ส่วนลด":"", "items_qty":0, "items_amount":0.0,
            "discount_times":0, "discount_amount":0.0, "amount":0.0
        })
        base["Invoice"] = base["Invoice"].fillna(base["invoice"].astype(str))
        base = base.rename(columns={"date":"Date"})
        # ลบทิ้งคอลัมน์ชั่วคราวกันชื่อซ้ำ
        drop_cols = [c for c in ["Date_r","Time_r","Time","Time_d","invoice"] if c in base.columns]
        tmp = base.drop(columns=drop_cols, errors="ignore")

        bills_summary = (
            tmp.assign(Invoice=lambda d: d["Invoice"].astype(str).str.zfill(6),
                       Time=lambda d: d["TimeFinal"])
               .rename(columns={"amount":"ยอดเงิน"})
               [["Invoice","Date","Time","สินค้า","ส่วนลด","ยอดเงิน"]]
               .sort_values(["Date","Time","Invoice"])
        )

        # (4) Bill Items
        bills_items = (
            df_items
            .assign(
                Invoice=lambda d: d["invoice"].astype(str).str.zfill(6),
                Date=lambda d: d["date"],
                Time=lambda d: d["time"],
                Item=lambda d: d["name"],
                Qty=lambda d: d["qty"].astype(int),
                Amount=lambda d: d["amount"].astype(float),
            )[["Invoice","Date","Time","Item","Qty","Amount"]]
            .sort_values(["Date","Time","Invoice","Item"])
        )

        # (5) Bill Discounts
        bills_discounts = (
            (df_discounts[["invoice","date","time","discount","times","amount"]]
             if not df_discounts.empty else
             pd.DataFrame(columns=["invoice","date","time","discount","times","amount"]))
            .assign(
                Invoice=lambda d: d["invoice"].astype(str).str.zfill(6) if len(d)>0 else d["invoice"],
                Date=lambda d: d["date"] if len(d)>0 else d["date"],
                Time=lambda d: d["time"] if len(d)>0 else d["time"],
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

        # ---------- Manager Adjustments ----------
        manager_view = pd.DataFrame(columns=["ManagerInvoice","Date","Time","LinkedInvoice","Count","Amount","ToInvoiceAmount"])
        if not df_manager.empty:
            manager_view = (
                df_manager.assign(
                    ManagerInvoice=lambda d: d["ManagerInvoice"].astype(str).str.zfill(6),
                    LinkedInvoice=lambda d: d["LinkedInvoice"].astype(str).str.zfill(6)
                )
                .sort_values(["Date","Time","ManagerInvoice"])
            )

        # ---------- Export ----------
        st.markdown("#### ⬇️ ดาวน์โหลด Excel — รายละเอียดตามบิล (ทั้งวัน)")
        excel_bytes = export_excel_bills_bytes(bills_summary, bills_items, bills_discounts, manager_view)
        st.download_button(
            "📥 Export Excel — Bills / Bill Items / Bill Discounts / Manager Adjustments",
            excel_bytes,
            file_name="EJ_bills_detail.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        with st.expander("👀 ตัวอย่างชีท Bills (Top 20)", expanded=False):
            st.dataframe(bills_summary.head(20), use_container_width=True, hide_index=True)

        with st.expander("🧯 บิลผู้จัดการ (สำหรับหักลบยอด)"):
            if manager_view.empty:
                st.info("ไม่พบบิลผู้จัดการ")
            else:
                focus = manager_view[manager_view["ManagerInvoice"].isin(["000066","000342"])]
                if not focus.empty:
                    st.markdown("**รายการที่คุณถามหา (000066, 000342):**")
                    st.dataframe(focus, use_container_width=True, hide_index=True)
                st.markdown("**ทั้งหมด:**")
                st.dataframe(manager_view, use_container_width=True, hide_index=True)

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
