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
    "ราคาขาย", "PRICE", "UNIT PRICE", "RETAIL PRICE", "ราคาต่อหน่วย"
}

EJ_ENCODINGS = ["utf-8-sig", "utf-8", "cp874", "tis-620", "utf-16le"]

NON_ITEM_KEYWORDS = (
    "รวม", "ยอดสุทธิ", "เงินสด", "ทอน", "บัตร", "รับชำระ", "ชำระ",
    "ส่วนลด", "คูปอง", "VAT", "ภาษี", "หัวบิล", "ท้ายบิล", "ยกเลิก", "VOID"
)
DISCOUNT_KEYWORDS = (
    "ส่วนลด", "คูปอง", "Coupon", "DISCOUNT", "โปร", "Promotion", "โปรฯ"
)

PAT_LINE_ITEM = re.compile(r"^\s*(?P<qty>\d+)\s+(?P<name>.+?)\s+(?P<amt>-?[\d\.,\(\)]+)\s*$")
PAT_DISCOUNT  = re.compile(r"^\s*(?:(?P<qty>\d+)\s+)?(?P<name>.+?)\s+(?P<amt>-?\(?[\d\.,]+\)?)\s*$")

# ==================== UTILITY FUNCTIONS ====================
def canonicalize_text(text: str) -> str:
    return re.sub(r"[\s_\-\+\.\(\)\[\]\{\}/\\]+", "", text.strip().upper())

def normalize_string(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = "" if value is None else str(value).strip()
    return "" if s.lower() == "nan" else s

def to_int_safe(value, default=0) -> int:
    try:
        x = pd.to_numeric(value, errors="coerce")
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default

def to_satang(value) -> int:
    if value is None:
        return 0
    try:
        if (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
            return 0
    except Exception:
        pass
    s = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return 0
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

# ==================== EXCEL READING (FIXED) ====================
def read_excel_smart(file_obj, manual_sheet: str | None = None) -> tuple[pd.DataFrame, str, int]:
    """
    อ่านไฟล์ Excel แบบฉลาด:
    - สแกน 'ทุกชีท' และ 'หลายแถวแรก' เพื่อหาแถวหัวตารางที่ดีที่สุดตาม CANDIDATE_HEADERS (+โบนัสคำว่า 'ราคา')
    - คืนค่า (DataFrame, ชื่อชีทที่เลือก, แถวหัวตารางที่เลือก)
    - ถ้า manual_sheet ระบุมา จะใช้ชีทนั้น (ยังคงหาแถวหัวตารางที่เหมาะในชีทนั้น)
    """
    data = file_obj.read()
    excel_file = pd.ExcelFile(BytesIO(data))

    target_sheets = [manual_sheet] if manual_sheet else excel_file.sheet_names

    best_sheet = None
    best_row   = 0
    best_score = -1
    candidate_set = {canonicalize_text(h) for h in CANDIDATE_HEADERS}

    for sheet_name in target_sheets:
        df_probe = pd.read_excel(BytesIO(data), sheet_name=sheet_name, header=None, dtype=str)
        limit = min(20, len(df_probe))
        local_best_row = 0
        local_best_score = -1

        for i in range(limit):
            row = [str(x) if pd.notna(x) else "" for x in df_probe.iloc[i].tolist()]
            score = sum(1 for v in row if canonicalize_text(v) in candidate_set)
            # โบนัสถ้ามีคำว่า "ราคา" ในแถวนี้
            if any("ราคา" in str(v) for v in row):
                score += 2
            # โบนัสเล็กน้อยถ้ามีจำนวนคอลัมน์ที่ไม่ว่างเยอะ (มักเป็นหัวตารางจริง)
            non_empty_cols = sum(1 for v in row if str(v).strip() != "")
            score += min(non_empty_cols, 3) * 0.1

            if score > local_best_score:
                local_best_score = score
                local_best_row = i

        if local_best_score > best_score:
            best_score = local_best_score
            best_sheet = sheet_name
            best_row   = local_best_row

    # fallback
    if best_sheet is None:
        best_sheet = target_sheets[0]
        best_row = 0

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
    col_price    = pick_column(["ราคาขาย","PRICE","UNIT PRICE","RETAIL PRICE","ราคาต่อหน่วย"])
    col_unitqty  = pick_column(["UNITQTY","QTY","PACK","ชิ้นต่อแพ็ค","รวมชิ้นต่อแพ็ค","หน่วยต่อแพ็ค"])

    out = pd.DataFrame()
    out["ITEMCODE"]  = df_raw[col_itemcode] if col_itemcode else ""
    out["ITEMNAME"]  = df_raw[col_itemname] if col_itemname else ""
    out["SCANCODE1"] = df_raw[col_barcode]  if col_barcode  else ""
    out["UNITQTY"]   = pd.to_numeric(df_raw[col_unitqty], errors="coerce").fillna(1).astype(int) if col_unitqty else 1

    # ราคา (เลขล้วน)
    if col_price:
        raw_price = df_raw[col_price].astype(str).str.strip()
        is_numeric = raw_price.str.fullmatch(r"[+-]?\d+(?:[.,]\d+)?")
        base_baht = pd.to_numeric(
            raw_price.str.replace(",", "", regex=False).str.replace("฿", "", regex=False),
            errors="coerce"
        ).where(is_numeric)
    else:
        base_baht = pd.Series([None]*len(df_raw), index=df_raw.index, dtype="float")
    out["UNITPRICE"] = base_baht

    # โปรข้อความคงที่
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
    out.loc[override.notna(), "UNITPRICE"] = override[override.notna()].astype(float)

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

def parse_ej_text(text: str):
    """Parse EJ text and return (receipts, items, discounts)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    receipts = []
    items = []
    discounts = []

    blocks = re.split(r"\n(?=S\n)", "\n" + text)
    for block in blocks:
        if not block.strip().startswith("S\n"):
            continue

        mode = None
        price_total = None
        canceled = False
        body_lines = []

        for raw_line in block.splitlines():
            if raw_line.startswith("HMODE="):
                mode = raw_line.split("=", 1)[1].strip()
            elif raw_line.startswith("HPRICE="):
                price_total = raw_line.split("=", 1)[1].strip()
            elif raw_line.startswith("B"):
                text_line = raw_line[1:].strip()
                if any(keyword in text_line for keyword in ("ยกเลิก","VOID","Cancel","CANCEL")):
                    canceled = True
                body_lines.append(text_line)

        if mode not in (None, "REG", "REG "):
            continue
        if canceled:
            continue

        for text_line in body_lines:
            if any(keyword in text_line for keyword in DISCOUNT_KEYWORDS):
                m2 = PAT_DISCOUNT.match(text_line)
                if m2:
                    qty_txt = m2.group("qty")
                    times = int(qty_txt) if qty_txt else 1
                    discount_name = m2.group("name").strip()
                    amount_text = m2.group("amt").strip()
                    if amount_text.startswith("(") and amount_text.endswith(")"):
                        amount_text = "-" + amount_text[1:-1]
                    discount_amount = extract_number_from_text(amount_text)
                    discounts.append({"discount": discount_name, "amount": discount_amount, "times": times})
                continue

            if any(keyword in text_line for keyword in NON_ITEM_KEYWORDS):
                continue

            m = PAT_LINE_ITEM.match(text_line)
            if not m:
                continue
            item_name = m.group("name").strip()
            quantity = int(m.group("qty"))
            amount_text = m.group("amt").strip()
            if amount_text.startswith("(") and amount_text.endswith(")"):
                amount_text = "-" + amount_text[1:-1]
            amount = extract_number_from_text(amount_text)
            items.append({"name": item_name, "qty": quantity, "amount": amount})

        if price_total and price_total.strip():
            receipts.append({"amount": extract_number_from_text(price_total)})

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
                # ให้ผู้ใช้ override ชีทได้ถ้าต้องการ
                data_first = uploaded_file.getvalue()
                xls = pd.ExcelFile(BytesIO(data_first))
                with st.expander("🗂️ เลือกชีทเอง (ไม่บังคับ)"):
                    manual_sheet = st.selectbox("ชีทที่ต้องการอ่าน", [None] + xls.sheet_names, index=0, format_func=lambda x: "อัตโนมัติ" if x is None else x)
                # ต้องสร้างไฟล์ใหม่จาก bytes เพราะ file_uploader ถูกอ่านไปแล้ว
                uploaded_file = BytesIO(data_first)

            if str(getattr(uploaded_file, "name", "")).lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
                chosen_sheet = "CSV"
                header_row = 0
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

        df_receipts = pd.concat(all_receipts, ignore_index=True) if all_receipts else pd.DataFrame(columns=["amount"]).astype({"amount":"float"})
        df_items    = pd.concat(all_items,    ignore_index=True) if all_items    else pd.DataFrame(columns=["name","qty","amount"])
        df_discounts= pd.concat(all_discounts,ignore_index=True) if all_discounts else pd.DataFrame(columns=["discount","amount","times"])

        total_receipts = len(df_receipts)
        total_amount = float(df_receipts["amount"].sum()) if total_receipts else float(df_items["amount"].sum())
        total_qty = int(df_items["qty"].sum()) if not df_items.empty else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนบิล (สำเร็จ)", f"{total_receipts:,}")
        c2.metric("จำนวนชิ้น (รวม)", f"{total_qty:,}")
        c3.metric("ยอดขายรวม", f"{total_amount:,.2f}")

        st.markdown("#### 📦 สรุปยอดตามสินค้า")
        df_summary = summarize_items(df_items)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.markdown("#### 🧾 ส่วนลด/คูปองที่ใช้")
        if df_discounts.empty:
            st.info("ไม่มีการใช้ส่วนลดในไฟล์ที่อัปโหลด")
        else:
            df_discount_summary = (
                df_discounts
                .groupby("discount", as_index=False)
                .agg({"times": "sum", "amount": "sum"})
            )
            df_discount_summary = (
                df_discount_summary
                .rename(columns={
                    "discount": "ส่วนลด",
                    "times": "จำนวนครั้ง",
                    "amount": "มูลค่ารวมลด",
                })
                .sort_values(["จำนวนครั้ง", "มูลค่ารวมลด"], ascending=[False, True])
            )
            st.dataframe(df_discount_summary, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Export CSV — สรุปตามสินค้า", export_csv_to_bytes(df_summary), file_name="EJ_items_summary.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("⬇️ Export Excel — สรุปตามสินค้า", export_excel_to_bytes(df_summary), file_name="EJ_items_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        if not df_discounts.empty:
            c3, c4 = st.columns(2)
            with c3:
                st.download_button("⬇️ Export CSV — ส่วนลด/คูปอง", export_csv_to_bytes(df_discount_summary), file_name="EJ_discounts_summary.csv", mime="text/csv", use_container_width=True)
            with c4:
                st.download_button("⬇️ Export Excel — ส่วนลด/คูปอง", export_excel_to_bytes(df_discount_summary, sheet_name="ส่วนลด"), file_name="EJ_discounts_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("💾 อย่าลืม Restart App หลังนำเข้า SQL • โปร: 3ชิ้น100→50฿, 4ชิ้น100→35฿, 50/2ชิ้น100→80฿ • ITEMPARMCODE=000001 • TAXCODE_1=01")
