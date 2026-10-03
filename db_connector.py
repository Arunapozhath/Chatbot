"""
MySQL Text-to-SQL connector for the vBiz live database.
Only whitelisted tables are exposed to the LLM.
"""
import re
from typing import Optional

from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from sqlalchemy import create_engine, inspect, text

from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASS, MYSQL_DB,
    ALLOWED_TABLES, OLLAMA_BASE_URL, OLLAMA_LLM_MODEL,
)

_engine   = None
_sql_db   = None
_valid_tables = None   # ALLOWED_TABLES filtered to those that actually exist in DB


# ── Table keyword map — picks relevant schemas for the question ────────────
TABLE_KEYWORDS = {
    "invoice":      ["invoice", "invoice_type", "invoice_item_type", "invoice_group"],
    "payment":      ["payment", "payment_type", "payment_method", "payment_group"],
    "facility":     ["facility", "facility_type", "facility_group", "facility_party"],
    "milk":         ["milk_transfer", "milk_transfer_item", "milk_receipt_detail"],
    "transfer":     ["milk_transfer", "milk_transfer_item"],
    "procurement":  ["milk_transfer", "procurement_abstract", "procurement_price"],
    "order":        ["order_header", "order_item", "order_type", "order_status"],
    "employee":     ["employee_detail", "employment", "empl_position"],
    "payroll":      ["payroll_header", "payroll_header_item", "pay_grade"],
    "leave":        ["empl_leave", "empl_leave_type", "empl_leave_balance_status"],
    "attendance":   ["empl_daily_attendance_detail"],
    "loan":         ["loan", "loan_recovery"],
    "salary":       ["payroll_header", "payroll_header_item", "salary_step"],
    "vehicle":      ["vehicle", "vehicle_trip", "vehicle_trip_status"],
    "shipment":     ["shipment", "shipment_item", "shipment_status", "shipment_receipt"],
    "inventory":    ["inventory_item", "inventory_item_type", "inventory_summary"],
    "product":      ["product", "product_type", "product_category", "product_price"],
    "subscription": ["subscription", "subscription_product", "period_billing"],
    "party":        ["party", "party_group", "person", "party_role"],
    "gl":           ["gl_account", "gl_account_type", "acctg_trans_type"],
    "account":      ["gl_account", "fin_account", "fin_account_type"],
    "lien":         ["lien", "lien_item"],
    "gst":          ["gst_category_map", "gst_rate_category"],
    "crate":        ["crate_can_account"],
    "bank":         ["bank_master", "fin_account"],
}


def get_engine():
    global _engine
    if _engine is None:
        url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        _engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
    return _engine


def get_valid_tables() -> list[str]:
    """Return ALLOWED_TABLES filtered to tables that actually exist in the DB."""
    global _valid_tables
    if _valid_tables is None:
        try:
            actual = set(inspect(get_engine()).get_table_names())
            _valid_tables = [t for t in ALLOWED_TABLES if t in actual]
        except Exception:
            _valid_tables = ALLOWED_TABLES
    return _valid_tables


def get_sql_db() -> SQLDatabase:
    global _sql_db
    if _sql_db is None:
        _sql_db = SQLDatabase(
            get_engine(),
            include_tables=get_valid_tables(),
            sample_rows_in_table_info=2,
        )
    return _sql_db


def pick_relevant_tables(question: str, max_tables: int = 5) -> list[str]:
    """Pick the most relevant tables based on keywords in the question."""
    q = question.lower()
    seen   = set()
    picked = []
    for kw, tables in TABLE_KEYWORDS.items():
        if kw in q:
            for t in tables:
                if t not in seen and t in get_valid_tables():
                    seen.add(t)
                    picked.append(t)
    if not picked:
        # fallback: first few valid tables
        picked = get_valid_tables()[:max_tables]
    return picked[:max_tables]


def get_schema_for_tables(tables: list[str]) -> str:
    """Return CREATE TABLE info with sample rows for the given tables."""
    try:
        return get_sql_db().get_table_info(table_names=tables)
    except Exception:
        return ", ".join(tables)


def is_safe_query(sql: str) -> bool:
    upper = sql.upper()
    blocked = [
        "DROP ", "DELETE ", "TRUNCATE ", "UPDATE ", "INSERT ", "ALTER ",
        "CREATE ", "GRANT ", "REVOKE ", "USER_LOGIN", "TOKEN_BLACKLIST",
        "FCM_TOKEN", "SECURITY_GROUP_PERMISSION",
    ]
    return not any(kw in upper for kw in blocked)


def extract_sql(raw: str) -> str:
    match = re.search(r"```sql\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"SELECT\s.+", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return raw.strip()


def run_sql(sql: str, user_party_id: Optional[str] = None) -> dict:
    if not is_safe_query(sql):
        return {"error": "Query blocked for safety reasons."}
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchmany(100)]
        return {"columns": columns, "rows": rows, "count": len(rows)}
    except Exception as e:
        return {"error": str(e)}


SQL_PROMPT = PromptTemplate.from_template(
    """You are a MySQL expert for a dairy cooperative management system.
Write a valid MySQL SELECT query to answer the question below.

Use ONLY these table schemas (with real column names):
{schema}

Rules:
- Write ONLY the SQL query — no explanation, no markdown, no ```
- Use exact column names from the schema above
- Clause order MUST be: SELECT → FROM → WHERE → GROUP BY → ORDER BY → LIMIT
- NEVER put GROUP BY before WHERE
- For date filters: DATE(INVOICE_DATE) = '2026-01-10' or DATE(column) = 'YYYY-MM-DD'
- For counting: SELECT COUNT(*) AS total FROM table WHERE ...
- LIMIT 50 always at the end

Examples:
  Count invoices on a date:
    SELECT COUNT(*) AS total FROM invoice WHERE DATE(INVOICE_DATE) = '2026-01-10'
  Count active facilities:
    SELECT COUNT(*) AS total FROM facility WHERE IS_DEFAULT = 'Y'

Question: {question}

SQL:"""
)


class VBizSQLChain:
    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=OLLAMA_LLM_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0,
            )
        return self._llm

    def query(self, question: str, user_party_id: Optional[str] = None) -> dict:
        try:
            relevant_tables = pick_relevant_tables(question)
            schema          = get_schema_for_tables(relevant_tables)
            prompt          = SQL_PROMPT.format(schema=schema, question=question)
            response        = self._get_llm().invoke(prompt)
            raw_sql         = response.content if hasattr(response, "content") else str(response)
            sql             = extract_sql(raw_sql)
            result          = run_sql(sql, user_party_id)
            return {"sql": sql, **result}
        except Exception as e:
            return {"sql": "", "error": str(e)}

    def reset(self):
        self._llm = None


def format_sql_result(result: dict) -> str:
    if "error" in result:
        return f"Database error: {result['error']}"
    if not result.get("rows"):
        return "No records found for that query."

    cols  = result["columns"]
    rows  = result["rows"]
    count = result["count"]

    lines = []
    if count >= 100:
        lines.append("*(Showing first 100 of many results)*\n")

    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows[:50]:
        vals = [str(row.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    if count > 50:
        lines.append(f"\n*...and {count - 50} more rows*")

    return "\n".join(lines)


def db_is_available() -> bool:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
