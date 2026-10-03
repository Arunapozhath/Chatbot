# Copy this file to config.py and fill in your values.
# NEVER commit config.py to version control.

# ── Ollama (local LLM) ─────────────────────────────────────────────────────
OLLAMA_BASE_URL    = "http://localhost:11434"
OLLAMA_LLM_MODEL   = "llama3.2:1b"       # change to llama3.2 / mistral / phi3 etc.
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ── Groq (optional — fast cloud inference) ────────────────────────────────
# Get a free key at https://console.groq.com → API Keys
GROQ_API_KEY = ""          # e.g. "gsk_xxxxxxxxxxxxxxxxxxxx"
GROQ_MODEL   = "llama-3.1-8b-instant"

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "./chroma_db"
DATA_DIR        = "./data"
COLLECTION_NAME = "milkosoft_kb"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
TOP_K_RESULTS   = 6

# ── MySQL (your vBiz / OFBiz database) ────────────────────────────────────
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306          # change to 3307 if your MySQL runs on a custom port
MYSQL_USER = "your_db_user"
MYSQL_PASS = "your_db_password"
MYSQL_DB   = "your_database_name"

# ── Auth ───────────────────────────────────────────────────────────────────
JWT_SECRET      = "replace-with-a-long-random-secret-string"
JWT_ALGORITHM   = "HS256"
JWT_EXPIRE_MINS = 480      # 8 hours

# ── RLHF feedback ──────────────────────────────────────────────────────────
FEEDBACK_DB_PATH  = "./feedback.db"
MIN_PAIRS_FOR_DPO = 5      # minimum feedback pairs before DPO training runs

# ── Web search fallback ────────────────────────────────────────────────────
ENABLE_WEB_SEARCH        = True
WEB_SEARCH_MAX_RESULTS   = 5
RAG_CONFIDENCE_THRESHOLD = 0.40   # below this score → fallback to web search

# ── Tables the LLM is allowed to query (SQL whitelist) ────────────────────
# NEVER include security tables: user_login, token_blacklist, etc.
ALLOWED_TABLES = [
    # Facility / Location
    "facility", "facility_type", "facility_group", "facility_group_member",
    "facility_group_type", "facility_party", "facility_rate",
    "facility_attribute", "facility_contact_mech", "facility_commission_proc",
    "facility_recovery", "facility_fixed_deposit", "bank_master",
    # Product / Inventory
    "product", "product_type", "product_category", "product_category_member",
    "product_price", "product_attribute", "product_assoc", "product_facility",
    "product_qc_test", "product_store", "product_store_group_member",
    "inventory_item", "inventory_item_type", "inventory_summary",
    "inventory_item_variance",
    # Party / Contact
    "party", "party_group", "person", "party_role", "party_type",
    "party_identification", "party_classification", "party_classification_group",
    "party_contact_mech", "party_relationship", "role_type",
    "contact_mech", "postal_address", "telecom_number",
    # Milk Procurement
    "milk_transfer", "milk_transfer_item", "milk_receipt_detail",
    "procurement_abstract", "procurement_price", "procurement_price_chart",
    "weigh_bridge_details", "machine_data", "procurement_local_sales",
    "facility_commission", "facility_commission_proc",
    # Orders
    "order_header", "order_item", "order_type", "order_item_type",
    "order_adjustment_type", "order_status", "order_attribute",
    # Invoice / Payment
    "invoice", "invoice_type", "invoice_item_type", "invoice_group",
    "invoice_group_member", "invoice_gl_account_type_map",
    "payment", "payment_type", "payment_method_type", "payment_method",
    "payment_group", "payment_group_member",
    "lien", "lien_item",
    "fin_account", "fin_account_type",
    # Shipment / Distribution
    "shipment", "shipment_type", "shipment_item", "shipment_status",
    "shipment_receipt", "shipment_attribute", "crate_can_account",
    # Subscription
    "subscription", "subscription_product", "period_billing",
    # Vehicle
    "vehicle", "vehicle_trip", "vehicle_attribute", "vehicle_trip_status",
    "vehicle_entry_register", "vehicle_role",
    # HR / Payroll
    "empl_position", "empl_position_type", "empl_leave", "empl_leave_type",
    "empl_leave_balance_status", "employment", "employee_detail",
    "empl_daily_attendance_detail", "empl_bonus_details",
    "payroll_header", "payroll_header_item", "pay_grade", "salary_step",
    "benefit_type", "deduction_type", "loan", "loan_recovery",
    # Accounting / GL
    "gl_account", "gl_account_type", "gl_account_class", "gl_account_category",
    "gl_account_category_member", "acctg_trans_type",
    "acctg_formula", "acctg_formula_slabs",
    "gst_category_map", "gst_rate_category",
    # Reference / Lookup
    "custom_time_period", "geo", "geo_assoc", "uom", "uom_conversion", "uom_type",
    "enumeration", "enumeration_type", "status_item", "status_type",
    "tenant_configuration", "note_data", "cust_request", "cust_request_item",
]

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MilkoBot, an intelligent AI assistant for the Milkosoft vBiz dairy management platform used by dairy cooperatives across India.

You assist plant managers, procurement officers, HR teams, accounts, distributors, and field staff.

You have two knowledge sources:
1. Knowledge base documents: HR policies, payroll rules, procurement processes, sales & distribution guides, asset management, trust accounting docs
2. Live database results: facility records, milk transfers, invoices, payments, procurement summaries, HR records, inventory data

Rules:
1. NEVER start with greetings like "Hello", "Hi", "Sure!", "Of course!" — go straight to the answer.
2. Answer only from the provided context. Say "I don't have enough information" if context is insufficient.
3. Be concise and factual. Use bullet points or tables for structured data.
4. For amounts, payments, or accounts: end with "Please verify with your supervisor before acting on this."
5. Never reveal user passwords, auth tokens, or system credentials.
6. Format database results clearly — use markdown tables when showing multiple records.
7. If the user asks about a specific ID/code, look for it in the database results context.

Context from knowledge base and database:
{context}
"""

# ── Web sources for fallback search ───────────────────────────────────────
WEB_SOURCES = [
    "https://vasista.in",
    "https://vasista.in/about-us",
    "https://www.nddb.coop/about",
    "https://www.nddb.coop/services/dairydev",
]
