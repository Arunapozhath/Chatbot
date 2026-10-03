"""
Route a user question to the right knowledge source:
  - "rag"    → ChromaDB document search
  - "sql"    → MySQL Text-to-SQL
  - "hybrid" → Both sources
  - "web"    → DuckDuckGo fallback (handled in rag_pipeline)
"""
import re
from typing import Literal

QuerySource = Literal["rag", "sql", "hybrid"]

# Keywords that strongly suggest a database/data query
SQL_SIGNALS = [
    r"\bhow many\b", r"\bcount\b", r"\blist all\b", r"\bshow me\b",
    r"\bshow all\b", r"\bget all\b", r"\bfind all\b", r"\bfetch\b",
    r"\btotal\b", r"\bsum\b", r"\baverage\b", r"\bmax\b", r"\bmin\b",
    r"\bwhat is the status\b", r"\bcurrent status\b",
    r"\brecords?\b", r"\bentries?\b", r"\bdata for\b",
    r"\binvoices?\b", r"\bpayments?\b", r"\borders?\b",
    r"\bmilk transfer\b", r"\bprocurement\b.*\bsociety\b",
    r"\bfacility\b.*\bcode\b", r"\bparty id\b", r"\bparty code\b",
    r"\bsubscription\b.*\bactive\b", r"\bvehicle trip\b",
    r"\bpending\b.*\binvoice\b", r"\bdue\b.*\bpayment\b",
    r"\bstock\b", r"\binventory\b.*\bquantity\b",
    r"\bpayroll\b.*\bmonth\b", r"\bsalary\b.*\bof\b",
    r"\bleave balance\b", r"\battendance\b.*\bdate\b",
    r"\bGL account\b", r"\bjournal entry\b",
]

# Keywords that suggest a document/policy question
RAG_SIGNALS = [
    r"\bhow does\b", r"\bwhat is the process\b", r"\bexplain\b",
    r"\bwhat are the rules\b", r"\bpolicy\b", r"\bprocedure\b",
    r"\bsteps?\b", r"\bguide\b", r"\bdocument\b",
    r"\bhow to\b", r"\bwhat should\b", r"\bwhen should\b",
    r"\beligib\b", r"\bcalcul\b", r"\bformula\b",
    r"\bapproval\b.*\bprocess\b", r"\bleave rules\b",
    r"\bpayroll process\b", r"\bHR policy\b",
    r"\bwhat is\b.*\bmodule\b", r"\bfeature\b",
    r"\bconfigur\b", r"\bsetup\b", r"\bwhat does\b.*\bmean\b",
    r"\bEPF\b", r"\bESI\b", r"\bPF\b", r"\bbonus rule\b",
    r"\bdeduction\b.*\brule\b", r"\bbenefit\b.*\btype\b",
    r"\bcattle\b.*\bfeed\b", r"\basset\b.*\bmodule\b",
    r"\btrust\b.*\baccounting\b",
]


def classify(question: str) -> QuerySource:
    q = question.lower()

    sql_score = sum(1 for p in SQL_SIGNALS if re.search(p, q))
    rag_score = sum(1 for p in RAG_SIGNALS if re.search(p, q))

    if sql_score > 0 and rag_score > 0:
        return "hybrid"
    if sql_score > rag_score:
        return "sql"
    if rag_score > sql_score:
        return "rag"

    # heuristic: short factual questions lean SQL, longer explanatory lean RAG
    if len(question.split()) <= 12 and any(
        kw in q for kw in ["id", "code", "number", "name of", "which", "who"]
    ):
        return "sql"

    return "rag"   # default to document search
