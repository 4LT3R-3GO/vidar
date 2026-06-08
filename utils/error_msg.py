SQL_ERROR_PATTERNS = [
    # Generic
    r"\bno such column\b",
    r"\bno such table\b",
    r"\bsyntax error\b",

    r"\berror: you have an erro in you sql syntax\b",
    # UNION column mismatch across engines
    r"used select statements have a different number of columns",                 
    r"selects?\s+to\s+the\s+left\s+and\s+right\s+of\s+union.*same.*result columns",
    r"each\s+union\s+query\s+must\s+have\s+the\s+same\s+number\s+of\s+columns",
]

