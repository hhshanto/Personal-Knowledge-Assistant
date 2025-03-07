try:
    # Try to import pysqlite3 and patch sqlite3
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
    print("Successfully patched sqlite3 with pysqlite3")
except ImportError:
    # If pysqlite3 is not available, use the default sqlite3
    print("pysqlite3 not available, using default sqlite3")
    pass