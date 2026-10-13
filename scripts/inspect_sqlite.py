import sqlite3

conn = sqlite3.connect("storage/studio.db")
cur = conn.cursor()
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
print("SQLite tables and row counts:")
for t in tables:
    count = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    print(f"  - {t}: {count} rows")
conn.close()
