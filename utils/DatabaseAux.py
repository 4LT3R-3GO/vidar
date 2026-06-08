import apsw
from environs import env
from pathlib import Path

import re

env.read_env()

class DatabaseConnection:
    def __init__(self, db_file: str | Path):
        self.db_filepath = Path(db_file)
        self.sql_version = apsw.sqlite_lib_version()

        self.validate_path()
        self.connection = None
        self.connect()
        
    def connect(self) -> apsw.Connection:
        """A function for opening and reading database to memory

        Args:
            pass
            
        Returns:
            pass
        """        
        if self.connection is None:
            in_memory_db = apsw.Connection(":memory:")
            on_disk_db = apsw.Connection(filename=str(self.db_filepath))
            with in_memory_db.backup("main", on_disk_db, "main") as backup:
                while not backup.done:
                    backup.step(50)

            # ADD SAFETY HERE:
            # limit execution to ~50k VM ops before aborting
            in_memory_db.setprogresshandler(lambda: 1, 500_000)
            # also prevent writer locks from spinning
            in_memory_db.setbusytimeout(20)
            
            self.connection = in_memory_db
        return self.connection
    
    def close(self):
        """Close the connection:"""
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception as exc:
                print(f"[!] DB WARNING: close() failed: {exc!r}")
            finally:
                self.connection = None
    def validate_path(self):
        """A function to validate the provided database path

        Raises:
            ValueError: If the provided path does not exists.
        """        
        if not self.db_filepath.exists():
            raise ValueError(f"[-] Non-existing SQLite database: {self.db_filepath}")
        
    def execute_query(self, sql):
        try:
            with self.connection as con:
                curr = con.cursor()
                curr.execute(sql)

                response = curr.fetchall()
                curr.close()

                return response
        except apsw.InterruptError:
            return "Error: Query exceeded VM instruction limit (Timeout)"
        except apsw.SQLError as sqlerr:
            #early training, this provide good feedback messages. 
            return str(sqlerr)
        except Exception as exc:
            print(f"[-] Error with APSW: {exc!r}")
            return ""
if __name__ == "__main__":
    db = DatabaseConnection("./database/backend.db")
    db.connect()
    sql = ["SELECT id,name FROM products UNION SELECT 1,address FROM users -- "]

    for query in sql: 
        print(f"Executed Query: {query}")
        response = db.execute_query(query)
        print(len(response), type(response))
        print(f"Received Response:\n{response}\n\n")

    lis_t = ['a']

    
    if not lis_t:
        print("test")
    print("syntax error" in response)
