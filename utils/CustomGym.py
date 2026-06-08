from typing import Optional
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

from .DatabaseAux import DatabaseConnection
from .actions import atomSqliteDict, operationsDict, tablesDict, HTTP_STATUS_DICT

import numpy as np

import re


from collections import Counter


class SqlCurriculumGym(gym.Env):
    metadata = None

    def __init__(self, stage:int = 1, episode_length:int = 1000, database_path:str = "database/backend.db", 
                 dbms:str = "sqlite", info_index:int = 1):
        """Initialization of the evaluation Gymanisum environment.

        Args:
            stage (int, optional): Curriculum stage used for reward shaping. Defaults to 1.
            episode_length (int, optional): Maximum number of steps per episode. Defaults to 1000.
            database_path (str, optional): Path to the database accessed by APSW. Defaults to "database/backend.db".
            dbms (str, optional): Database dialect, e.g. 'sqlite', 'mysql', 'mariadb', or 'postgresql'. Defaults to "sqlite".
            info_index (int, optional): Column index for expected return. Defaults to 1.
        """        
        super().__init__()

        self.stage = stage
        self.episode_length = episode_length
        self.database_path = database_path
        self.database = None

        self.information_index = info_index
        self.dbms = dbms

        self.MAX_PAYLOAD_LENGTH = 128
        
        # -------- Action Space ---------
        self.select_action_tokens()

        # -------- Observation Space ---------
        self.MAX_OBS_LEN = 256
        #Edit - only the payload, not prefix and suffix
        self.observation_space = spaces.Dict({
            "payload": spaces.MultiDiscrete([len(atomSqliteDict) +1 ] * self.MAX_PAYLOAD_LENGTH),  #+1 since I use 0 as padding 
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


    def configure_stage(self):
        """Configuration of the selected stage. 
        Sets the simulation of the injectable parameter, and the enables the rotation of database size during training.
        """        
        match self.stage:
            case 1:
                self.template_prefix = "SELECT * FROM users WHERE id = '"
                self.template_suffix = "' ;"

            case 2: #2 | 3
                n_cols = self.np_random.integers(2, len(tablesDict[self.target_table_name]) + 1)
                chosen_cols = self.np_random.choice(tablesDict[self.target_table_name], size=n_cols, replace=False)
                col_str = ", ".join(chosen_cols)

                self.target_column_count = len(chosen_cols)

                self.template_prefix = f"SELECT {col_str} FROM {self.target_table_name} WHERE {chosen_cols[0]} = '"
                self.template_suffix = "' ;"

                self.target_table_columns = tablesDict[self.target_table_name]
            
            #Loop with getting the a table in addition to table names.
            # 
            case 3 | 4 | 5: #4 | 5 | 6
                n_cols = self.np_random.integers(2, len(tablesDict[self.target_table_name]) + 1)
                chosen_cols = self.np_random.choice(tablesDict[self.target_table_name], size=n_cols, replace=False)
                col_str = ", ".join(chosen_cols)

                self.target_column_count = len(chosen_cols)
                self.target_table_columns = tablesDict[self.target_table_name]

                self.template_prefix = f"SELECT {col_str} FROM {self.target_table_name} WHERE {chosen_cols[0]} = '"
                self.template_suffix = "' ;"

                self.extraction_table = self.select_random_table()
                self.extraction_table_columns = tablesDict[self.extraction_table]

                tables_to_extract_num = self.np_random.integers(2, len(tablesDict.keys()))
                self.tables_to_extract = self.np_random.choice(list(tablesDict.keys()), size=tables_to_extract_num, replace=False)
                
                

    def __get_obs(self) -> dict:
        """Build the current observation dictionary.

        The observation includes the payload atoms, response-category flags, HTTP status, and progress indicators for schema and data extraction.

        Returns:
            dict: Observation matching 'self.observation_space'.
        """

        ATOM_INDEX = {k: i+1 for i, k in enumerate(atomSqliteDict.keys())}

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
            #"step": np.array([self.current_step], dtype=np.int32),
            "http_status": np.array([self.last_status_code/1000], dtype=np.float32),
            "has_tables": np.array([has_tables], dtype=np.float32),
            "has_columns": np.array([has_columns], dtype=np.float32),
            "order_index_norm": np.array([self.order_index / 50], dtype=np.float32),
            #"limit_discovered": np.array([1.0 if self.order_roof else 0.0], dtype=np.float32),
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
        #self.last_execute_new_tables = 0
        #self.last_execute_new_columns = 0
        #self.last_execute_nonempty = 0.0
        self.last_execute_sql_error = 0.0
        
        # step_cost_list = [0.0, 0.01, 0.001, 0.01, 0.01]
        # self.STEP_COST = step_cost_list[self.stage]
        # ----------Reset payload array ----------------
        self.query_atoms = []
        # Number of columns
        self.order_index = 1
        self.order_roof = False  # A signal to introduce that the order index roof has been reached. 
        self.column_count_correct = False
        #What the last order index was when the agent executed that got 200..
        self.last_order_index = 1

        #Configure the stage
        self.target_table_name = None
        self.target_table_name = self.select_random_table()
        self.configure_stage()

        #reset response categories:
        self.response_categories = self.process_response_categories()
        # set the pointer and the cursors accordingly to the prefix:

                #reset the database
        if self.database:
            self.database.close()

        self.database = DatabaseConnection(self.database_path)

        # ---------- Set reward params ----------------
        self.GT_ROWS = self.compute_ground_truth() #Ground truth of total rows in database or per stage 
        self.remaining_rows = set(self.GT_ROWS)
        self.total_rows = len(self.GT_ROWS)
        

        #print(f"Total Rows: {self.total_rows}")

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
                if self.stage >= 2:
                    penalty -= 0.2
            if self.stage >= 3 and self.TABLE_NAME_LIST:
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
            if self.stage >= 4 and (not self.TABLE_NAME_LIST or not self.COLUMN_NAMES):
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
                sql_piece = atomSqliteDict[atom]
                payload += sql_piece(self) if callable(sql_piece) else sql_piece


            sql = f"{self.template_prefix}{payload}{self.template_suffix}"
            self.last_sql = sql
            self.last_payload = payload

            response = self.database.execute_query(sql)
            current_response_hash = hash(str(response))
            
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

            # set variables based on the return
            if isinstance(response, str):
                self.last_execute_sql_error = 1.0
                #self.last_execute_nonempty = 0.0
                self.last_status_code = 500
            elif isinstance(response, list):
                self.last_execute_sql_error = 0.0
                #self.last_execute_nonempty = 1.0 if len(response) > 0 else 0.0
                self.last_status_code = 200
            else:
                self.last_execute_sql_error = 0.0
                #self.last_execute_nonempty = 0.0
                self.last_status_code = 200

            if self.last_status_code == 200 and self.last_execute_sql_error == 0.0:
                self.last_order_index = self.order_index


            if self.stage > 1:
                if "UNION_SQLITE_MASTER" in self.query_atoms:
                    prev_tables = set(self.TABLE_NAME_LIST)
                    tablenames = self.extract_table_names(response)
                    if tablenames:
                        if self.order_index != self.target_column_count:
                            print(f"[*] Tables found, but wrong index, {self.order_index}/{self.target_column_count}")
                        self.TABLE_NAME_LIST = tablenames
                        self.column_count_correct = (self.order_index == self.target_column_count and self.last_status_code == 200)

                if "UNION_SQLITE_COLUMN" in self.query_atoms:
                    prev_cols = set(self.COLUMN_NAMES)
                    column_names = self.extract_column_names(response)
                    if column_names:
                        self.COLUMN_NAMES = column_names
            

            if self.stage >= 4:
                if self.COLUMN_NAMES:
                    if "UNION_CURRENT_TABLE" in self.query_atoms:
                        if isinstance(response, list) and len(response) >= 1:
                            current_target = self.TABLE_NAME_LIST[self.TABLE_LIST_POINTER]
                            if current_target not in self.EXTRACTED_TABLES:
                                if current_target in sql:
                                    self.EXTRACTED_TABLES.append(current_target)
                                    self.TOTAL_EXTRACTED_ROWS += len(response)
    
        #get reward for action based on atoms and / or response:
        #reduce for stage 6, since ep.length is longer. penalty is negative
        if self.stage == 5:
            penalty *= 0.0
        reward += penalty

        #Reward for endstate and phi
        _reward, done = self.get_reward(illegal_action, reused_payload)
        reward += _reward

        #For debugging
        #end condition

        if self.current_step >= self.episode_length:
            if not self.executed:
                reward -= 0.5
            truncated = True
            reward -= 5.0
            
        return self.__get_obs(), reward, done, truncated, {
                **self.__get_info()
            }

    def extract_column_names(self, response):
        """Extract column names from a schema-discovery response.

        For SQLite, it parses 'CREATE TABLE' statements and extracts the column identifiers from the statement.

        Args:
            response: Parsed target response.

        Returns:
            list: Extracted column names.
        """      
        SKIP_ITEMS = {"PRIMARY","FOREIGN","CONSTRAINT","UNIQUE","CHECK","KEY","REFERENCES"}
        columns = []
        # check if the response is a list. The first tuple in the list is the CREATE table
        if isinstance(response, list):
            try:
                for element in response:
                    # Check if the tuple contains the SQL string (in index 0, since we are looking at null columns)
                    if isinstance(element[0], str) and "CREATE TABLE" in element[0]:
                        sql_text = element[0]
                        #Find everything between the FIRST '(' and the LAST ')'
                        start_idx = sql_text.find('(')
                        end_idx = sql_text.rfind(')')
                        
                        if start_idx != -1 and end_idx != -1:
                            inner_content = sql_text[start_idx + 1:end_idx].strip()
                            
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
                                    columns.append(column_name)
            except Exception as exc:
                pass
        return columns

    def extract_table_names(self, response):
        """Extract table names from a parsed response.

        Args:
            response: Parsed response data.

        Returns:
            list: List of table names if the response is a list, else empty list.
        """     
        try:
            if isinstance(response, list):
                tablenames = [t[0] for t in response if all(x is None for x in t[1:])]
                #print(f"Found the following Tablenames at step {self.current_step}: {tablenames}")
                if tablenames:
                    if self.stage == 3 or self.stage == 4:
                        tablenames = [self.extraction_table]  # we only do one at stage 5, all at stage 6
                    elif self.stage == 5:
                        tablenames = self.tables_to_extract.tolist()
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

    def add_atom(self, atom, index=None):
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


    def remove_atom(self, atom):
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

    def set_database(self):
        if hasattr(self, "database") and self.database is not None:
            self.database.close()
        self.database = DatabaseConnection(self.database_path)
            
    def process_response_categories(self) -> dict[str, bool]:
        """"
        Analyzes the last execution and determining the capability level of the query, used for dynamic rewarding
        """
        response_categories = {
            "grammar_valid": False,     #grammar valid and query executed   
            "sql_error": False,
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
        STRUCTURAL_ATOMS = {
            "SQL_COMMENT", "OR_TRUE", 
            "OR_STRING_TRUE", "OR_BOOLEAN_TRUE", "UNION_SQLITE_MASTER",
            "UNION_CURRENT_TABLE", "ORDER_BY"
        }
        #constant for distance rewarding
        progress = 0.4
        
        if self.stage == 1:
            ghost_penalty = 0.05
        else:
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
        if self.stage == 1:
            if self.check_data_returned():
                phi += 0.5                     

        if self.stage == 2:
            # stagnation is not rewarded:
            if self.response_categories.get('stagnation_signal', False):
                phi -= 0.2

            # Bonus for proximity to Ground Truth.
            dist = abs(self.last_order_index - self.target_column_count)
            # Reward increases as dist approachs 0 (max 0.4 bonus)
            phi += progress * (1.0 - min(dist, 10)/10.0)

            if cO_index != float('inf'):
                if "UNION_SQLITE_MASTER" in self.query_atoms[:cO_index]:
                    phi += 0.5

        if self.stage == 3:
            if self.column_count_correct:
                phi += 0.2

            if self.TABLE_NAME_LIST:
                phi += 0.3 #Slightly higher than not having table.
                if cO_index != float('inf'):
                    if "UNION_SQLITE_COLUMN" in self.query_atoms: 
                        phi += 0.25 #Phi for using the correct atom

            if self.COLUMN_NAMES:
                phi += 0.25
                if Counter(self.COLUMN_NAMES) == Counter(self.extraction_table_columns):
                    phi += 0.2

            if self.response_categories.get("stagnation_signal", False):
                phi -= 0.2

        if self.stage == 4:
            #Maintaining the identified amount of columns are crucuial. 
            if self.column_count_correct:
                phi += 0.2

            if self.TABLE_NAME_LIST:
                phi += 0.2

            if self.COLUMN_NAMES:
                    phi += 0.2
            if cO_index != float('inf'):
                if "UNION_CURRENT_TABLE" in self.query_atoms[:cO_index]:
                    phi += 0.2
                    
                if self.EXTRACTED_TABLES:
                    phi += 0.2

            if self.response_categories.get("stagnation_signal", False):
                phi -= 0.2
                

        elif self.stage == 5:
            #Total phi= 1.2

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

    def check_data_returned(self) -> float:
        if self.executed and isinstance(self.last_response, list):
            if len(self.last_response) > 0:
                if any(x in self.query_atoms for x in ["UNION_SQLITE_MASTER", "UNION_SQLITE_COLUMN"]):
                    return 0.0
                else:
                    return 1.0
        return 0.0

    def calculate_pbrs_reward(self, phi_now) -> float:
        """Compute the potential-based shaping reward from the current potential

        Args:
            phi_now (float): Current state's potential

        Returns:
            float: Current shaping reward
        """    
        phi_prev = self.prev_phi
        self.prev_phi = phi_now
        return self.gamma * phi_now - phi_prev


    def get_reward(self, illegal_action: bool, reused_payload: bool):
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
            reward -= 0.05 if self.stage < 5 else 0.005
        if reused_payload:
            reward -= 0.05 if self.stage < 5 else 0.005

        penalties = len(re.findall(r"'{2,}|--{2,}", self.last_payload))
        reward -= 0.02 * penalties if self.stage < 5 else 0.02

        if self.op_name == "EXECUTE":
            if len(self.last_response) > 0:
                new_rows = self.extract_rows_from_response(self.last_response) & self.remaining_rows       
                if new_rows:
                    if self.stage <= 5:
                        self.remaining_rows -= new_rows
                        reward += 0.01 * len(new_rows)
            if len(self.query_atoms) == 0:
                reward -= 0.1
        
        #Specific win conditions
        success = False
        match self.stage:
            case 1:
                success = self.check_data_returned()
            case 2:
                success = self.TABLE_NAME_LIST
            case 3:
                success = Counter(self.COLUMN_NAMES) == Counter(self.extraction_table_columns)
            case 4:
                success = self.EXTRACTED_TABLES
            case 5:
                if self.TABLE_NAME_LIST and self.EXTRACTED_TABLES:
                    success = Counter(self.EXTRACTED_TABLES) == Counter(self.TABLE_NAME_LIST)

        if success:
            terminal_reward = 5.0 if self.stage == 5 else 1.5
            #To aid the agent and critic in evaluating the attempt, we need to normalize the step usage of a 3 column target and a 27 column target. 
            #The steps taken is normalized by steps taken by each column
            #100 steps for 27 columns are better than 100 steps for 3 columns.

            MAX_EFFICIENCY_VALUE = 0.3

                #Complexity buffer is the steps we set as between efficient and non efficient. 
                #for stage 5, we have column count + steps per table 
            if self.stage == 5:
                complexity_buffer = (self.target_column_count * 3) + (len(self.TABLE_NAME_LIST) * 4) 
            else:
                complexity_buffer = self.target_column_count * 5
                
                complexity_buffer = (self.target_column_count * 3) + (len(self.TABLE_NAME_LIST) * 4) #a max of 27 * 3 + 11 * 10 == 191

            


            if self.current_step <= complexity_buffer:
                efficiency_bonus = complexity_buffer - self.current_step
                bonus_ratio = efficiency_bonus / complexity_buffer
                efficiency_modifier = bonus_ratio * MAX_EFFICIENCY_VALUE
            else:
                efficiency_penalty = self.current_step - complexity_buffer
                max_efficiency_penalty = max(1, self.episode_length - complexity_buffer) # Prevent division by zero
                bonus_ratio = efficiency_penalty / complexity_buffer
                efficiency_modifier = bonus_ratio * MAX_EFFICIENCY_VALUE

                penalty_ratio = efficiency_penalty / max_efficiency_penalty

                efficiency_modifier = -min(MAX_EFFICIENCY_VALUE, penalty_ratio * MAX_EFFICIENCY_VALUE)
            
            terminal_reward -= efficiency_modifier

            print(f"[+] STAGE {self.stage} SOLVED | Step: {self.current_step} | Complexity: {self.target_column_count} | Reward: {terminal_reward}")

            return terminal_reward + reward, True 

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
        if self.database:
            self.database.close()
            self.database = None

    def select_action_tokens(self):
        self.set_action_space()

    def set_action_space(self) -> spaces.Dict:
        """Create the environment's discrete action space."""
        self.action_space = spaces.Discrete(len(operationsDict))
        
    def compute_ground_truth(self):
        """
        Returns a set of row hashes representing all extractable data.
        """
        gt = set()

        
        match self.stage:
            case 1:
                rows = self.database.execute_query(sql="SELECT * FROM users;")
                for row in rows:
                        gt.add(hash(row))

            # case 2 :
            #     rows = self.database.execute_query(f"{self.template_prefix}' OR 1=1 --")
            #     for row in rows:
            #             gt.add(hash(row))

            case 2:
                rows = self.database.execute_query(sql="SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                for row in rows:
                        gt.add(hash(row))

            case 3:
                if self.extraction_table_columns:
                    for col in self.extraction_table_columns:
                        gt.add(col)

            case 4:
                rows = self.database.execute_query(sql=f"SELECT * FROM {self.extraction_table}")
                for row in rows:
                        gt.add(hash(row))

            case 5:
                tables = self.tables_to_extract
                for table in tables:
                    rows = self.database.execute_query(f"SELECT * FROM {table}")
                    for row in rows:
                        gt.add(hash(row))
            case 6:
                tables = self.database.execute_query(sql="SELECT name FROM sqlite_master WHERE type='table'")
                for (table,) in tables:
                    rows = self.database.execute_query(f"SELECT * FROM {table}")
                    for row in rows:
                        gt.add(hash(row))

        return gt

    
    def extract_rows_from_response(self, response_rows):         
        extracted = set()
        for row in response_rows:
            extracted.add(hash(tuple(row)))
        return extracted
        
    def select_random_table(self):
            table_names = list(tablesDict.keys())
            random_table = str(self.np_random.choice(table_names))
            while self.target_table_name and self.target_table_name == random_table:
                random_table = str(self.np_random.choice(table_names))
            return random_table

if __name__ == "__main__":
    print()