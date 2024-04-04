import datetime
import logger
import pandas as pd
import re
from configparser import ConfigParser
from file_read_backwards import FileReadBackwards
from sqlalchemy import create_engine 
from sqlalchemy import text

# Logger
myLogger = logger.getLogger(__name__)

# Read config
config = ConfigParser()
config.read('config.ini')


class Connection:
    def __init__(self, mode):
        self.mode = mode
        self.conn_engine = self.create_sqlache_config()

    def create_sqlache_config(self):
        """
            Connect to different database environment with respect to mode
        """
        if self.mode == "prd":
            HOST = config.get('database', 'host')
            DATABASE = config.get('database', 'database')
            PORT = config.get('database', 'port')
            USER = config.get('database', 'user')
            PASSWORD = config.get('database', 'password')

        elif self.mode == "qas":
            HOST = config.get('database_QAS', 'host')
            DATABASE = config.get('database_QAS', 'database')
            PORT = config.get('database_QAS', 'port')
            USER = config.get('database_QAS', 'user')
            PASSWORD = config.get('database_QAS', 'password')

        # Create a connection engine
        engine_config = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        conn_engine = create_engine(engine_config)

        return conn_engine
    
    def execute_sql(self, command):
        """
            Execute SQL command
        """
        
        try: 
            with self.conn_engine.connect() as conn:
                conn.execute(command)

        except:
            myLogger.exception("execute_sql connection failed")

            
    def fetch_control_table(self, plant) -> pd.DataFrame:
        """
            Fetch undone plants from analysis_console (Select the recent one).
        """
        
        command = f"""SELECT * FROM sourcer.analysis_console WHERE (data_ready = 'Y' and ai_process is Null and plant = '{plant}') ORDER BY mrp_run_date DESC LIMIT 1"""
        
        try: 
            with self.conn_engine.connect() as conn:
                df_control_table = pd.read_sql(sql=command, con=conn)
                
            return df_control_table

        except:
            myLogger.exception("fetch_control_table connection failed")


    def fetch_mrp_sourcer_code_table(self, plant, mrp_run_date, batch_id, post_datetime) -> pd.DataFrame:
        """
            Fetch data from mrp_sourcer_code_analysis table
        """

        command = f"""SELECT * FROM sourcer.mrp_sourcer_code_analysis 
                            WHERE plant = '{plant}' 
                            And mrp_run_date = '{mrp_run_date}'
                            And batch_id = '{batch_id}' 
                            And post_datetime = '{post_datetime}'"""

        try: 
            with self.conn_engine.connect() as conn:
                df_mrp_sourcer_code = pd.read_sql(sql = command, con=conn)
            
            return df_mrp_sourcer_code

        except:
            myLogger.exception("fetch_mrp_sourcer_code_table connection failed")


    def update_analysis_console(self, id, column_name, value):
        """
            Update column_name status to value in analysis_console table
        """

        command = f"""UPDATE sourcer.analysis_console SET {column_name} = '{value}' 
                      WHERE (id = '{id}')"""
        
        try: 
            with self.conn_engine.connect() as conn:
                conn.execute(command)

        except:
            myLogger.exception("update_analysis_console connection failed")


    def update_ai_process(self, id, status):
        """
            Update AI process status in analysis_console table
        """

        command = f"""UPDATE sourcer.analysis_console SET ai_process='{status}'
                      WHERE (id='{id}')"""
        
        try: 
            with self.conn_engine.connect() as conn:
                conn.execute(command)

        except:
            myLogger.exception(f"update_analysis_console connection failed in step: {status}")


    def read_log_file_and_upload_to_DB(self, plant, source_file="predict_classification.py"):
        """
            Read a log file containing recent logs and upload them to Postgresql DB
        """
        
        filepath = "./logs/predict_classification.log"
        # Fetch the most recent logs
        list_uploads = []  # A list for gathering contents to be uploaded

        # We use FileReadBackwards to read a log file in reverse order because we only need to read recent (latest) logs from the end of the file.
        # What's more, we don't have to read the whole contents of the file in case the file size is getting larger and larger.
        # https://stackoverflow.com/questions/2301789/how-to-read-a-file-in-reverse-order
        try:
            with FileReadBackwards(filepath, encoding="utf-8") as f:
                # Get lines by lines starting from the last line up
                for line in f:
                    # We use regular expression to get contents with matched datetime format in order to fetch recent logs (currently from now to up to 15 minutes ago)
                    # https://docs.python.org/3/library/re.html
                    m = re.match(r"\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d", line)
                    if m is not None:  # If this line has a matched datetime format
                        time = datetime.datetime.strptime(m[0], "%Y-%m-%d %H:%M:%S")
                        if time > datetime.datetime.now() - datetime.timedelta(minutes=15):  # We only fetch recent logs, not others
                            #print(f"Matched time: {time}")
                            list_uploads.append(line)
                        else:  # This line has outdated time stamp which we don't want
                            break
                    else:  # Perhaps this line does not contain a time stamp, but is actually followed by a line with matched time stamp
                        list_uploads.append(line)
        except:
            myLogger.exception(f"An error while proceeding a log file.")

        # Because contents in the list are appended in reverse order, we reverse them back to be in chronological order. 
        # Then, we use 'newline' to join all contents to a string
        str_uploads = '\n'.join(list(reversed(list_uploads)))
        #print(f"str_uploads: {str_uploads}")

        # We upload logs to Postgresql DB
        command = f"""INSERT INTO sourcer.log_message (update_time, plant, message, source_file, status) VALUES ('{datetime.datetime.now()}', '{plant}', '{str_uploads}', '{source_file}', 'undone')"""
        self.execute_sql(text(command))