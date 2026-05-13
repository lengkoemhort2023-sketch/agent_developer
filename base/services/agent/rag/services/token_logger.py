import pandas as pd
from datetime import datetime
import json
import os
import uuid
from typing import Dict

class TokenLogger:
    """ Simple token usage logger that saves to Excel"""

    def __init__(self, file_path="logdata/token_usage.csv"):
        self.log_file_path = file_path
        self.logs = []

        self.query_sessions: Dict[str, Dict] = {}
        # Create directory if don't have 
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def current_log_file_path(self):
        """ Return the CSV path for today's date."""
        date_str = datetime.now().strftime("%d-%m-%Y")
        file_path = f"logdata/token-usage-{date_str}.csv"
        return file_path
    

    def start_query_session(self, query_id: str = None, question: str = None, meta_data: Dict = None):
        """Start a new query session and return the session ID"""
        if query_id is None:
            query_id = str(uuid.uuid4())
        
        self.query_sessions[query_id] = {
            'session_id': query_id,
            'question': question,
            'start_time': datetime.now().isoformat(),
            'operation_totals': {},
            'meta_data': meta_data or {}
        }
        return query_id

    def log_token_usage(self, model, operation, usage_metadata: Dict, meta_json=None, query_id: str = None):
        """"Log token usage from LLM response by operation type"""
        try:

            input_tokens = usage_metadata.get('input_tokens', 0)
            output_tokens = usage_metadata.get('output_tokens', 0)
            total_tokens = usage_metadata.get('total_tokens', input_tokens + output_tokens)
            
            # If part of a query session, accumulate tokens there
            if query_id and query_id in self.query_sessions:
                session = self.query_sessions[query_id]

                if operation not in session['operation_totals']:
                    session['operation_totals'][operation] = {
                        'model': model,
                        'operation': operation,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0,
                        'call_count': 0,
                        'meta_data': []
                    }
                
                session['operation_totals'][operation]['input_tokens'] += input_tokens
                session['operation_totals'][operation]['output_tokens'] += output_tokens
                session['operation_totals'][operation]['total_tokens'] += total_tokens
                session['operation_totals'][operation]['call_count'] += 1

                if meta_json:
                    session['operation_totals'][operation]['meta_data'].append(meta_json)
            
            else:
                # Individual log
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'model': model,
                    'operation': operation,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens,
                    'meta_json': json.dumps(meta_json if meta_json else None),
                    'query_id': query_id,
                    'call_count': 1
                }
                self.logs.append(log_entry)
                self.save_to_csv()

        except Exception as e:
            print(f"Log Error")


    def finish_query_session(self, query_id:str):
        """"Finish a query session and log each operation's the aggregated total"""
        if query_id not in self.query_sessions:
            return{}
        
        session = self.query_sessions[query_id]
        session['end_time'] = datetime.now().isoformat()
        
        # Log each operation type's aggregated totals as separate entries
        for operation_name, operation_data in session['operation_totals'].items():
            log_entry = {
                'timestamp': session['end_time'],
                'model': operation_data['model'],
                'operation': operation_name,
                'input_tokens': operation_data['input_tokens'],
                'output_tokens': operation_data['output_tokens'],
                'total_tokens': operation_data['total_tokens'],
                'meta_json': json.dumps({
                    'query_id': query_id,
                    'question': session['question'],
                    'call_count': operation_data['call_count'],
                    'session_start_time': session['start_time'],
                    'session_end_time': session['end_time'],
                    'operation_meta_data': operation_data['meta_data']
                }),
                'query_id': query_id,
                'call_count': operation_data['call_count']
            }
            self.logs.append(log_entry)
        
        self.save_to_csv()
        
        # Clean up session and return summary
        completed_session = self.query_sessions.pop(query_id)
        return completed_session
        
    def log_aggregated_tokens(self, model, operation, total_input_tokens, total_output_tokens, meta_json=None, query_id=None):
        """"Log aggregated token usage (For PDF processing with multiple batch) """
    
        try: 
            log_entry = {
                'timestamp' : datetime.now().isoformat(),
                'model': model,
                'operation' : operation,
                'input_tokens' : total_input_tokens,
                'output_tokens' : total_output_tokens,
                'total_tokens' : total_input_tokens + total_output_tokens,
                'meta_json' : json.dumps(meta_json if meta_json else None),
                'query_id': query_id,
                'call_count': meta_json.get('call_count', 1) if meta_json else 1
            }
            self.logs.append(log_entry)
            self.save_to_csv()
        
        except Exception as e:
            print(f"Log Error: {str(e)}")

    def save_to_csv(self):
        """ Save all log to Excel file """
        try: 
            if self.logs:
                df = pd.DataFrame(self.logs)

                current_file = self.current_log_file_path()
                #if file exist 
                if os.path.exists(current_file):
                    existing_df = pd.read_csv(current_file)
                    df = pd.concat([existing_df,df], ignore_index=True)
                
                df.to_csv(current_file, index=False)
                
                self.logs =[]
        except Exception as e :
            print(f"Error saving log : {e}")
    






