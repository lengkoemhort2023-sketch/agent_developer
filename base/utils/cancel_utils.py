cancelled_sessions = {}

# Map session_id -> celery task id for running generation tasks
session_task_map = {}

def set_session_task(session_id: str, task_id: str):
	session_task_map[str(session_id)] = task_id

def get_session_task(session_id: str):
	return session_task_map.get(str(session_id))

def clear_session_task(session_id: str):
	if str(session_id) in session_task_map:
		del session_task_map[str(session_id)]






