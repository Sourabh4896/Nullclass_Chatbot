# scheduled_update.py
import schedule, time
from utils import update_vector_db

def job():
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    update_vector_db(url)

# Run every 6 hours
schedule.every(6).hours.do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
