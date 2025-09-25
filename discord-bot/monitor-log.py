import os
import sys
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = os.environ.get("LOG_FILE", "/home/user/Github/senior-thesis/discord-bot/train.log")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", 300))
TIMEOUT = int(os.environ.get("TIMEOUT", 300))
DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
USER_ID = os.environ.get("USER_ID", None)

def send_alert(message):
    payload = {"content": message}
    requests.post(DISCORD_WEBHOOK_URL, json=payload)

def tail_last_line(file_path):
    try:
        with open(file_path, 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            return f.readline().decode(errors='replace').strip()
    except Exception as e:
        return f"(Error reading last line: {e})"

def main():
    ticker = 0
    send_alert("`Beginning monitoring...`")
    while True:
        try:
            last_modified = os.path.getmtime(LOG_FILE)
            elapsed = time.time() - last_modified
            now = datetime.now().strftime('%m-%d %H:%M')

            if elapsed >= TIMEOUT:
                msg = f"`[{now}] Training may have crashed (last update was` <t:{int(last_modified)}:R>`).`"
                if USER_ID:
                    msg += f" <@{USER_ID}>"
                print(msg)
                send_alert(msg)
                sys.exit(0)
            elif ticker % (3600 // CHECK_INTERVAL) != 0:
                ticker += 1
            else:
                last_line = tail_last_line(LOG_FILE)
                print(f"[{now}] {last_line}")
                send_alert(f"`[{now}] {last_line}`")
                ticker += 1
        except Exception as e:
            print(f"Error checking log: {e}")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()