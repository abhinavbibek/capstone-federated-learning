import os
import datetime


def log_message(message, file_path):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    line = f"[{timestamp}] {message}"

    with open(file_path, "a") as f:
        f.write(line + "\n")

    print(line)