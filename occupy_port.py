import socket
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 8765))
s.listen(1)
print("Occupying port 8765")
while True:
    time.sleep(1)
