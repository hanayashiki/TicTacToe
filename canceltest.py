import time
import atexit

atexit.register(lambda : print("fuck"))

try:
    while True:
        time.sleep(0.1)
finally:
    print("fuck!")