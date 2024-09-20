# import requests
# import time
# from random import randbytes

# URL = 'https://modal-labs--latency-test-crazy-web.modal.run'
# long = randbytes(12 * 1024 * 1024)
# short = randbytes(6 * 1024 * 1024)

# # session = requests.Session()

# # #warmup tcp connection
# # for _ in range(5):
# #     session.post(URL, data=long)

# start = time.monotonic()  
# requests.post(URL, files = {'file': ('wavs/thirty.wav', open('wavs/thirty.wav', 'rb'), 'audio/wav')})
# # requests.post(URL, data=short)
# end = time.monotonic()
# print(f"Time taken: {end - start}")


import requests
import time
from random import randbytes

short = randbytes(6 * 1024 * 1024)
long = randbytes(12 * 1024 * 1024)
URL = "https://modal-labs--latency-test-crazy-final-web.modal.run"
session = requests.Session()
for _ in range(5):
    requests.post(URL, data=long)


start = time.monotonic() 
session.post(URL, data=short)
end = time.monotonic()
print(f"Time taken: {end - start}")


