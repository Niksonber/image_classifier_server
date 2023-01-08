import sys
import requests


file_path = sys.argv[1]
res = requests.post('http://0.0.0.0:5000/classify', files={'image': file_path})
print(res.content.decode())
