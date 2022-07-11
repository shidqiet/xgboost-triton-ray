"""
Test requesting inference via HTTP POST
"""

import requests
import json

url = "http://127.0.0.1:6060/api/infer"

payload = json.dumps(
{
  "sepal_length": 0.0,
  "sepal_width": 0.0,
  "petal_length": 0.0,
  "petal_width": 0.0
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
