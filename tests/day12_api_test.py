import json
import requests


URL = "http://127.0.0.1:8000/ask"


question = (
    "What Python topics are covered "
    "in Data Science courses in Kerala?"
)


payload = {
    "question": question
}


print("=" * 80)
print("KERALA IT HUB - DAY 12 API TEST")
print("=" * 80)


response = requests.post(
    URL,
    json=payload,
    timeout=120
)


print("\nStatus code:")
print(response.status_code)


print("\nContent-Type:")
print(response.headers.get("content-type"))


print("\nAPI Response:")
print(
    json.dumps(
        response.json(),
        indent=2,
        ensure_ascii=False
    )
)


print("\n" + "=" * 80)
print("DAY 12 API TEST COMPLETE")
print("=" * 80)