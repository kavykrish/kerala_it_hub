import json
import requests


URL = "http://127.0.0.1:8000/ask"


questions = [
    "What are the eligibility requirements for Data Science courses in Kerala?",

    "What is the duration of Data Science courses in Kerala?",

    "Which IT courses are available in Kerala?",

    "What cybersecurity courses are available in Kerala?",

    "Which courses include Python?"
]


print("=" * 80)
print("KERALA IT HUB - DAY 13 RAG QUALITY TEST")
print("=" * 80)


for number, question in enumerate(
    questions,
    start=1
):

    print("\n" + "-" * 80)
    print(f"QUESTION {number}")
    print("-" * 80)

    print("\nQuestion:")
    print(question)

    payload = {
        "question": question
    }

    try:

        response = requests.post(
            URL,
            json=payload,
            timeout=180
        )

        print("\nStatus code:")
        print(response.status_code)

        data = response.json()

        print("\nAnswer:")
        print(
            data.get(
                "answer",
                "No answer returned."
            )
        )

        print("\nSources:")

        sources = data.get(
            "sources",
            []
        )

        if sources:

            for index, source in enumerate(
                sources,
                start=1
            ):

                print(
                    f"{index}. "
                    f"{source.get('title', '')}"
                )

                print(
                    f"   {source.get('url', '')}"
                )

        else:

            print("No sources returned.")

        print("\nRetrieval:")

        retrieval = data.get(
            "retrieval",
            {}
        )

        print(
            json.dumps(
                retrieval,
                indent=2
            )
        )

    except Exception as e:

        print("\nERROR:")
        print(e)


print("\n" + "=" * 80)
print("DAY 13 RAG QUALITY TEST COMPLETE")
print("=" * 80)