import sys
import os


# =========================================================
# PROJECT ROOT
# =========================================================

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    project_root
)


# =========================================================
# IMPORT
# =========================================================

from backend.rag_generator import (
    generate_rag_answer
)


# =========================================================
# TEST DATA
# =========================================================

query = (
    "What Python topics are covered "
    "in this course?"
)


retrieved_results = [

    {
        "content": """
The programme covers Core Python.

Key topics include data types,
control flow, functions, file handling,
and object-oriented programming.

The programme also covers Pandas,
NumPy, Scikit-learn and Matplotlib.
""",

        "source_title": (
            "Python Data Management Course | "
            "ASAP Kerala"
        ),

        "source_url": (
            "https://asapkerala.gov.in/"
            "course/python-for-data-management/"
        )
    },

    {

        "content": """
The course includes Python programming,
AI and ML fundamentals, neural networks,
deep learning, generative AI, Power BI
and Tableau.
""",

        "source_title": (
            "Data Science Course in Kerala"
        ),

        "source_url": (
            "https://example.com/course"
        )
    }

]


# =========================================================
# RUN
# =========================================================

print("=" * 70)

print(
    "KERALA IT HUB - DAY 11 RAG GENERATOR TEST"
)

print("=" * 70)


print("\nQuestion:")
print(query)


print("\nGenerating answer...")


result = generate_rag_answer(
    query=query,
    retrieved_results=retrieved_results
)


# =========================================================
# DISPLAY
# =========================================================

print("\n" + "=" * 70)

print("GENERATED ANSWER")

print("=" * 70)

print(
    result["answer"]
)


print("\n" + "=" * 70)

print("SOURCES")

print("=" * 70)


for source in result["sources"]:

    print(
        f"- {source['title']}"
    )

    print(
        f"  {source['url']}"
    )


print("\n" + "=" * 70)

print(
    "DAY 11 RAG GENERATOR TEST COMPLETE"
)

print("=" * 70)