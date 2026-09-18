import sys
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, project_root)

from web_retrieval.embedding_model import (
    load_embedding_model,
    create_embedding
)

from sklearn.metrics.pairwise import cosine_similarity


def main():

    print("\n" + "=" * 70)
    print("KERALA IT HUB - DAY 7 SEMANTIC SIMILARITY TEST")
    print("=" * 70)

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------
    # TEXTS
    # --------------------------------------------------

    query = "Who can join this course?"

    relevant_text = (
        "Eligibility: Open to all graduates "
        "and diploma holders."
    )

    unrelated_text = (
        "The course covers Python programming "
        "and machine learning."
    )

    # --------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------

    query_embedding = create_embedding(
        model,
        query
    )

    relevant_embedding = create_embedding(
        model,
        relevant_text
    )

    unrelated_embedding = create_embedding(
        model,
        unrelated_text
    )

    # --------------------------------------------------
    # CALCULATE SIMILARITY
    # --------------------------------------------------

    relevant_score = cosine_similarity(
        [query_embedding],
        [relevant_embedding]
    )[0][0]

    unrelated_score = cosine_similarity(
        [query_embedding],
        [unrelated_embedding]
    )[0][0]

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\nQuery:")
    print(query)

    print("\nRelevant text:")
    print(relevant_text)

    print("\nSemantic similarity:")
    print(round(relevant_score, 4))

    print("\nUnrelated text:")
    print(unrelated_text)

    print("\nSemantic similarity:")
    print(round(unrelated_score, 4))

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    print("\n" + "=" * 70)

    if relevant_score > unrelated_score:

        print(
            "SUCCESS: Relevant text has higher "
            "semantic similarity."
        )

    else:

        print(
            "REVIEW: Similarity result needs investigation."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()