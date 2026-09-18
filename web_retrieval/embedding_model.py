import numpy as np

from fastembed import TextEmbedding


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():

    model = TextEmbedding(
        model_name=MODEL_NAME
    )

    return model


def create_embedding(
    model,
    text: str
):

    if not text:
        return None

    embedding = next(
        model.embed([text])
    )

    # fastembed does not guarantee L2-normalized output for every
    # model, so normalize explicitly (matches the previous
    # normalize_embeddings=True behaviour).
    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    return embedding


if __name__ == "__main__":

    print("=" * 70)
    print("KERALA IT HUB - EMBEDDING TEST")
    print("=" * 70)

    model = load_embedding_model()

    text = "Data Science course in Kerala"

    embedding = create_embedding(
        model,
        text
    )

    print("\nText:")
    print(text)

    print("\nEmbedding created successfully.")

    print(
        "Vector dimensions:",
        len(embedding)
    )

    print("\nFirst 10 values:")
    print(
        embedding[:10]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EMBEDDING TEST COMPLETE"
    )

    print(
        "=" * 70
    )
