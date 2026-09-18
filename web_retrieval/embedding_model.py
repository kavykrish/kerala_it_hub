from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():

    model = SentenceTransformer(
        MODEL_NAME
    )

    return model


def create_embedding(
    model,
    text: str
):

    if not text:
        return None

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding


if __name__ == "__main__":

    print("=" * 70)
    print("KERALA IT HUB - DAY 7 EMBEDDING TEST")
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
        "DAY 7 EMBEDDING TEST COMPLETE"
    )

    print(
        "=" * 70
    )