import re


def split_large_section(
    text: str,
    chunk_size: int = 1000,
    overlap_lines: int = 2
):
    """
    Split a large section using complete lines.

    This avoids cutting words in the middle.
    A small number of complete lines are repeated
    between chunks to preserve context.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:

        line_length = len(line)

        # If adding the line exceeds the limit,
        # save the current chunk.
        if (
            current_lines
            and current_length + line_length + 1 > chunk_size
        ):

            chunks.append(
                "\n".join(current_lines)
            )

            # Keep last few complete lines as overlap
            overlap = current_lines[-overlap_lines:]

            current_lines = overlap.copy()

            current_length = sum(
                len(item) + 1
                for item in current_lines
            )

        current_lines.append(line)

        current_length += line_length + 1

    # Add remaining lines
    if current_lines:

        chunks.append(
            "\n".join(current_lines)
        )

    return chunks


def remove_irrelevant_content(text: str):
    """
    Remove obvious marketing/testimonial sections
    that are not useful for course information retrieval.
    """

    if not text:
        return ""

    lines = text.split("\n")

    cleaned_lines = []

    stop_phrases = [
        "IT Professionals",
        "Upgraded",
        "View All Placements",
        "Top Companies Hiring Our Students",
        "Explore Other Courses",
        "What Students Say"
    ]

    for line in lines:

        line_lower = line.lower().strip()

        should_stop = False

        for phrase in stop_phrases:

            if phrase.lower() in line_lower:
                should_stop = True
                break

        if should_stop:
            break

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def section_chunk_text(
    text: str,
    max_chunk_size: int = 1000,
    overlap_lines: int = 2
):
    """
    Create meaningful chunks from course webpage text.

    Small sections remain intact.

    Large sections are split using complete lines.
    """

    if not text:
        return []

    # Remove obvious irrelevant marketing content
    text = remove_irrelevant_content(text)

    headings = [
        "Duration",
        "Eligibility",
        "Course Highlights",
        "Course Modules",
        "Introduction to Machine Learning & Data Science in Industry",
        "Programming Foundations for ML",
        "Mathematics & Statistics for Machine Learning",
        "Data Acquisition, Cleaning, and Preprocessing",
        "Supervised Learning Techniques",
        "Unsupervised Learning Techniques",
        "Neural Networks & Deep Learning",
        "Deep Learning",
        "Generative AI",
        "NLP"
    ]

    pattern = "|".join(
        re.escape(heading)
        for heading in headings
    )

    parts = re.split(
        f"(?=^(?:{pattern})$)",
        text,
        flags=re.MULTILINE | re.IGNORECASE
    )

    chunks = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if len(part) < 20:
            continue

        # Keep small sections intact
        if len(part) <= max_chunk_size:

            chunks.append(part)

        else:

            smaller_chunks = split_large_section(
                part,
                chunk_size=max_chunk_size,
                overlap_lines=overlap_lines
            )

            chunks.extend(smaller_chunks)

    return chunks


if __name__ == "__main__":

    sample_text = """
AI & Data Science Training

Duration
Flexible schedule based on offline & online classes

Eligibility
Open to all graduates and diploma holders

Course Highlights
Python Expertise
Advanced Machine Learning
Project Oriented Learning

Course Modules
- Python
- NumPy
- Pandas
- Machine Learning
- Deep Learning
- Generative AI
"""

    print("=" * 70)
    print("KERALA IT HUB - FINAL CHUNKING TEST")
    print("=" * 70)

    chunks = section_chunk_text(
        sample_text,
        max_chunk_size=1000,
        overlap_lines=2
    )

    print("\nTotal chunks:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):

        print("\n" + "-" * 70)
        print(f"CHUNK {i}")
        print("-" * 70)

        print(chunk)

        print("\nCharacters:", len(chunk))

    print("\n" + "=" * 70)
    print("FINAL CHUNKING TEST COMPLETE")
    print("=" * 70)