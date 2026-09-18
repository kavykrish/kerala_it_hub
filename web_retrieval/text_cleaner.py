import re


def clean_text(text: str) -> str:
    """
    Clean extracted webpage text.

    Performs basic text normalization:
    - Removes excessive spaces
    - Removes excessive blank lines
    - Removes very short meaningless lines
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Ignore extremely short fragments
        if len(line) <= 2:
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    return cleaned_text.strip()


def is_useful_course_page(text: str) -> bool:
    """
    Check whether webpage text appears to contain
    useful IT course information.
    """

    if not text:
        return False

    text_lower = text.lower()

    course_keywords = [
        "course",
        "training",
        "program",
        "eligibility",
        "duration",
        "fees",
        "fee",
        "curriculum",
        "syllabus",
        "admission",
        "certification",
        "data science",
        "data analytics",
        "artificial intelligence",
        "machine learning",
        "cyber security",
        "cybersecurity",
        "cloud computing",
        "devops",
        "python",
        "java",
        "full stack",
        "web development"
    ]

    matched_keywords = []

    for keyword in course_keywords:

        if keyword in text_lower:
            matched_keywords.append(keyword)

    # Require at least two course-related indicators
    if len(matched_keywords) >= 2:
        return True

    return False


if __name__ == "__main__":

    sample_text = """
    Home
    About
    Courses


    AI & Data Science Training


    Duration: 6 Months


    Eligibility: Graduates and Diploma Holders


    Fees: ₹75,000


    Contact Us
    """

    print("\nRAW TEXT")
    print("=" * 60)
    print(sample_text)

    cleaned = clean_text(sample_text)

    print("\nCLEANED TEXT")
    print("=" * 60)
    print(cleaned)

    print("\nCOURSE PAGE QUALITY CHECK")
    print("=" * 60)

    if is_useful_course_page(cleaned):
        print("Useful course page: YES")
    else:
        print("Useful course page: NO")