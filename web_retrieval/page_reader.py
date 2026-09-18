import sys
import requests
import trafilatura


def fetch_page(url: str):
    """
    Fetch and extract readable text from a course webpage.
    """

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/142.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        downloaded = response.text

        text = trafilatura.extract(
            downloaded,
            include_links=False,
            include_images=False
        )

        if not text:

            print(
                "Could not extract readable text from the page.",
                file=sys.stderr
            )

            return None

        return text


    except requests.exceptions.RequestException as e:

        print(
            f"Error fetching webpage: {e}",
            file=sys.stderr
        )

        return None


    except Exception as e:

        print(
            f"Error extracting webpage content: {e}",
            file=sys.stderr
        )

        return None


if __name__ == "__main__":

    url = (
        "https://www.rogersoft.com/"
        "course/data-science-training"
    )

    print("\nFetching webpage...")
    print("URL:", url)
    print("\n" + "=" * 70)

    page_text = fetch_page(url)

    if page_text:

        print("Webpage successfully extracted!")
        print("=" * 70)
        print(page_text[:5000])

        print("\n" + "=" * 70)
        print(
            "Total characters extracted:",
            len(page_text)
        )

    else:

        print("Failed to extract webpage content.")