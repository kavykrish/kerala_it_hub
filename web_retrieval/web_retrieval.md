# Kerala IT Hub - Day 3 Web Retrieval Pipeline

## Objective

Design the web retrieval pipeline for Kerala IT Hub.

The system should search current web sources for IT and
technology-related courses in Kerala and retrieve relevant
information for RAG-based answer generation.

---

## Retrieval Pipeline

Student Question
        ↓
Query Analysis
        ↓
Search Query Generation
        ↓
Web Search
        ↓
Search Result Filtering
        ↓
Relevant Web Pages
        ↓
Web Page Extraction
        ↓
Text Cleaning
        ↓
Text Chunking
        ↓
Relevant Chunk Retrieval
        ↓
RAG Context
        ↓
Groq LLM
        ↓
Grounded Answer

---

## Query Analysis

The system should identify relevant information from the
user's question such as:

- Course category
- Location
- Budget
- Eligibility
- Learning mode
- User background
- Comparison requirements

Example:

User:
"Find Data Science courses in Kochi under ₹1 lakh."

Extracted:

- Category: Data Science
- Location: Kochi
- Budget: ₹1 lakh
- Intent: Course discovery

---

## Web Search

The system generates a search query using the user's
requirements.

Example:

"Data Science courses Kochi Kerala"

Search results should contain:

- Page title
- URL
- Search snippet
- Source name
- Source type

---

## Source Priority

1. Government sources
2. University sources
3. Government skill-development organizations
4. Technical education sources
5. Official institute websites
6. Trusted education platforms

---

## Page Extraction

Relevant webpages will be fetched and converted into
readable text.

Unnecessary content such as:

- Navigation menus
- Advertisements
- Cookie messages
- Footer content
- Repeated links

should be removed.

---

## Text Chunking

Clean webpage content will be divided into smaller chunks.

Example:

Chunk 1:
Course Overview

Chunk 2:
Eligibility

Chunk 3:
Duration

Chunk 4:
Fees

Chunk 5:
Curriculum

Chunk 6:
Admission Information

---

## Retrieval

The system will retrieve the chunks most relevant to the
user's question.

Example:

Question:
"What is the eligibility?"

The retriever should prioritize chunks containing
eligibility information instead of sending the entire
webpage to the LLM.

---

## RAG

Retrieved chunks will be supplied as context to the LLM.

The LLM should generate answers only from the retrieved
information.

---

## Grounding Rules

- Do not invent course information.
- Do not assume missing fees.
- Do not assume missing eligibility.
- Do not assume course availability.
- Clearly state when information is unavailable.
- Preserve the source URL.
- Recommend verification of changing information from
  the official source.

---

## MCP Tools

The MCP layer will expose web retrieval capabilities
to the Agno agent.

Initial tools:

### search_kerala_courses(query)

Searches the web for relevant Kerala IT course pages.

### fetch_course_page(url)

Fetches and extracts content from a relevant course page.

---

## Project Components

web_retrieval/

- search_engine.py
- page_reader.py
- text_cleaner.py
- chunker.py
- retriever.py

---

## Future Flow

Kivy App
    ↓
FastAPI
    ↓
Agno Agent
    ↓
Groq
    ↓
MCP
    ↓
Web Search
    ↓
Web Page Extraction
    ↓
RAG
    ↓
Grounded Answer