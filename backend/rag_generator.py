import os
import time

from dotenv import load_dotenv
from groq import APIStatusError, Groq

from web_retrieval.course_extractor import NOT_AVAILABLE
from web_retrieval.course_advisor import explain_match


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv(
    "GROQ_API_KEY"
)

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please check your .env file."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=api_key
)


MODEL_NAME = "openai/gpt-oss-20b"


# ============================================================
# RETRY ON TRANSIENT RATE LIMITS
# ============================================================
# Groq's own client auto-retries 429s and 5xxs, but not this specific
# 413 "tokens per minute" rate_limit_exceeded error -- it's treated as
# a permanent "your request is too large" client error. In practice
# it's transient: the account's rolling per-minute token window clears
# itself within well under a minute, so the identical request usually
# succeeds shortly after. Retry it ourselves instead of surfacing a
# failure that would very likely have worked on the next try.

GROQ_RATE_LIMIT_RETRY_WAIT_SECONDS = 20
GROQ_RATE_LIMIT_MAX_RETRIES = 3


def call_groq_with_retry(**kwargs):

    last_error = None

    for attempt in range(GROQ_RATE_LIMIT_MAX_RETRIES + 1):

        try:
            return client.chat.completions.create(**kwargs)

        except APIStatusError as e:

            is_token_rate_limit = (
                e.status_code == 413
                and "rate_limit_exceeded" in str(e)
            )

            if (
                not is_token_rate_limit
                or attempt == GROQ_RATE_LIMIT_MAX_RETRIES
            ):
                raise

            last_error = e

            time.sleep(
                GROQ_RATE_LIMIT_RETRY_WAIT_SECONDS
            )

    raise last_error


# ============================================================
# STRUCTURED CONTEXT (Step 3 -- deduplicated course records)
# ============================================================
# Builds the LLM context from already-deduplicated/merged Course
# records (see web_retrieval/course_merger.py) instead of raw,
# fragmented per-chunk text -- each COURSE block below already
# represents exactly one real institute/course offering, with fields
# merged from however many source pages described it. This is what
# lets the model stop treating the same institute's multiple pages as
# multiple different courses.

_COURSE_CONTEXT_FIELDS = (
    ("Category", "course_category"),
    ("Level", "level"),
    ("Duration", "duration"),
    ("Fees", "fees"),
    ("Eligibility", "eligibility"),
    ("Learning Mode", "learning_mode"),
    ("Location", "location"),
    ("Certification", "certification"),
    ("Admission", "admission_information"),
    ("Placement", "placement_information"),
)


def _build_structured_context(course_records: list, char_budget: int):
    """
    Returns (context_text, sources) built from deduplicated Course
    records, or ("", []) if there's nothing usable to show -- callers
    fall back to the raw-chunk context in that case (see
    generate_rag_answer below).
    """

    context_parts = []
    sources = []
    seen_source_urls = set()
    context_chars = 0

    for index, course in enumerate(course_records, start=1):

        if context_chars >= char_budget:
            break

        institute = course.get("institute_name", NOT_AVAILABLE)
        course_name = course.get("course_name", NOT_AVAILABLE)

        # A record with neither an institute nor a course name isn't
        # useful structured context -- e.g. a page that turned out not
        # to be a course page at all. Skip it here rather than showing
        # the model an empty-looking COURSE block.
        if institute == NOT_AVAILABLE and course_name == NOT_AVAILABLE:
            continue

        lines = [
            f"COURSE {index}",
            "",
            f"Institute: {institute}",
            f"Course: {course_name}",
        ]

        for label, field in _COURSE_CONTEXT_FIELDS:
            lines.append(f"{label}: {course.get(field, NOT_AVAILABLE)}")

        curriculum = course.get("curriculum") or []

        if curriculum:
            lines.append("Curriculum: " + ", ".join(curriculum))

        source_urls = course.get("source_urls") or (
            [course["source_url"]] if course.get("source_url") else []
        )

        if source_urls:
            lines.append("Source URL(s): " + ", ".join(source_urls))

        part = "\n".join(lines)

        context_parts.append(part)
        context_chars += len(part)

        source_title = course.get("source_title", "")

        for url in source_urls:

            if url in seen_source_urls:
                continue

            seen_source_urls.add(url)

            sources.append({
                "title": source_title if source_title != NOT_AVAILABLE else "",
                "url": url
            })

    return "\n\n".join(context_parts), sources


# ============================================================
# RAG ANSWER GENERATOR
# ============================================================

def _build_query_intent_note(query_intent: dict | None) -> str:
    """
    A short, deterministic instruction block telling the model what
    the user specifically asked about (Step 5 Part 8) -- built purely
    from query_intent (see web_retrieval/query_intent.py), no extra
    LLM call. Returns "" when the intent is empty/None, so a query
    with nothing detectable adds nothing to the prompt.
    """

    if not query_intent:
        return ""

    lines = []

    topics = query_intent.get("requested_topics") or []
    fields = query_intent.get("requested_fields") or []
    level = query_intent.get("requested_level")
    location = query_intent.get("requested_location")

    if topics:
        lines.append(f"The user's requested topic is: {', '.join(topics)}.")

    if fields:
        lines.append(
            "The user specifically asked about: "
            + ", ".join(fields)
            + ". Prioritize these details when present, and if a "
            "course/record doesn't have them, say so rather than "
            "omitting it silently."
        )

    if level:
        lines.append(f"The user is asking about the {level} level specifically.")

    if location:
        lines.append(f"The user is asking about courses in {location}.")

    if not lines:
        return ""

    lines.append(
        "Do not treat a related but different technology (for "
        "example Machine Learning or Artificial Intelligence when the "
        "user asked about Data Science) as if it were the requested "
        "topic. Only describe a course as matching the requested "
        "topic when its own course name or category explicitly "
        "supports that. Do not invent missing values."
    )

    return "\n".join(lines)


def _build_advisor_context(course_records: list, advisor_preferences: dict) -> str:
    """
    Advisor Step 2: when the Course Advisor is active, the model gets
    the SAME deterministic ranking/explanations course_advisor.py
    already computed -- it is told the courses are pre-ranked and to
    present them in that order, never to recompute or override which
    one "wins". Every explanation line comes straight from
    course_advisor.explain_match (which itself now leads with a
    deterministic "Overall match status: FULL_MATCH/PARTIAL_MATCH/
    CONFLICT" line -- see course_advisor.overall_match_status), so
    nothing here is invented -- an empty return means there was
    nothing usable to explain (e.g. no course_records), and the caller
    falls back to the plain query-intent note instead.

    Advisor Step 2B: a live response leaked this internal analysis
    almost verbatim as a user-facing "Criterion / Match? / Explanation"
    block (and, in one malformed case, literally echoed those header
    words back as if they were data). The per-course lines are
    unambiguously framed as INTERNAL analysis the model must reason
    from, not copy.

    Advisor Step 2C: two more live problems, both fixed at THIS layer
    (the deterministic classification in course_advisor.py was already
    correct):
    1. A course's raw "Placement" field (shown elsewhere in the
       structured COURSE block context) can contain real but WEAK text
       (a portfolio/project sentence) that course_advisor already
       correctly does not count as placement evidence -- but the model
       was reading that raw field directly and overstating a match
       anyway, since nothing told it the deterministic analysis
       overrides what the raw field says. Now explicit below.
    2. The model was free-handing its own "matches"/"does not match"
       verdict per course instead of using the deterministic Overall
       match status line -- now explicitly required to use ONLY that
       literal status.
    """

    if not course_records or not advisor_preferences:
        return ""

    lines = [
        "INTERNAL ADVISOR ANALYSIS (for your reasoning only -- do not "
        "copy this block, its structure, or its labels into your "
        "answer; use it only to write your own natural-language "
        "summary):",
        "",
        "The courses below have already been ranked by a deterministic "
        "course advisor according to the user's stated preferences. "
        "Present them to the user in this exact order -- never re-rank "
        "them or decide a different course is the best match yourself. "
        "Each course starts with a deterministic \"Overall match "
        "status\" of FULL_MATCH, PARTIAL_MATCH, or CONFLICT, followed "
        "by which of the user's specific preferences it does or "
        "doesn't confirm.",
    ]

    for index, course in enumerate(course_records, start=1):

        explanation = explain_match(course, advisor_preferences)

        if not explanation:
            continue

        course_name = course.get("course_name", NOT_AVAILABLE)
        institute = course.get("institute_name", NOT_AVAILABLE)

        lines.append(
            f"\nCourse {index} ({course_name} - {institute}):\n{explanation}"
        )

    if len(lines) == 3:
        # Nothing had a usable explanation (e.g. the user's query had
        # no detectable preferences at all) -- no advisor-specific
        # context to add.
        return ""

    has_full_match = any(
        "Overall match status: FULL_MATCH" in line for line in lines
    )

    lines.append(
        "\nEND OF INTERNAL ADVISOR ANALYSIS.\n\n"
        "When you write your answer:\n"
        "- Use ONLY the \"Overall match status\" given above for each "
        "course -- never decide for yourself whether a course "
        "matches. If it says FULL_MATCH, you may say the course fully "
        "matches the request. If it says PARTIAL_MATCH, say it "
        "partially matches and name which details are unconfirmed. If "
        "it says CONFLICT, say it conflicts with at least one "
        "requested preference and name which one(s).\n"
        + (
            "- No course above has an Overall match status of "
            "FULL_MATCH -- clearly tell the user that no course can "
            "be fully confirmed as matching every preference from the "
            "retrieved information, then describe the closest partial "
            "matches.\n"
            if not has_full_match else ""
        )
        + "- Summarize each course's match in your own plain, "
        "conversational sentences (for example: mention the fee, the "
        "mode, and a short sentence on why it fits or doesn't).\n"
        "- Never use the literal words \"Criterion\", \"Match?\", or "
        "\"Explanation\" as labels, headers, or column names anywhere "
        "in your answer -- write natural prose or a simple bullet "
        "list instead of a rigid criteria table.\n"
        "- Never mention that this analysis is from an \"advisor\", "
        "a \"ranking\", a \"tier\", or any other internal mechanism -- "
        "just present the course information and why it fits.\n"
        "- Only say a preference (topic, level, location, budget, "
        "mode, or placement support) is matched when the analysis "
        "above actually shows it as confirmed for that course. If the "
        "analysis says information is not available or not confirmed "
        "for a preference, say plainly that it wasn't confirmed -- "
        "never describe a missing or weak detail as an explicit "
        "match.\n"
        "- The course information elsewhere may show a raw "
        "\"Placement\" field with text such as a portfolio or project "
        "description. Do NOT treat that raw text as placement "
        "assistance evidence yourself -- only the placement line in "
        "the analysis above (or its absence) tells you whether "
        "placement support was confirmed. Portfolio, project, resume, "
        "employability, or career-guidance language is explicitly NOT "
        "placement assistance evidence.\n"
        "- Do not infer a course's level, location, mode, or any "
        "other detail from a title, headline, or marketing phrase -- "
        "only from what the analysis above and the retrieved course "
        "information actually state.\n"
        "- Do not invent any detail that isn't in the retrieved "
        "course information or the analysis above."
    )

    return "\n".join(lines)


def generate_rag_answer(
    query: str,
    retrieved_results: list,
    history: list | None = None,
    course_records: list | None = None,
    query_intent: dict | None = None,
    advisor_active: bool = False,
    advisor_preferences: dict | None = None
):
    """
    Generate an answer using retrieved RAG context.

    The LLM answers only from the retrieved
    information and avoids unsupported claims.

    history: recent {"question", "answer"} turns, oldest first, so
    follow-up questions ("what about the fees for that one?") can be
    understood and answered in context instead of as a fresh,
    unrelated question.

    course_records: deduplicated/merged/ranked Course records for this
    question (see web_retrieval/course_merger.py and
    web_retrieval/course_ranker.py), if any were produced. When
    present and non-empty, the model is given these clean,
    one-record-per-real-course blocks instead of raw retrieved
    chunks -- this is what stops the same institute's multiple source
    pages from being presented, and therefore answered, as separate
    courses. Falls back to the original raw-chunk context exactly as
    before when course_records is empty/None, so general questions
    that didn't produce any structured course records keep working
    unchanged.

    query_intent: the deterministic intent dict for this question (see
    web_retrieval/query_intent.py), if available. Purely used to add a
    short instruction note to the prompt (Step 5 Part 8) -- never a
    second LLM call, and never used to drop any course_records/context
    already assembled above.

    advisor_active / advisor_preferences (Advisor Step 2): when the
    deterministic Course Advisor (see web_retrieval/course_advisor.py)
    decided this query is an advisory/recommendation-style question,
    advisor_active is True and advisor_preferences carries the parsed
    preferences (a superset of query_intent). The model is given the
    Advisor's own pre-computed ranking/explanations and told to
    present courses in that order rather than deciding for itself --
    the LLM never calculates the ranking, it only writes prose around
    a decision that was already made deterministically. Falls back to
    the plain query_intent note when advisor_active is False, exactly
    as before Advisor Step 2.
    """

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not query:

        return {
            "answer": "Please provide a question.",
            "sources": []
        }


    # --------------------------------------------------------
    # Validate retrieved results
    # --------------------------------------------------------

    if not retrieved_results:

        return {
            "answer": (
                "I could not find relevant course "
                "information for this question."
            ),
            "sources": []
        }


    # ========================================================
    # PREPARE RETRIEVED CONTEXT
    # ========================================================
    # Capped to a fixed character budget -- Groq's free tier for this
    # model enforces an 8000 tokens-per-minute limit, and widening
    # retrieval depth (more chunks per page, more pages for
    # comparisons) had been pushing some requests to ~13,500 tokens,
    # failing with a 413 "Request too large" error.

    CONTEXT_CHAR_BUDGET = 15000

    # --------------------------------------------------------
    # Preferred path: deduplicated/merged Course records (Step 3).
    # Each block already represents exactly one real course, so this
    # skips straight past the raw-chunk assembly below entirely when
    # it has something usable.
    # --------------------------------------------------------

    context = ""
    sources = []

    if course_records:
        context, sources = _build_structured_context(
            course_records,
            CONTEXT_CHAR_BUDGET
        )

    # --------------------------------------------------------
    # Fallback: original raw-chunk two-pass assembly, unchanged --
    # runs whenever course_records didn't produce anything usable
    # (empty/None, or every record turned out to have no institute/
    # course identity), so general questions keep working exactly as
    # before Step 3. Every distinct source gets its single most
    # relevant chunk first (so a comparison still covers as many
    # institutes as possible), and only once every source has that
    # does a second pass add each source's remaining chunks for extra
    # detail, while budget allows.
    # --------------------------------------------------------

    if not context:

        context_parts = []

        sources = []

        seen_sources = set()

        included_chunks = set()

        source_index = 1

        context_chars = 0


        # ----------------------------------------------------
        # Pass 1: one chunk per distinct source (breadth)
        # ----------------------------------------------------

        for result in retrieved_results:

            if context_chars >= CONTEXT_CHAR_BUDGET:
                break

            content = result.get(
                "content",
                ""
            )

            source_title = result.get(
                "source_title",
                ""
            )

            source_url = result.get(
                "source_url",
                ""
            )

            if not content:
                continue

            source_key = source_url.strip()

            if (
                not source_key
                or source_key in seen_sources
            ):
                continue

            part = f"""
SOURCE {source_index}

Title:
{source_title}

URL:
{source_url}

Content:
{content}
"""

            context_parts.append(
                part
            )

            context_chars += len(
                part
            )

            seen_sources.add(
                source_key
            )

            included_chunks.add(
                (source_key, content)
            )

            sources.append({
                "title": source_title,
                "url": source_url
            })

            source_index += 1


        # ------------------------------------------------
        # Pass 2: remaining chunks for sources already included
        # (extra detail, only while budget allows)
        # ------------------------------------------------

        for result in retrieved_results:

            if context_chars >= CONTEXT_CHAR_BUDGET:
                break

            content = result.get(
                "content",
                ""
            )

            source_title = result.get(
                "source_title",
                ""
            )

            source_url = result.get(
                "source_url",
                ""
            )

            if not content:
                continue

            source_key = source_url.strip()

            if source_key not in seen_sources:
                continue

            if (source_key, content) in included_chunks:
                continue

            part = f"""
ADDITIONAL CONTENT

Title:
{source_title}

URL:
{source_url}

Content:
{content}
"""

            context_parts.append(
                part
            )

            context_chars += len(
                part
            )

            included_chunks.add(
                (source_key, content)
            )


        # ------------------------------------------------
        # Combine all retrieved content
        # ------------------------------------------------

        context = "\n".join(
            context_parts
        )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = """
You are Kerala IT Hub, an AI-powered IT course
and institute information assistant.

Your job is to answer questions about IT and
technology courses in Kerala.

Use ONLY the information provided in the
retrieved context.

IMPORTANT RULES:

1. Do not invent course names, fees, durations,
   eligibility requirements, locations, admission
   details, curriculum details, or other course
   information.

2. Treat the retrieved information as evidence
   from the listed sources, not as complete
   information about every IT course in Kerala.

3. Never use words such as:
   "all", "only", "every", "none", "best",
   "most", or "the complete list"
   unless the retrieved information explicitly
   supports that claim.

4. If the question asks for a list of courses,
   present the courses found in the retrieved
   results.

   Clearly indicate that the list is based on
   the retrieved sources and may not represent
   every course available in Kerala.

5. If a course or institute was not found in the
   retrieved results, do not claim that it
   does not exist.

6. If the retrieved information does not contain
   enough evidence to answer the question,
   clearly say that the information was not found
   in the retrieved sources.

7. Do not combine eligibility, fees, duration,
   curriculum, or other details from different
   institutes and present them as if they belong
   to the same institute.

8. When information comes from multiple institutes,
   clearly associate each detail with the correct
   institute or course.

9. Do not treat marketing claims such as
   "best", "No.1", "guaranteed job", "100%
   placement", or similar promotional statements
   as verified facts.

10. Preserve uncertainty.

    If the source says that fees, duration,
    eligibility, or other details depend on
    the batch, program, or conditions, do not
    provide a fixed value.

11. Prefer specific information over general
    assumptions.

12. Keep the answer concise and easy to understand.

13. When listing courses, institutes, or programs,
    mention the associated institute or provider
    whenever that information is available.

14. Do not claim that the retrieved list is
    complete unless the source explicitly states
    that it is complete.

15. Do not infer information that is not directly
    supported by the retrieved content.

16. At the end of the answer, provide a
    "Sources" section containing the relevant
    source titles and URLs.

    List sources as plain numbered lines
    ("1. Title - URL"), one per line. Never format
    the Sources section as a markdown table (no "|"
    characters), and never mix source citations into
    the comparison table from rule 18 as if they were
    another institute row.

    List each distinct source URL only ONCE in this
    section, even if you drew on it for several rows,
    bullet points, or parts of the answer.

    Never write a URL, or a numbered citation like
    "1. Title - URL", anywhere else in the answer.
    The body of the answer (before "Sources") should
    read as plain prose, a table, or bullet points --
    never a list of links. If you want to credit a
    specific claim to a source while writing the body,
    name the institute or publication in words instead
    (e.g. "according to Coursera") rather than pasting
    its URL inline.

17. When presenting information from a source,
    do not change its meaning or create facts
    that are not present in the retrieved content.

18. Whenever the retrieved information covers two
    or more distinct institutes or courses, present
    them as a markdown table so they can be compared
    side by side, instead of separate paragraphs.

    Use a header row and include whichever of these
    columns the retrieved content actually supports:
    Institute, Course, Duration, Fees, Eligibility,
    Mode (Online/Offline/Hybrid), Location,
    Certification.

    Leave a cell blank (write "-") rather than
    guessing if that detail was not present in the
    retrieved content for that row.

    Never put one column's value into a different
    column -- e.g. if you only know the LOCATION but
    not the duration, write "-" in the Duration cell.
    Do not fill it with the location, the institute
    name, or any other value that belongs to a
    different column. A cell's value must specifically
    answer that column's own question, or be "-".

19. Be as detailed and specific as the retrieved
    content allows. Prefer including a concrete
    detail (an exact duration, fee, eligibility
    requirement, or admission step) over a vague
    summary, as long as that detail is explicitly
    present in the retrieved content.

19b. When comparing two career paths, fields, or
    course TYPES rather than specific institutes
    (e.g. "data science vs data analytics", where
    rule 18's table doesn't apply), structure the
    answer with a bold subheading for each side
    followed by its own bullet points, instead of
    interleaving both sides' points into one mixed
    list. For example:

    **Data Science covers:**
    - point
    - point

    **Data Analytics covers:**
    - point
    - point

    Keep every bullet under the side it actually
    belongs to -- do not mix a Data Science point
    into the Data Analytics section or vice versa.

20. If a previous conversation is provided, use it
    only to understand what the current question
    refers to (e.g. "it", "that one", "the second
    institute"). Still answer strictly from the
    retrieved course information provided for THIS
    question, not from memory of the previous answer.

    Do not repeat the previous answer's full content
    -- focus on what the current question actually
    asks, and refer back briefly ("as mentioned for
    X above") only where useful.

21. If the retrieved information is provided as
    "COURSE" blocks rather than "SOURCE" blocks, each
    COURSE block has already been deduplicated and
    merged from however many pages described it --
    treat each COURSE block as exactly one institute/
    course entity. Do not split a single COURSE block
    into more than one row or mention, and do not
    treat two different COURSE blocks as the same
    institute/course even if they look similar.

22. If the question includes an INTERNAL ADVISOR
    ANALYSIS section, that section is for your own
    reasoning only -- never copy its structure or
    labels into the answer. Do not use the literal
    words "Criterion", "Match?", or "Explanation" as
    labels, headers, or column names anywhere in your
    answer, and do not build a generic criteria/labels
    table. Write a natural, conversational summary of
    each course instead. Present the courses in the
    exact order given in that section -- do not re-rank
    them yourself. Only describe a preference (topic,
    level, location, budget, mode, or placement) as
    matched when that section actually confirms it for
    that course; if it says a detail is not available,
    say so plainly rather than calling it a match.
"""


    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    history_section = ""

    if history:

        history_lines = []

        for turn in history:

            history_lines.append(
                f"User: {turn.get('question', '')}"
            )

            history_lines.append(
                f"Assistant: {turn.get('answer', '')}"
            )

        history_section = (
            "Previous conversation (for context only -- "
            "the current question is answered strictly "
            "from the retrieved course information below, "
            "not from this history):\n\n"
            + "\n".join(history_lines)
            + "\n\n\n"
        )


    # ========================================================
    # QUERY INTENT NOTE (Step 5 Part 8)
    # ========================================================

    if advisor_active:
        query_intent_note = _build_advisor_context(course_records, advisor_preferences)
    else:
        query_intent_note = ""

    if not query_intent_note:
        query_intent_note = _build_query_intent_note(query_intent)

    query_intent_section = (
        f"{query_intent_note}\n\n\n" if query_intent_note else ""
    )


    # ========================================================
    # USER PROMPT
    # ========================================================

    user_prompt = f"""
{history_section}Question:

{query}


{query_intent_section}Retrieved course information:

{context}


Using only the retrieved course information,
answer the question accurately.

Remember:

- Do not invent information.
- Do not assume missing information.
- Clearly distinguish between different institutes.
- If the results are incomplete, say that they
  are based on the retrieved sources.
- Do not claim that a course does not exist simply
  because it was not found in the retrieved results.
- If two or more institutes or courses appear in the
  retrieved information, present them as a markdown
  comparison table (see rule 18).
- Include specific details (durations, fees,
  eligibility, etc.) wherever the retrieved content
  actually provides them (see rule 19).
"""


    # ========================================================
    # GROQ API CALL
    # ========================================================

    response = call_groq_with_retry(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0,

        # Without max_tokens, Groq reserves capacity for a very large
        # default completion length against the account's
        # tokens-per-minute limit -- on top of the actual prompt
        # tokens -- which was the real cause of 413 "Request too
        # large" errors even after capping context size.
        max_tokens=2048,

        # openai/gpt-oss-20b is a reasoning model: its internal
        # "thinking" tokens are billed as part of completion_tokens,
        # separate from the visible answer. At the default reasoning
        # effort, a harder comparison task was consuming the *entire*
        # max_tokens budget on reasoning alone, leaving nothing for
        # the actual answer (finish_reason: length, empty content).
        # "low" is enough for this task (a grounded lookup/comparison
        # over already-retrieved text, not novel problem-solving) and
        # confirmed fixing it: reasoning dropped to ~100-600 tokens,
        # finish_reason back to "stop", with a complete answer.
        reasoning_effort="low"
    )


    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    answer = (
        response
        .choices[0]
        .message
        .content
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "answer": answer,
        "sources": sources
    }