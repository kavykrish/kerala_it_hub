"""
Structured course extraction -- foundation layer.

Turns the raw, field-labelled chunks retrieved for ONE source page
(see web_retrieval.retriever.retrieve_field_aware_chunks) into a single
structured Course record for that page, using the schema already
defined in data/sources.json's "course_information_fields" (extended
with a couple of fields -- level, placement_information, source_title
-- that the schema didn't have yet, but no existing field names were
changed).

Extraction here is entirely deterministic (regex/keyword-based against
text that is already known to exist in the retrieved chunks) -- no LLM
call is made. This keeps the fix free of any extra Groq usage (the
account's tokens-per-minute budget is already tight, see
backend/rag_generator.py's retry logic) and satisfies the project's own
rule to prefer deterministic extraction when the retrieved text already
contains what's needed. parse_llm_course_json is provided as a
validated entry point for an LLM-assisted fallback, but nothing in the
current pipeline calls it yet.

This module does NOT merge records across sources/pages -- each
Course record stays tied to exactly the one source_url it came from.
Cross-source institute matching/merging is the next step, not this one.
"""

import json
import os
import re
from urllib.parse import urlsplit

from web_retrieval.chunker import HEADINGS
from web_retrieval.retriever import detect_field_category
from web_retrieval.kerala_locations import KERALA_LOCATIONS


# ============================================================
# COURSE SCHEMA
# ============================================================
# Matches data/sources.json's course_information_fields where the name
# already existed there (institute_name, course_name, course_category,
# duration, eligibility, fees, location, learning_mode, curriculum,
# certification, admission_information, source_url, source_priority).
# "level", "placement_information" and "source_title" are new -- they
# didn't exist in sources.json's field list, so nothing there needed to
# change. sources.json also has "course_status", "official_website" and
# "source_type", which aren't produced here since they're not part of
# the extraction this step covers.

NOT_AVAILABLE = "Not available"

COURSE_FIELDS = (
    "institute_name",
    "course_name",
    "course_category",
    "level",
    "duration",
    "eligibility",
    "fees",
    "location",
    "learning_mode",
    "curriculum",
    "certification",
    "admission_information",
    "placement_information",
    "source_url",
    "source_title",
    "source_priority",
)

# Fields whose value is a list (currently just curriculum) rather than
# a plain string.
_LIST_FIELDS = ("curriculum",)

# Which structured field each field-aware retrieval category (see
# web_retrieval/retriever.py's FIELD_LABEL_CATEGORIES) fills.
_CATEGORY_TO_FIELD = {
    "duration": "duration",
    "eligibility": "eligibility",
    "fees": "fees",
    "learning_mode": "learning_mode",
    "location": "location",
    "certification": "certification",
    "admission_information": "admission_information",
    "placement_information": "placement_information",
    "curriculum": "curriculum",
    "course_name": "course_name",
    "course_category": "course_category",
}

# Fields extracted with a strict pattern -- a real match, or
# "Not available". Never a truncated blob of surrounding text (see the
# extractor functions below, one per field).
_PATTERN_FIELDS = ("duration", "fees", "learning_mode", "location")

# Fields that are inherently free-text (a real sentence, not a fixed
# pattern) but still must not become a whole marketing paragraph --
# capped to the first sentence, see _extract_first_sentence.
_SENTENCE_FIELDS = (
    "eligibility", "certification", "admission_information",
    "placement_information",
)

_SENTENCE_FIELD_CHAR_LIMIT = 200


def empty_course_record(
    source_url: str = "",
    source_title: str = "",
    source_priority: str = "unknown"
):
    """
    A fully-shaped Course record with every field set to
    "Not available" (or [] for list fields) except source metadata.
    Never omits a key -- callers can always safely index every field
    in COURSE_FIELDS.
    """

    record = {field: NOT_AVAILABLE for field in COURSE_FIELDS}

    for field in _LIST_FIELDS:
        record[field] = []

    record["source_url"] = source_url or ""
    record["source_title"] = source_title or NOT_AVAILABLE
    record["source_priority"] = source_priority or "unknown"

    return record


# ============================================================
# SOURCE PRIORITY
# ============================================================
# Reuses the priority scale already defined in data/sources.json's
# source_policy (government=1, university=2,
# government_skill_development=3, technical_education=4,
# official_institute=5, trusted_education_platform=6) rather than
# inventing a new one. sources.json's own entries are human-readable
# category descriptions with no url/domain field to match against --
# it explicitly says official institute sources are "discovered
# dynamically" -- so this is a best-effort heuristic bridge from a
# real URL to that existing scale, not a literal lookup table.

_SOURCES_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sources.json"
)


def _load_source_policy():

    try:
        with open(_SOURCES_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)

    except (OSError, json.JSONDecodeError):
        return {}


SOURCE_POLICY = _load_source_policy()

_GOV_DOMAIN_HINTS = (".gov.in", ".nic.in")
_SKILL_DEV_DOMAIN_HINTS = ("asapkerala", "kite.kerala.gov.in")
_TECHNICAL_EDU_DOMAIN_HINTS = ("sbte", "dtekerala", "itikerala")
_UNIVERSITY_DOMAIN_HINTS = (
    "keralauniversity", "cusat.ac.in", "mgu", "uoc.ac.in",
    "calicutuniversity", "kannuruniversity", "duk.ac.in",
    "cukerala.ac.in", "ktu.edu.in", ".ac.in"
)
_THIRD_PARTY_PLATFORM_HINTS = (
    "coursera.org", "udemy.com", "simplilearn.com", "edx.org",
    "analyticsvidhya.com", "geeksforgeeks.org", "shiksha.com",
    "collegedunia.com", "careers360.com", "naukri.com",
    "upgrad.com", "greatlearning.in", "internshala.com"
)


def determine_source_priority(url: str) -> str:
    """
    Best-effort classification of a source URL against the priority
    scale already defined in data/sources.json, so a later merge step
    can prefer a higher-authority source when two sources disagree on
    a field. Falls back to priority "5" (official_institute_sources,
    the JSON's own catch-all for a dynamically-discovered institute
    site) since most search results for a specific course ARE an
    institute's own page, not a government/university/aggregator one.
    """

    if not url:
        return "unknown"

    lowered = url.lower()

    # Specific known categories are checked before the generic .gov.in
    # / .ac.in catch-alls, since e.g. ASAP Kerala's own domain IS a
    # .gov.in domain but sources.json specifically categorizes it under
    # government_skill_development (priority 3), not plain government
    # (priority 1) -- the specific match should win.
    if any(hint in lowered for hint in _SKILL_DEV_DOMAIN_HINTS):
        return "3"

    if any(hint in lowered for hint in _TECHNICAL_EDU_DOMAIN_HINTS):
        return "4"

    if any(hint in lowered for hint in _UNIVERSITY_DOMAIN_HINTS):
        return "2"

    if any(hint in lowered for hint in _GOV_DOMAIN_HINTS):
        return "1"

    if any(hint in lowered for hint in _THIRD_PARTY_PLATFORM_HINTS):
        return "6"

    return "5"


# ============================================================
# COURSE CATEGORY / LEVEL KEYWORDS
# ============================================================
# course_category is detected from data/course_keywords.json, which
# (per the audit) existed but was never actually loaded by any code --
# this is the first thing that puts it to use. Matching is a literal
# substring check against phrases that are already in the JSON, so a
# match is always something that literally appears in the retrieved
# text, never an invented category.

_COURSE_KEYWORDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "course_keywords.json"
)


def _load_course_keywords():

    try:
        with open(_COURSE_KEYWORDS_PATH, encoding="utf-8") as f:
            return json.load(f)

    except (OSError, json.JSONDecodeError):
        return {}


COURSE_KEYWORDS = _load_course_keywords()

_LEVEL_LABELS = {
    "beginner": "Beginner",
    "fresher": "Beginner",
    "freshers": "Beginner",
    "no prior experience": "Beginner",
    "no coding experience": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced",
    "professionals": "Advanced",
    "working professionals": "Advanced",
    "experienced": "Advanced",
}


def _guess_course_category(course_name):
    """
    Without an explicit "Category"/"Course Category" heading (handled
    separately, in the field_values merge loop), the only remaining
    safe evidence for course_category is the already-resolved
    course_name itself -- NOT an independent "longest keyword found
    anywhere in the page's retrieved text" scan.

    That scan previously picked whichever known keyword was textually
    LONGER, regardless of whether it had anything to do with the
    actual course -- a page whose course_name correctly resolved to
    "Data Science" (via the careful H1/page_context logic) could still
    get course_category = "Artificial Intelligence" just because that
    phrase happened to appear, once, in an unrelated sentence
    elsewhere on the page, and is a longer string. Confirmed as a real
    failure against a real page during the Step 4 audit.

    course_name is a short, already-vetted string (not an entire page
    of raw text), so searching WITHIN it for the longest known keyword
    is safe and scoped -- e.g. "Data Science with Machine Learning" ->
    "Machine Learning" is a legitimate category read off a compound
    course name that genuinely includes both subjects, not a random
    unrelated aside.
    """

    if not course_name or course_name == NOT_AVAILABLE:
        return None

    return _find_best_keyword_match([course_name])


def _guess_level(chunks):

    combined = " ".join(chunk for chunk in chunks if chunk).lower()

    for keyword, label in _LEVEL_LABELS.items():

        if keyword in combined:
            return label

    return None


# ============================================================
# INSTITUTE NAME
# ============================================================
# A page/article TITLE is not reliably the institute name -- SEO
# titles are commonly a marketing headline ("Best Data Science Course
# in Kerala") or a listicle-style subtitle ("... -- Scope, Skills &
# Jobs 2026") with no real organization name in it at all, and blindly
# trusting either half of a title split on "|"/"-" produced exactly
# that kind of garbage. source_title is kept as-is for display
# separately; it is never used as a source for institute_name here.
#
# Two real signals are used instead, strongest first:
#
# 1. The organization's own name mentioned in the page's CONTENT,
#    recognized by a capitalized phrase immediately followed by a
#    common organization-type word ("Hub", "Institute", "Academy",
#    "Technolab", ...). A page describing itself ("Codeme Hub offers
#    ...") is the strongest evidence there is.
# 2. The site's own domain name, de-slugified (e.g. "codemehub.com"
#    -> "Codeme Hub", "rogersoft.com" -> "Rogersoft"). An institute's
#    own website domain is almost always its own brand name -- far
#    more reliable than free-text title guessing, though not perfect
#    for a third-party blog/aggregator domain that merely writes
#    ABOUT courses rather than teaching one itself.
#
# If neither signal fires, institute_name stays "Not available" --
# never a guess from the title.

# Deliberately excludes generic tech/CS words that collide constantly
# with ordinary course-content phrases and were confirmed, by testing
# against real search results, to produce false-positive matches:
# "Learning" ("Machine Learning", "Deep Learning"), "Systems"
# ("Recommender Systems", "Operating Systems", "Database Systems"),
# "Technologies" ("Web Technologies", "Cloud Technologies"),
# "Foundation" ("Programming Foundations" course-module naming),
# "Centre"/"Center" ("Data Center"). Kept to words that are reliably
# organization-type words in practice, not also common curriculum
# vocabulary.
# "Classes" removed (Step 4 audit): "[Qualifier] Classes" is
# extremely common MODE/SCHEDULE phrasing ("Online Classes", "Weekend
# Classes", "Evening Classes") that reliably collides with this
# pattern and was confirmed, against real page content, to produce a
# mode phrase as an "institute name". _MODE_SCHEDULE_QUALIFIER_WORDS
# below is a second, more general layer of the same protection -- in
# case a different suffix word ever collides with a mode/schedule
# qualifier the same way.
_ORG_SUFFIX_WORDS = (
    "Institute", "Academy", "Technolab", "College", "University",
    "Hub", "School", "Consultancy",
    "Proschool", "Edutech", "Infotech", "Bootcamp",
)

# An org-mention candidate whose leading qualifier word is itself a
# known mode/schedule term is rejected outright, regardless of which
# suffix word follows it -- this is the general rule (not a "Online
# Classes" special case): any "[mode word] [suffix word]" phrase is
# describing HOW a course is delivered, not naming who delivers it.
_MODE_SCHEDULE_QUALIFIER_WORDS = {
    "online", "offline", "hybrid", "weekend", "weekday", "weekdays",
    "evening", "morning", "flexible", "live", "self-paced",
    "part-time", "full-time", "virtual",
}

_CAPWORD = r"[A-Z][A-Za-z&\.]*"

# [ \t]+ (not \s+) between words on purpose -- an organization's own
# name is always written on one line/sentence fragment in practice, so
# this stops the match from crossing a newline and stitching together
# unrelated capitalized words from two different lines (e.g. the last
# word of one curriculum bullet plus the first word of the next).
_ORG_MENTION_PATTERN = re.compile(
    r"\b((?:" + _CAPWORD + r"[ \t]+){0,3}(?:"
    + "|".join(re.escape(w) for w in _ORG_SUFFIX_WORDS)
    + r"))\b"
)

# Only used to decide whether a single-word domain label ends in an
# org-type word worth splitting off (e.g. "codemehub" -> "Codeme" +
# "Hub") -- lowercase forms of the same list above.
_ORG_SUFFIX_WORDS_LOWER = tuple(w.lower() for w in _ORG_SUFFIX_WORDS)

# When the registrable domain's own TLD-ish segment is itself a real
# word (site is literally "<brand>.academy", "<brand>.courses", ...),
# it's worth folding into the guessed name.
_MEANINGFUL_DOMAIN_SUFFIX_WORDS = {
    "academy", "courses", "training", "education", "school",
}


# A capitalized sentence-starter that can precede the real name
# without being part of it ("At Codeme Hub, we offer..." -> "At" isn't
# part of the name). Stripped from the front of a match, not rejected
# outright, since the rest of the candidate is still usable.
# "thankyou"/"thanks"/"thank" confirmed necessary against a real page
# during Step 3E testing: a student testimonial ending "...Thankyou
# Blitz Academy." otherwise produced "Thankyou Blitz Academy" as a
# candidate DISTINCT from the page's other, correct "Blitz Academy"
# mentions, which made _looks_like_third_party_comparison_page count
# 2 "different" organizations on a genuine single-institute page and
# wrongly suppress its institute_name.
_LEADING_STOPWORDS = {
    "at", "join", "the", "our", "we", "in", "best", "top",
    "welcome", "about", "visit", "thankyou", "thanks", "thank",
}


def _find_org_mentions(chunk_texts):
    """
    Every DISTINCT capitalized-phrase-plus-org-suffix-word candidate
    found anywhere in the page's retrieved chunks (e.g. "Codeme Hub",
    "Luminar Technolab"). Used both to pick the single best candidate
    (_find_org_mention) and to detect a page that names more than one
    organization -- a strong sign it's a third-party comparison/
    listing page rather than one institute's own site (see
    _looks_like_third_party_comparison_page).
    """

    combined = " ".join(chunk for chunk in chunk_texts if chunk)

    candidates = set()

    for match in _ORG_MENTION_PATTERN.finditer(combined):

        candidate = re.sub(r"\s+", " ", match.group(1)).strip()

        words = candidate.split()

        while words and words[0].lower() in _LEADING_STOPWORDS:
            words.pop(0)

        # Reject a bare org-suffix word with nothing in front of it
        # ("The Institute") -- not a real name on its own.
        if len(words) < 2:
            continue

        # Reject "[mode word] [suffix word]" ("Online Classes",
        # "Weekend Classes") -- describes how a course is delivered,
        # not who delivers it. See _MODE_SCHEDULE_QUALIFIER_WORDS.
        if words[0].lower() in _MODE_SCHEDULE_QUALIFIER_WORDS:
            continue

        candidates.add(" ".join(words))

    return candidates


def _find_org_mention(chunk_texts):
    """Longest of _find_org_mentions's candidates, or None."""

    candidates = _find_org_mentions(chunk_texts)

    if not candidates:
        return None

    return max(candidates, key=len)


# ============================================================
# THIRD-PARTY COMPARISON / LISTICLE PAGE DETECTION
# ============================================================
# A page's own domain looking innocuous (even a plausible-sounding
# "<brand>.academy") doesn't mean it's actually that institute's own
# course page -- it can just as easily be a third-party blog reviewing
# OTHER institutes' courses. Trusting the domain/content-mention as
# "the" institute in that case attributes a real institute's identity,
# or an invented one, to the wrong page. These are best-effort textual
# signals, not a hard-coded list of known blog domains.

# Requires a genuine list-sized number (2+) -- "No.1 Best Data Science
# Course" / "#1 Training Institute" are superlative ranking CLAIMS
# about a single institute, not a listicle of several, and matching
# them here was incorrectly suppressing a real institute's own page.
# A real listicle title always has 2+ items ("10 Best ...", "Top 5
# ...").
_LISTICLE_NUMBER = r"(?:[2-9]|\d{2,})"

_LISTICLE_TITLE_PATTERN = re.compile(
    r"\b(" + _LISTICLE_NUMBER + r"\s+best|top\s*" + _LISTICLE_NUMBER
    + r"|best\s*" + _LISTICLE_NUMBER + r")\b",
    re.IGNORECASE
)

_COMPARISON_PAGE_PHRASES = (
    "best courses", "best institutes", "best colleges",
    "top institutes", "top courses", "top colleges",
    "comparison", "compare ", " vs ", " vs.", "which is better",
    "review of", "list of institutes", "list of courses",
)


def _looks_like_third_party_comparison_page(chunk_texts, source_title):
    """
    Best-effort signal that a page discusses MULTIPLE institutes/
    courses (a listicle, comparison, or review article) rather than
    being one institute's own page. When true, institute_name is left
    "Not available" regardless of what a single content-mention or the
    domain name would otherwise suggest -- see
    _guess_institute_name.
    """

    title_lower = (source_title or "").lower()

    if _LISTICLE_TITLE_PATTERN.search(title_lower):
        return True

    if any(phrase in title_lower for phrase in _COMPARISON_PAGE_PHRASES):
        return True

    combined_lower = " ".join(chunk for chunk in chunk_texts if chunk).lower()

    if any(phrase in combined_lower for phrase in _COMPARISON_PAGE_PHRASES):
        return True

    # Multiple DISTINCT organization names mentioned in the content is
    # itself strong evidence this page is describing several
    # institutes, not just its own.
    if len(_find_org_mentions(chunk_texts)) >= 2:
        return True

    return False


def _institute_name_from_domain(url):
    """
    De-slugifies the site's own domain into a plausible display name,
    but ONLY when there's a genuine word boundary to split on --
    either a hyphen ("example-institute.com" -> "Example Institute")
    or a recognized organization-suffix word the label ends with
    ("codemehub.com" -> "Codeme Hub").

    A single-word label with NEITHER (e.g. "stthomas" from
    stthomas.ac.in, or "rogersoft" from rogersoft.com) has no reliable
    word boundary at all -- it could be one legitimate brand word, or
    it could be two real words concatenated with nothing to mark where
    ("St" + "Thomas"). There is no general way to tell these apart
    from the domain text alone, and confirmed against a real page
    (stthomas.ac.in), blindly capitalizing the whole label produces a
    wrong-looking, unreadable result ("Stthomas"). Per the project's
    own rule -- wrong information is worse than missing information --
    this now returns None for EVERY such label, not just ones that
    happen to look bad. This is a deliberate, known trade-off: a
    single-word domain that genuinely was already a fine brand name on
    its own (no evidence found either way from domain text alone) now
    also returns None instead of a lucky-guess capitalization.
    """

    if not url:
        return None

    host = urlsplit(url).netloc.lower()
    host = re.sub(r"^www\.", "", host)

    if not host:
        return None

    host_parts = [p for p in host.split(".") if p]

    if not host_parts:
        return None

    label = host_parts[0]
    tld_word = host_parts[1] if len(host_parts) > 1 else ""

    if not label:
        return None

    words = re.split(r"[-_]+", label)

    if len(words) == 1:

        word = words[0]
        suffix_found = False

        for suffix in _ORG_SUFFIX_WORDS_LOWER:

            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                words = [word[: -len(suffix)], suffix]
                suffix_found = True
                break

        if not suffix_found:
            return None

    display = " ".join(w.capitalize() for w in words if w)

    if not display:
        return None

    if tld_word in _MEANINGFUL_DOMAIN_SUFFIX_WORDS:
        display = f"{display} {tld_word.capitalize()}"

    return display


def _guess_institute_name(chunk_texts, source_url, source_priority, source_title):
    """
    Two independent reasons institute_name can correctly come back
    "Not available" rather than a guess:

    - source_priority "6" (trusted_education_platform -- see
      determine_source_priority) means this source is already known
      by its own domain to be a third-party discovery/comparison
      platform (collegedunia, shiksha, careers360, ...), never an
      institute itself.
    - _looks_like_third_party_comparison_page catches the harder case:
      a source whose domain looks perfectly innocuous (even a
      plausible "<brand>.academy") but whose title/content shows it's
      actually a blog/listicle discussing OTHER institutes' courses.

    In either case, picking whichever name a content-mention or the
    domain happens to suggest would misattribute a real institute's
    identity, or an invented one, to the wrong page -- leaving it
    unavailable is the correct, safer answer.
    """

    if source_priority == "6":
        return None

    if _looks_like_third_party_comparison_page(chunk_texts, source_title):
        return None

    mention = _find_org_mention(chunk_texts)

    if mention:
        return mention

    return _institute_name_from_domain(source_url)


# ============================================================
# COURSE NAME
# ============================================================
# course_name is only ever set from:
#
# Priority A -- an explicit "Course Name" / "Program Name" / "Course
# Title" heading on the page, if one exists (see chunker.HEADINGS).
# Rare, but unambiguous when present. Handled where field_values are
# assembled, in extract_course_record itself.
#
# Priority B -- the page's own <h1> (see
# web_retrieval.page_reader.fetch_page_structured), run through
# normalize_h1_to_course_name below. Step 3D's forensic investigation
# found that two real pages (collegedunia.com, blitzacademy.org) both
# have an unambiguous <h1> naming "Data Science" -- but the pipeline
# discarded it entirely (trafilatura flattens heading hierarchy into
# plain paragraph text), so course_name fell through to a keyword
# match against the wrong prose sentence, which happened to mention
# "Artificial Intelligence" as a related field.
#
# Priority C -- a JSON-LD identity signal on the page (breadcrumb or
# Article/Course headline), if the page has one and B found nothing.
# Also run through normalize_h1_to_course_name -- an Article's
# headline is typically identical text to the page's own <h1>.
#
# Priority D -- the first line, in the page's OWN reading order, that
# contains a known subject keyword from data/course_keywords.json (see
# page_context below). This is deliberately NOT "the longest keyword
# found anywhere on the page": a page about "Data Science" commonly
# also mentions "Artificial Intelligence", "Machine Learning", etc. in
# its curriculum list or in a descriptive sentence about what Data
# Science covers, and those are LONGER strings than "Data Science" --
# picking the longest match regardless of where it appeared previously
# produced "Artificial Intelligence" as the course name for a page
# that is actually about Data Science. Stopping at the FIRST matching
# LINE (almost always the page's own heading/opening sentence, since
# page_context is a bounded prefix of the page's own cleaned text in
# original order) fixed the common case; it can still be fooled when
# that first line is itself one long flowing paragraph mentioning a
# longer keyword before/alongside the real subject -- which is
# precisely why B/C (structural signals) now take priority over it.
#
# Priority E -- if nothing above found anything, course_name stays
# "Not available". There is deliberately no further fallback to
# searching retrieved/curriculum chunks -- a wrong course name is
# worse than a missing one (see this module's top-level docstring).


# Words that, as the FIRST word of an <h1>/JSON-LD headline, make the
# the words PRECEDING the matched subject keyword a value judgment
# about a course rather than a neutral identification of one ("Best
# Data Science Course in Kerala" is a marketing claim, not a course's
# actual name) -- such a heading is treated as unreliable rather than
# stripped down to the keyword inside it. A genuine level/scope
# qualifier ("Advanced", "Beginner") is NOT in this list -- those are
# legitimate parts of a course's identity and are preserved, not
# rejected (see normalize_h1_to_course_name's docstring). Checked
# against every word in the prefix, not just the heading's first word
# -- an FAQ-style heading ("What is the best Data Science Course...")
# has the hype word several words in, not first.
_H1_HYPE_PREFIX_WORDS = {
    "best", "top", "no.1", "no1", "#1", "leading", "premier",
    "finest", "number", "no",
}

# A call-to-action verb opening the heading ("Become a Data Science
# Pro with...") is an instruction to the reader, not a course
# identification -- confirmed against a real page (futurixacademy.com)
# during Step 3E testing, where "Become a" would otherwise have been
# kept as a "qualifier" prefix.
_H1_CTA_VERBS = {
    "become", "join", "get", "start", "enroll", "enrol", "master",
    "discover", "explore", "unlock", "achieve", "build",
}

# A heading phrased as a question ("What is the best Data Science
# Course in Kerala for beginners?") isn't identifying a course, it's
# asking about one -- the words leading up to the subject keyword are
# the question itself, not a qualifier.
_H1_QUESTION_WORDS = {"what", "why", "how", "which", "who", "when", "where"}

# A prefix longer than this many words no longer reads as a simple
# qualifier ("Advanced", "Introduction to") -- it's a full clause, and
# trusting it risks keeping question/marketing language verbatim.
_H1_MAX_RELIABLE_PREFIX_WORDS = 3

# A connector immediately after the primary keyword, leading straight
# into a SECOND recognized keyword, is kept ("Data Science with
# Machine Learning") -- anything else that follows (a generic
# "Course"/"Courses in <location>" trailer, a "|"-separated site
# brand, a ":"-separated subtitle list) is dropped rather than guessed
# at.
_H1_CONNECTOR_WORDS = ("with", "and", "&")


def _h1_prefix_is_reliable(prefix):
    """
    Whether the text BEFORE a matched subject keyword in an H1/JSON-LD
    heading reads like a simple, trustworthy qualifier -- see
    normalize_h1_to_course_name's docstring for the reasoning and a
    real example that motivated each check.
    """

    if not prefix:
        return True

    words = [w.strip(".,:?!#") for w in prefix.split(" ") if w.strip()]

    if not words:
        return True

    if len(words) > _H1_MAX_RELIABLE_PREFIX_WORDS:
        return False

    lowered_words = [w.lower() for w in words]

    if lowered_words[0] in _H1_QUESTION_WORDS:
        return False

    if lowered_words[0] in _H1_CTA_VERBS:
        return False

    if any(w in _H1_HYPE_PREFIX_WORDS for w in lowered_words):
        return False

    return True


def _find_leading_keyword(text):
    """Longest known keyword that TEXT starts with, or None."""

    lowered = text.lower().strip()

    best = None

    for phrases in COURSE_KEYWORDS.values():

        for phrase in phrases:

            if lowered.startswith(phrase.lower()):

                if best is None or len(phrase) > len(best):
                    best = phrase

    return best


def normalize_h1_to_course_name(heading_text):
    """
    Deterministically turns a raw <h1> (or JSON-LD headline/breadcrumb
    text) into a course_name, or None if it doesn't reliably identify
    one. Never a blind copy of the whole heading -- see the module
    docstring's H1 examples.

    - Finds the first known subject keyword (from
      data/course_keywords.json) and keeps everything from the start
      of the heading through that keyword -- this is what preserves a
      genuine qualifier like "Advanced" in "Advanced Data Science"
      rather than stripping it down to "Data Science".
    - Rejects the match if what precedes the keyword doesn't read like
      a simple qualifier: more than 3 words, or containing a marketing
      superlative ("Best", "Top", "No.1", "Leading", ...) or a
      question word ("What", "Why", "How", ...) anywhere in it. A real
      FAQ-style heading like "What is the best Data Science Course in
      Kerala for beginners?" would otherwise keep the entire question
      clause as a prefix -- confirmed against a real page during
      testing (synnefo.academy's <h1> is exactly this pattern).
    - Extends past the keyword only when it's immediately followed by
      a connector ("with"/"and"/"&") leading straight into ANOTHER
      recognized keyword ("Data Science with Machine Learning").
      Anything else that follows is dropped.
    - None if no known keyword is found in the heading at all -- never
      falls back to returning the raw heading text.
    """

    if not heading_text or not heading_text.strip():
        return None

    text = re.sub(r"\s+", " ", heading_text).strip()

    lowered = text.lower()

    best_keyword = None
    best_start = None
    best_end = None

    for phrases in COURSE_KEYWORDS.values():

        for phrase in phrases:

            idx = lowered.find(phrase.lower())

            if idx == -1:
                continue

            # Prefer whichever known keyword appears CLOSEST to the
            # start of the heading -- that's the page's own primary
            # subject framing, not a later aside.
            if best_start is None or idx < best_start:
                best_keyword = phrase
                best_start = idx
                best_end = idx + len(phrase)

    if best_keyword is None:
        return None

    prefix = text[:best_start].strip()

    if not _h1_prefix_is_reliable(prefix):
        return None

    result = f"{prefix} {best_keyword}".strip() if prefix else best_keyword

    remainder = text[best_end:].lstrip()
    remainder_lower = remainder.lower()

    for connector in _H1_CONNECTOR_WORDS:

        if not remainder_lower.startswith(connector + " "):
            continue

        after_connector = remainder[len(connector):].strip()
        second_keyword = _find_leading_keyword(after_connector)

        if second_keyword:
            result = f"{result} {connector} {second_keyword}"

        break

    return result


def _guess_course_name_from_page_context(page_context):
    """
    page_context: a bounded prefix of the page's OWN cleaned text, in
    original reading order (see mcp_server/course_search_server.py's
    page_context_by_source) -- never source_title, and never the
    reordered chunks retrieve_field_aware_chunks returns.
    """

    if not page_context:
        return None

    for line in page_context.splitlines():

        line = line.strip()

        if not line:
            continue

        match = _find_best_keyword_match([line])

        if match:
            return match

    return None


def _find_best_keyword_match(text_list):
    """
    Longest literal keyword match (from data/course_keywords.json)
    found in the given text. Used by _guess_course_name_from_page_context
    (scored per-line, first matching line wins -- see above) and by
    _guess_course_category (scored across all retrieved content, since
    a course's category is reasonably inferred from everything the
    page covers, unlike its specific name).
    """

    combined = " ".join(t for t in text_list if t).lower()

    best_match = None
    best_length = 0

    for phrases in COURSE_KEYWORDS.values():

        for phrase in phrases:

            if phrase.lower() in combined and len(phrase) > best_length:
                best_match = phrase
                best_length = len(phrase)

    return best_match


# ============================================================
# FIELD-SPECIFIC PATTERN EXTRACTORS
# ============================================================
# Each of these looks for an actual, recognizable value inside a
# field's stripped chunk text and returns None if nothing confidently
# matches -- never the raw surrounding text. This is what stops
# "Flexible schedule based on offline & online classes" from becoming
# a "duration", or a whole marketing paragraph from becoming a
# "learning_mode".

_DURATION_PATTERN = re.compile(
    r"\b(\d{1,3}(?:\.\d+)?)\s*[-–]?\s*"
    r"(hours?|hrs?|weeks?|months?|years?|days?)\b",
    re.IGNORECASE
)


def _extract_duration(text):

    match = _DURATION_PATTERN.search(text)

    if not match:
        return None

    return f"{match.group(1)} {match.group(2)}"


_FEES_PATTERN = re.compile(
    r"(?:₹|Rs\.?|INR)\s*[\d][\d,]*(?:\.\d+)?",
    re.IGNORECASE
)


def _extract_fees(text):

    match = _FEES_PATTERN.search(text)

    if not match:
        return None

    return match.group(0).strip()


# Order matters -- more specific patterns are checked first, so
# "24x7 LMS Access" is recognized as self-paced BEFORE the generic
# "online" check would otherwise miss it, and isn't mislabeled as live
# online training (see Step 3B item 5's explicit example).
_MODE_RULES = (
    (
        re.compile(r"\bself[\s-]?paced\b", re.IGNORECASE),
        "LMS / Self-paced",
    ),
    (
        re.compile(r"\bLMS\b", re.IGNORECASE),
        "LMS / Self-paced",
    ),
    (
        re.compile(
            r"\bonline\s*(?:&|and)\s*offline\b|\bhybrid\b",
            re.IGNORECASE
        ),
        "Online & Offline",
    ),
    (
        re.compile(r"\blive\s+(?:training|class(?:es)?|session)", re.IGNORECASE),
        "Live Training",
    ),
    (
        re.compile(r"\bonline\b", re.IGNORECASE),
        "Online",
    ),
    (
        re.compile(
            r"\boffline\b|\bclassroom\b|\bin[\s-]?person\b",
            re.IGNORECASE
        ),
        "Offline",
    ),
)


def _extract_learning_mode(text):

    for pattern, label in _MODE_RULES:

        if pattern.search(text):
            return label

    return None


_LOCATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(loc) for loc in KERALA_LOCATIONS) + r")\b"
    r"(?:\s*,\s*Kerala\b)?",
    re.IGNORECASE
)


def _extract_location(text):

    match = _LOCATION_PATTERN.search(text)

    if not match:
        return None

    return match.group(0).strip()


# A period after one of these (case-insensitive) doesn't end a
# sentence -- confirmed against a real page (collegedunia.com's FAQ
# block "Ques. What are the eligibility requirements...") where the
# period in "Ques." was being read as a complete sentence, producing
# "Ques." as the entire extracted value.
_SENTENCE_ABBREVIATIONS = {
    "ques", "ans", "q", "a", "mr", "mrs", "dr", "no", "vs", "etc",
    "eg", "ie", "prof", "st", "jr", "sr", "no1",
}


def _extract_first_sentence(text, max_chars=_SENTENCE_FIELD_CHAR_LIMIT):
    """
    Used for the free-text fields (eligibility, certification,
    admission_information, placement_information): the first sentence
    of the stripped field value, so a real one-line answer like "Any
    graduate" is preserved as-is, while a marketing paragraph is cut
    at its first sentence boundary rather than dumped whole.

    Skips a sentence-ending punctuation mark that's actually part of a
    known abbreviation ("Ques.", "Ans.", "Dr.", ...) rather than
    stopping there -- otherwise "Ques. What are the eligibility
    requirements...?" gets read as the one-word "sentence" "Ques."
    """

    text = text.strip()

    if not text:
        return None

    sentence = text

    for match in re.finditer(r"[.!?](?:\s|$)", text):

        preceding = text[: match.start()]
        word_match = re.search(r"([A-Za-z]+)$", preceding)
        preceding_word = word_match.group(1).lower() if word_match else ""

        if preceding_word in _SENTENCE_ABBREVIATIONS:
            continue

        sentence = text[: match.end()].strip()
        break

    if len(sentence) > max_chars:

        truncated = sentence[:max_chars]
        last_space = truncated.rfind(" ")

        sentence = truncated[:last_space] if last_space > 0 else truncated

    return sentence or None


# ============================================================
# FIELD VALUE EXTRACTION
# ============================================================

_HEADING_WORD_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in HEADINGS) + r")\b",
    re.IGNORECASE
)

_HEADING_SEPARATOR_PATTERN = re.compile(r"\s*[:\-–]\s*")


def strip_heading_prefix(chunk_text: str) -> str:
    """
    Removes the leading field heading from a chunk -- but ONLY when it
    is genuinely functioning as a label: a standalone heading line
    ("Duration\\n3 months"), or an inline "Label: Value" /
    "Label - Value" (colon/dash separator required for the inline
    form). Uses the exact same heading vocabulary the chunker split on
    (web_retrieval.chunker.HEADINGS), so this is guaranteed consistent
    with how the chunk was produced rather than guessing separately.

    A heading word immediately followed by MORE TEXT ON THE SAME LINE
    with no colon/dash is left completely untouched -- that's just the
    start of an ordinary sentence that happens to begin with the same
    word, not a label ("Certificate of completion is issued upon
    passing the final exam" must keep "Certificate", not become the
    grammatically broken "of completion is issued..."; confirmed
    against a real page where this exact pattern, "Certificate of
    completion...", is how a genuine certification section begins).
    """

    if not chunk_text or not chunk_text.strip():
        return ""

    lines = [line for line in chunk_text.strip().splitlines() if line.strip()]

    if not lines:
        return ""

    first_line = lines[0]
    match = _HEADING_WORD_PATTERN.match(first_line)

    if not match:
        return chunk_text.strip()

    remainder_of_line = first_line[match.end():]
    separator_match = _HEADING_SEPARATOR_PATTERN.match(remainder_of_line)

    if separator_match:
        value_on_first_line = remainder_of_line[separator_match.end():].strip()

    elif not remainder_of_line.strip():
        # The heading word is the entire first line -- a genuine
        # standalone heading, with the value (if any) on later lines.
        value_on_first_line = ""

    else:
        # Heading word immediately continues into more text with no
        # separator -- just an ordinary sentence, not a label. Leave
        # the whole chunk untouched.
        return chunk_text.strip()

    rest_lines = [line.strip() for line in lines[1:]]

    body = ([value_on_first_line] if value_on_first_line else []) + rest_lines

    return "\n".join(body).strip()


# Dispatches a category's combined stripped chunk text to its
# field-specific extractor (see "FIELD-SPECIFIC PATTERN EXTRACTORS"
# above). Returns None (never a raw text blob) when nothing
# confidently matches.
# certification/admission/placement each have a genuine heading
# ("Certification", "Admission", "Placement Assistance") on some real
# pages, but the CONTENT sitting under that heading (as trafilatura
# flattens the page) can drift into unrelated marketing copy -- a
# stats-badge widget and a generic "why choose us" paragraph, for a
# real page, confirmed to have zero words related to an actual
# credential anywhere in it. Requiring the extracted value to contain
# at least one field-specific corroborating word is what stops that
# from being accepted as if it answered the field at all. eligibility
# doesn't get the same treatment -- it has no comparably narrow shared
# vocabulary (a real eligibility answer could be "graduates only",
# "12th pass", "no prior experience", "open to all", ...), and its
# specific known failure mode (the "Ques."/FAQ-block false match) is
# already fixed at the source by detect_field_category no longer
# matching that chunk as "eligibility" in the first place.
_CERTIFICATION_EVIDENCE_WORDS = (
    "certificate", "certification", "certified", "credential",
    "credentials", "accredited", "accreditation", "recognized",
    "recognised", "qualification",
)

_ADMISSION_EVIDENCE_WORDS = (
    "admission", "apply", "application", "enroll", "enrol",
    "enrollment", "enrolment", "registration", "register",
)

_PLACEMENT_EVIDENCE_WORDS = (
    "placement", "job", "jobs", "hire", "hiring", "hired",
    "career support", "career assistance", "employment",
)


def _contains_any_keyword(text, keywords):

    lowered = text.lower()

    return any(keyword in lowered for keyword in keywords)


def _extract_corroborated_sentence(text, evidence_words):
    """
    Like _extract_first_sentence, but only returns a value when TEXT
    contains at least one word that actually corroborates the claimed
    field -- otherwise returns None ("Not available") rather than
    whatever unrelated text happened to sit under the heading.
    """

    if not text or not _contains_any_keyword(text, evidence_words):
        return None

    return _extract_first_sentence(text)


def _extract_certification(text):
    return _extract_corroborated_sentence(text, _CERTIFICATION_EVIDENCE_WORDS)


def _extract_admission_information(text):
    return _extract_corroborated_sentence(text, _ADMISSION_EVIDENCE_WORDS)


def _extract_placement_information(text):
    return _extract_corroborated_sentence(text, _PLACEMENT_EVIDENCE_WORDS)


_FIELD_EXTRACTORS = {
    "duration": _extract_duration,
    "fees": _extract_fees,
    "learning_mode": _extract_learning_mode,
    "location": _extract_location,
    "eligibility": _extract_first_sentence,
    "certification": _extract_certification,
    "admission_information": _extract_admission_information,
    "placement_information": _extract_placement_information,
}


# ============================================================
# EXTRACTION
# ============================================================

def extract_course_record(
    retrieved_chunks,
    source_metadata=None,
    page_context="",
    page_h1=None,
    page_h2=None,
    json_ld_identity=None
):
    """
    Build one structured Course record for a single source page from
    its retrieved chunks (see
    web_retrieval.retriever.retrieve_field_aware_chunks).

    Args:
        retrieved_chunks: list of chunk text strings retrieved for ONE
            source page. Accepts either plain strings or the
            (final_score, ..., chunk) tuples semantic_retrieve_chunks /
            retrieve_field_aware_chunks return -- the chunk text is
            pulled out of either shape automatically.
        source_metadata: dict with "source_url" and optionally
            "source_title" / "source_priority". If source_priority
            isn't given, it's derived from source_url via
            determine_source_priority.
        page_context: a bounded prefix of the page's OWN cleaned text,
            in original reading order -- distinct from
            retrieved_chunks, which field-aware retrieval reorders by
            relevance score and can omit the page's true opening
            heading from entirely. Used for course_name only when
            page_h1/json_ld_identity found nothing (see "COURSE NAME"
            above _guess_course_name_from_page_context).
        page_h1: the page's own <h1> text (see
            web_retrieval.page_reader.fetch_page_structured), if
            available. The highest-confidence course-identity signal
            after an explicit "Course Name" heading -- see
            normalize_h1_to_course_name.
        page_h2: the page's first <h2>, only populated by
            fetch_page_structured when the page has NO <h1> at all
            (some real pages skip straight to h2 as their primary
            heading -- confirmed against rogersoft.com). Checked only
            if page_h1 found nothing.
        json_ld_identity: a breadcrumb/headline identity string from
            the page's JSON-LD structured data, if it has any (also
            from fetch_page_structured). Checked only if page_h1 and
            page_h2 both found nothing.

        All four are optional -- an empty/None value for each just
        means course_name falls through to the next priority, down to
        "Not available" if none of them find anything.

    Never invents a value: any field not explicitly present in the
    retrieved chunks is left as "Not available" (or [] for
    curriculum). Never raises -- malformed input degrades to an
    all-"Not available" record instead of crashing the caller.
    """

    source_metadata = source_metadata or {}

    source_url = source_metadata.get("source_url") or ""
    source_title = source_metadata.get("source_title") or ""
    source_priority = source_metadata.get("source_priority")

    if not source_priority:
        source_priority = determine_source_priority(source_url)

    record = empty_course_record(
        source_url=source_url,
        source_title=source_title,
        source_priority=source_priority
    )

    # Accept either plain chunk strings or (score, ..., chunk) tuples.
    chunk_texts = []

    for item in (retrieved_chunks or []):

        if isinstance(item, str):
            chunk_texts.append(item)

        elif isinstance(item, (list, tuple)) and item:
            chunk_texts.append(item[-1])

    chunk_texts = [c for c in chunk_texts if c and c.strip()]

    if not chunk_texts:
        return validate_course_record(record)

    field_values = {}

    for chunk in chunk_texts:

        category = detect_field_category(chunk)

        if category:
            value = strip_heading_prefix(chunk)

            if value:
                field_values.setdefault(category, []).append(value)

    for category, field_name in _CATEGORY_TO_FIELD.items():

        values = field_values.get(category)

        if not values:
            continue

        if field_name in _LIST_FIELDS:

            items = []

            for value in values:
                for line in value.splitlines():

                    cleaned = line.strip(" -•\t")

                    if cleaned:
                        items.append(cleaned)

            if items:
                record[field_name] = items

            continue

        combined = " ".join(v for v in values if v)

        if field_name in ("course_name", "course_category"):
            # An explicit "Course Name"/"Program Name"/"Course Title"
            # or "Category"/"Course Category" heading is the
            # highest-confidence signal there is for that field --
            # still capped to one sentence so a mis-split page can't
            # dump a whole paragraph in here either.
            extracted = _extract_first_sentence(combined, max_chars=120)
        else:
            extracted = _FIELD_EXTRACTORS[field_name](combined)

        if extracted:
            record[field_name] = extracted

    # course_name priority: explicit heading (handled above, if
    # present) > page's own <h1> > <h2> fallback (only populated when
    # there's no h1 at all) > JSON-LD identity > first line of the
    # page's own content naming a known subject. Never falls back to
    # source_title/page title, and never searches the
    # reordered/retrieved chunks -- see "COURSE NAME" above for why.
    if record["course_name"] == NOT_AVAILABLE and page_h1:
        record["course_name"] = (
            normalize_h1_to_course_name(page_h1) or NOT_AVAILABLE
        )

    if record["course_name"] == NOT_AVAILABLE and page_h2:
        record["course_name"] = (
            normalize_h1_to_course_name(page_h2) or NOT_AVAILABLE
        )

    if record["course_name"] == NOT_AVAILABLE and json_ld_identity:
        record["course_name"] = (
            normalize_h1_to_course_name(json_ld_identity) or NOT_AVAILABLE
        )

    if record["course_name"] == NOT_AVAILABLE:

        guessed_course = _guess_course_name_from_page_context(page_context)

        if guessed_course:
            record["course_name"] = guessed_course

    guessed_institute = _guess_institute_name(
        chunk_texts,
        source_url,
        source_priority,
        source_title
    )

    if guessed_institute:
        record["institute_name"] = guessed_institute

    # Only fall back to guessing from course_name if there wasn't an
    # explicit "Category"/"Course Category" heading (handled above, in
    # the field_values merge loop) -- explicit evidence always wins.
    if record["course_category"] == NOT_AVAILABLE:

        guessed_category = _guess_course_category(record["course_name"])

        if guessed_category:
            record["course_category"] = guessed_category

    guessed_level = _guess_level(chunk_texts)

    if guessed_level:
        record["level"] = guessed_level

    return validate_course_record(record)


# ============================================================
# VALIDATION
# ============================================================

def validate_course_record(record):
    """
    Defensively normalizes any dict-like input into a fully-shaped,
    safe Course record -- every key in COURSE_FIELDS guaranteed
    present, string fields guaranteed to be non-empty strings (else
    "Not available"), curriculum guaranteed to be a list of non-empty
    strings. Never raises, regardless of how malformed the input is
    (wrong types, missing keys, extra keys, not even a dict).
    """

    if not isinstance(record, dict):
        return empty_course_record()

    safe = empty_course_record(
        source_url=(
            record.get("source_url")
            if isinstance(record.get("source_url"), str)
            else ""
        ),
        source_title=(
            record.get("source_title")
            if isinstance(record.get("source_title"), str)
            else ""
        ),
        source_priority=(
            record.get("source_priority")
            if isinstance(record.get("source_priority"), str)
            else "unknown"
        ),
    )

    for field in COURSE_FIELDS:

        if field in ("source_url", "source_title", "source_priority"):
            continue

        value = record.get(field, NOT_AVAILABLE)

        if field in _LIST_FIELDS:

            if isinstance(value, list):
                safe[field] = [
                    str(item).strip()
                    for item in value
                    if str(item).strip()
                ]
            else:
                safe[field] = []

        else:

            if isinstance(value, str) and value.strip():
                safe[field] = value.strip()
            else:
                safe[field] = NOT_AVAILABLE

    return safe


def parse_llm_course_json(raw_text, source_metadata=None):
    """
    Parse and validate a JSON Course record that an LLM-assisted
    extraction fallback *would* return, if one is added later -- not
    currently called anywhere in the pipeline (extraction is fully
    deterministic for now, see this module's docstring).

    Never trusts the input blindly: a parse failure, a non-dict
    payload, unknown keys, or wrong-typed values all degrade to a safe
    "Not available" record via validate_course_record rather than
    raising or letting the model's own claims about source_url /
    source_title / source_priority override the caller-supplied
    metadata, which is always authoritative.
    """

    try:
        data = json.loads(raw_text)

    except (TypeError, json.JSONDecodeError):
        data = None

    if not isinstance(data, dict):
        data = {}

    record = validate_course_record(data)

    meta = source_metadata or {}

    if meta.get("source_url"):
        record["source_url"] = meta["source_url"]

    if meta.get("source_title"):
        record["source_title"] = meta["source_title"]

    if meta.get("source_priority"):
        record["source_priority"] = meta["source_priority"]
    elif record["source_priority"] == "unknown" and record["source_url"]:
        record["source_priority"] = determine_source_priority(record["source_url"])

    return record
