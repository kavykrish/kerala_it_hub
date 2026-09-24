"""
Shared list of known Kerala place names, used both by
web_retrieval.course_merger (to recognize a trailing location suffix
on an institute name) and web_retrieval.course_extractor (to validate
that an extracted "location" field value is an actual place, not
arbitrary text from a marketing paragraph). Kept as its own tiny
module so both can import the exact same list instead of drifting.
"""

KERALA_LOCATIONS = (
    "calicut", "kozhikode", "kochi", "cochin", "ernakulam",
    "trivandrum", "thiruvananthapuram", "kollam", "kottayam",
    "thrissur", "trichur", "palakkad", "palghat", "kannur",
    "cannanore", "alappuzha", "alleppey", "malappuram", "idukki",
    "wayanad", "kasaragod", "pathanamthitta",
)
