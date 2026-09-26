"""Known Whisper mistranscriptions of BJJ terminology.

Keys are the incorrect string Whisper produces; values are the correct term.
Matching is case-insensitive. Add new entries as you discover them in real
transcripts — the corrector.py pass runs on every transcript before chunking.
"""

CORRECTIONS: dict[str, str] = {
    # ashi garami variants
    "ashy geronimo": "ashi garami",
    "achey garami": "ashi garami",
    "ashy garami": "ashi garami",
    "ashi garone": "ashi garami",
    "ashy garramy": "ashi garami",

    # de la riva variants
    "de la river": "de la riva",
    "de la reeva": "de la riva",
    "de la riber": "de la riva",
    "de la reeva guard": "de la riva guard",

    # guillotine variants
    "giloteen": "guillotine",
    "gilloteen": "guillotine",
    "gillateen": "guillotine",
    "gila teen": "guillotine",

    # berimbolo variants
    "barimbolo": "berimbolo",
    "berry and bolo": "berimbolo",
    "bear and bolo": "berimbolo",

    # kimura variants
    "kimora": "kimura",
    "kimorah": "kimura",

    # americana variants
    "americana lock": "americana",

    # darce variants
    "dark choke": "darce choke",
    "darcy": "darce",

    # toreando variants
    "toreanda": "toreando",
    "toureando": "toreando",

    # misc
    "ezequiel": "ezekiel",
    "mata leo": "mata leao",
    "sankaku jime": "triangle choke",
}
