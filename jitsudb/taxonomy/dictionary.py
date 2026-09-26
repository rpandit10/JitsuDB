"""Hand-curated BJJ taxonomy dictionaries.

Each key is the canonical label; the value is the list of surface forms
that Whisper might produce. Extend these lists as you discover new
transcription variants — bump TAXONOMY_VERSION in constants.py when you do
so that reclassify.py knows to re-run the classifier on all chunks.
"""

POSITIONS: dict[str, list[str]] = {
    "mount": [
        "mount", "mounted", "full mount", "s-mount", "s mount", "technical mount",
    ],
    "side_control": [
        "side control", "side mount", "cross side", "crossbody",
        "100 kilos", "kesa gatame",
    ],
    "half_guard": [
        "half guard", "half-guard", "z guard", "z-guard", "knee shield",
        "lockdown", "deep half", "shallow half",
    ],
    "closed_guard": [
        "closed guard", "full guard",
    ],
    "open_guard": [
        "open guard", "butterfly", "de la riva", "dlr", "spider guard",
        "lasso", "collar sleeve", "x guard", "single leg x", "slx",
        "k guard", "reverse de la riva", "rdlr", "matrix",
    ],
    "back_control": [
        "back control", "back mount", "back take", "rear mount",
        "hooks in", "body triangle", "seatbelt",
    ],
    "turtle": [
        "turtle", "quad pod", "all fours",
    ],
    "north_south": [
        "north south", "north-south", "69 position",
    ],
    "knee_on_belly": [
        "knee on belly", "knee ride", "kob",
    ],
    "front_headlock": [
        "front headlock", "front head", "snap down",
    ],
    "standing": [
        "standing", "takedown", "wrestling", "clinch", "underhook battle",
    ],
    "crucifix": [
        "crucifix",
    ],
    "50_50": [
        "50-50", "fifty fifty", "50 50", "5050",
    ],
    "ashi_garami": [
        "ashi garami", "single leg x", "slx", "saddle", "inside sankaku",
        "outside ashi", "inside ashi",
    ],
}

INTENTS: dict[str, list[str]] = {
    "attack": [
        "attack", "submit", "finish", "offensive", "hunting",
        "setup", "entry", "catch",
    ],
    "defense": [
        "defend", "defending", "defense", "escape", "escaping",
        "survive", "surviving", "counter", "countering",
        "get out", "recover guard", "reguard",
    ],
    "sweep": [
        "sweep", "sweeping", "reversal", "off balance", "come on top",
    ],
    "pass": [
        "pass", "passing", "smash", "smashing", "pressure", "toreando",
        "knee cut", "leg drag", "body lock pass", "over under",
    ],
    "transition": [
        "transition", "flow", "chain", "scramble", "react",
    ],
    "control": [
        "control", "maintain", "pin", "hold", "pressure", "flatten",
    ],
    "takedown": [
        "takedown", "take down", "shot", "single leg", "double leg",
        "trip", "throw", "pull guard",
    ],
}

TECHNIQUES: dict[str, list[str]] = {
    "kimura": ["kimura", "double wristlock"],
    "americana": ["americana", "keylock", "key lock", "paintbrush"],
    "armbar": ["armbar", "arm bar", "juji gatame", "straight armlock"],
    "triangle": ["triangle", "sankaku"],
    "guillotine": ["guillotine", "gilloteen", "gillateen"],
    "rear_naked_choke": ["rear naked", "rnc", "mata leao", "lion killer"],
    "darce": ["darce", "d'arce", "brabo"],
    "anaconda": ["anaconda"],
    "ezekiel": ["ezekiel", "ezequiel"],
    "crossface": ["crossface", "cross face", "cross-face"],
    "underhook": ["underhook", "under hook", "under-hook"],
    "overhook": ["overhook", "over hook", "over-hook", "whizzer"],
    "frame": ["frame", "framing", "stiff arm"],
    "heel_hook": ["heel hook", "inside heel hook", "outside heel hook"],
    "knee_bar": ["knee bar", "kneebar"],
    "ankle_lock": ["ankle lock", "straight ankle", "straight foot lock"],
    "toe_hold": ["toe hold", "toehold"],
    "leg_drag": ["leg drag"],
    "berimbolo": ["berimbolo", "baby bolo"],
    "arm_drag": ["arm drag"],
    "collar_tie": ["collar tie"],
    "body_lock": ["body lock", "bodylock"],
    "hip_escape": ["hip escape", "shrimp", "shrimping"],
    "bridge": ["bridge", "bridging", "upa", "bump"],
    "sit_up_guard": ["sit up guard", "sit-up guard"],
    "lasso": ["lasso"],
    "spider_guard": ["spider guard"],
    "wrist_lock": ["wrist lock", "wristlock"],
    "bow_and_arrow": ["bow and arrow"],
    "clock_choke": ["clock choke"],
    "loop_choke": ["loop choke"],
    "baseball_bat_choke": ["baseball bat choke", "baseball choke"],
    "omoplata": ["omoplata"],
    "gogoplata": ["gogoplata"],
    "peruvian_necktie": ["peruvian necktie"],
    "north_south_choke": ["north south choke"],
    "arm_triangle": ["arm triangle", "head and arm", "kata gatame"],
    "crucifix_choke": ["crucifix choke"],
    "Von_flue_choke": ["von flue", "von flue choke"],
}

CONTEXTS: dict[str, list[str]] = {
    "top": [
        "top", "on top", "passing", "smash", "pressure", "pin", "flatten",
    ],
    "bottom": [
        "bottom", "on bottom", "guard", "sweep", "recover",
        "escape", "reguard", "retain",
    ],
    "standing": [
        "standing", "on the feet", "clinch", "takedown", "throw",
    ],
}

# Folder name fragments that hint at bottom/defensive context
DEFENSE_CATEGORY_HINTS: list[str] = [
    "escape", "defense", "defence", "survival", "recover", "bottom",
]

# Folder name fragments that hint at top/offensive context
ATTACK_CATEGORY_HINTS: list[str] = [
    "attack", "offense", "offence", "submission", "finish", "top",
    "passing", "mount", "side control",
]
