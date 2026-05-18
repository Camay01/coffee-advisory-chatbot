
OLLAMA_MODEL = "qwen2.5:3b"

CROP_VARIETIES = {
    "Arabica": ["Cauvery", "Kent/old Arabica", "Selection 9", "Chandragiri", "S.795"],
    "Robusta": ["CxR", "Peridenia", "Old Robusta", "Clonal Robusta", "S.274"],
}
COFFEE_CROPS = {"arabica", "robusta", "coffee"}

# Soil Parameters 
SOIL_PARAMS = [
    ("pH", "pH"),
    ("OC",  "OC%"),
    ("N",   "N kg/ha"),
    ("P",   "P kg/ha"),
    ("K",   "K kg/ha"),
    ("Zn",  "Zn mg/kg"),
    ("B",   "B mg/kg"),
]

# Classification Thresholds 
SOIL_THRESHOLDS: dict[str, list] = {
    "pH": [
        (5.0,  "significant acidity concern — below 5.0",   True),
        (5.5,  "moderately acidic — below target range",    True),
        (6.5,  "within target range (5.5–6.5)",             False),
        (None, "above target range",                        True),
    ],
    "OC": [
        (0.5,  "very low organic carbon — needs attention", True),
        (0.75, "low organic carbon — below adequate level", True),
        (None, "adequate",                                  False),
    ],
    "N": [
        (200,  "LOW — deficient",   True),
        (400,  "MEDIUM — adequate", False),
        (None, "HIGH",              False),
    ],
    "P": [
        (10,   "LOW — deficient (<10 kg/ha)",     True),
        (25,   "MEDIUM — adequate (10–25 kg/ha)", False),
        (None, "HIGH (>25 kg/ha)",                False),
    ],
    "K": [
        (100,  "LOW — deficient (<100 kg/ha)",       True),
        (200,  "MEDIUM — adequate (100–200 kg/ha)",  False),
        (None, "HIGH (>200 kg/ha)",                  False),
    ],
    "Zn": [
        (0.6,  "LOW — deficient (<0.6 mg/kg)", True),
        (None, "ADEQUATE (≥0.6 mg/kg)",        False),
    ],
    "B": [
        (0.2,  "LOW — deficient (<0.2 mg/kg)", True),
        (None, "ADEQUATE (≥0.2 mg/kg)",        False),
    ],
}

# Unit Conversion 
PPM_TO_KG_HA_FACTOR = 1.68

UNIT_ALIASES: dict[str, str] = {
    # kg/ha
    "kg/ha": "kg/ha", "kg ha": "kg/ha", "kgha": "kg/ha",
    # mg/kg / ppm
    "mg/kg": "mg/kg", "mg kg": "mg/kg", "mgkg": "mg/kg",
    "ppm": "mg/kg",
    "ppm p": "mg/kg", "ppm n": "mg/kg", "ppm k": "mg/kg",
    "ppm zn": "mg/kg", "ppm b": "mg/kg", "ppm s": "mg/kg",
    "ppm fe": "mg/kg", "ppm mn": "mg/kg", "ppm cu": "mg/kg",
    "ppm ca": "mg/kg", "ppm mg": "mg/kg",
    # percent
    "%": "%", "percent": "%", "g/100g": "%",
    # g/kg
    "g/kg": "g/kg",
    # US fertiliser recommendation — must be rejected
    "lb/a": "lb/a", "lbs/a": "lb/a", "lb/acre": "lb/a",
    "lbs/acre": "lb/a", "lb a": "lb/a",
    # dimensionless
    "": "none", "none": "none", "-": "none",
}

# Known Coffee Zones
KNOWN_ZONES = [
    "idukki", "wayanad", "kodagu", "coorg", "hassan",
    "chikmagalur", "chickmagalur", "sakleshpur", "madikeri",
    "virajpet", "somwarpet", "belur", "mudigere",
]
