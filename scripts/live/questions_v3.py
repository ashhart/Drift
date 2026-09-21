"""Question templates for the v3 live QA (see configs/preregistration.live-qa-v3.json). Only 'count' differs from v2."""
PRIORITY = ["code", "person", "time", "ingredient", "animal", "colour", "count", "weight", "room", "day"]
NUMERIC = {"code", "time", "count", "weight", "room"}
QUESTION = {"code": "What is the code mentioned in the {genre} about the {thing}?", "person": "Who is the person responsible in the {genre} about the {thing}?",
            "time": "What time is given in the {genre} about the {thing}?", "ingredient": "What is the special ingredient mentioned in the {genre} about the {thing}?",
            "animal": "Which animal is involved in the {genre} about the {thing}?", "colour": "Which colour is mentioned in the {genre} about the {thing}?",
            "count": "What number of items is stated in the {genre} about the {thing}?", "weight": "What weight in kilograms is given in the {genre} about the {thing}?",
            "room": "What is the room or plot number in the {genre} about the {thing}?", "day": "Which day of the week is mentioned in the {genre} about the {thing}?"}
