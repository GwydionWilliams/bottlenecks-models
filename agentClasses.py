params = {
    "abstract-hierarchical":
        {
            "class": "abstract-hierarchical",
            "representsHistory": False,
            "statesForAbstraction": ["B0L", "B0R"],
            "selectionStrategy": "sequential",
            "policy": "structured-softmax"
        },

    "structured-hierarchical":
        {
            "class": "structured-hierarchical",
            "representsHistory": False,
            "statesForAbstraction": [],
            "selectionStrategy": "sequential",
            "policy": "structured-softmax"
        },

    "hierarchical":
        {
            "class": "hierarchical",
            "representsHistory": False,
            "statesForAbstraction": [],
            "selectionStrategy": "free",
            "policy": "softmax"
        },

    "flat-history":
        {
            "class": "flat-history",
            "representsHistory": True,
            "statesForAbstraction": [],
            "selectionStrategy": "free",
            "policy": "softmax"
        },

    "flat":
        {
            "class": "flat",
            "representsHistory": False,
            "statesForAbstraction": [],
            "selectionStrategy": "free",
            "policy": "softmax"
        }

}
