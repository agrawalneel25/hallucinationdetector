# Reference documents with labelled claims.
# True  = claim is supported by the reference (grounded)
# False = claim is not supported (hallucinated)
#
# Hallucinated claims use vocabulary largely absent from the reference
# so the TF-IDF approach has a realistic chance of catching them.

EXAMPLES = [
    {
        "reference": """
            Marie Curie was a Polish-French physicist and chemist who conducted pioneering
            research on radioactivity. She was the first woman to win a Nobel Prize, and the
            only person to win Nobel Prizes in two different sciences. She received the Nobel
            Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911. Curie was born
            in Warsaw in 1867 and died in 1934 from aplastic anemia caused by prolonged
            exposure to radiation. She discovered the elements polonium and radium.
        """,
        "claims": [
            ("Marie Curie researched radioactivity and was a physicist.",          True),
            ("She received Nobel Prizes in Physics and Chemistry.",                True),
            ("Curie discovered the elements polonium and radium.",                 True),
            ("She was an accomplished pianist who performed across Europe.",       False),
            ("Curie founded a school of philosophy in Berlin.",                    False),
            ("She developed a new theory of gravity and stellar motion.",          False),
        ],
    },
    {
        "reference": """
            The Python programming language was created by Guido van Rossum and first released
            in 1991. Python emphasises code readability and uses significant indentation.
            It supports multiple programming paradigms including procedural, object-oriented,
            and functional programming. Python is dynamically typed and garbage-collected.
            The language is widely used in data science, web development, and automation.
        """,
        "claims": [
            ("Python was created by Guido van Rossum.",                           True),
            ("Python supports object-oriented and functional programming.",        True),
            ("The language is used in data science and web development.",          True),
            ("Python compiles to native machine code for maximum performance.",   False),
            ("It was originally designed for embedded microcontroller firmware.",  False),
            ("Python requires explicit memory allocation and pointer management.", False),
        ],
    },
]
