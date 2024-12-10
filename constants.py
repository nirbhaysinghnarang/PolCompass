category_ideology_mapping = {
    # Foreign Policy
    "101 - Foreign Special Relationships: Positive": (0, 2),   # Supportive of international cooperation
    "102 - Foreign Special Relationships: Negative": (0, -2),  # Critical of certain international relationships
    "103 - Anti-Imperialism": (0, -4),  # Strongly progressive stance against imperialism
    "104 - Military: Positive": (1, 3),  # Supportive of military strength and defense
    "105 - Military: Negative": (-1, -3),  # Opposes military expenditure and promotes peace
    "106 - Peace": (-1, -4),  # Strongly progressive emphasis on peace
    "107 - Internationalism: Positive": (0, 4),  # Highly supportive of international cooperation
    "108 - European Community/Union: Positive": (0, 3),  # Supportive of EU integration and cooperation
    "109 - Internationalism: Negative": (0, -4),  # Strongly nationalistic, opposed to internationalism
    "110 - European Community/Union: Negative": (0, -3),  # Critical of EU policies and contributions

    # Democracy and Freedoms
    "201 - Freedom and Human Rights": (-2, 4),  # Progressive support for individual freedoms
    "202 - Democracy": (0, 2),  # General support for democratic principles
    "203 - Constitutionalism: Positive": (1, 2),  # Supportive of constitutional frameworks
    "204 - Constitutionalism: Negative": (-1, -2),  # Opposes existing constitutional structures

    # Structure of Government
    "301 - Federalism": (-2, 2),  # Supports decentralization and regional autonomy
    "302 - Centralisation": (2, -2),  # Advocates for a strong centralized government
    "303 - Governmental and Administrative Efficiency": (1, 1),  # Neutral to slightly right-leaning emphasis on efficiency
    "304 - Political Corruption": (0, 0),  # Neutral stance on corruption (neutral scores as it's an issue rather than an ideology)
    "305 - Political Authority": (2, 3),  # Supports strong and stable government authority

    # Economic
    "401 - Free Market Economy": (5, 0),  # Strongly right-leaning, pro-free market
    "402 - Incentives": (3, 0),  # Pro-business incentives, slightly right-leaning
    "403 - Market Regulation": (-2, 0),  # Slightly left-leaning support for regulation
    "404 - Economic Planning": (-3, 0),  # Left-leaning support for government planning
    "405 - Corporatism/ Mixed Economy": (-1, 0),  # Slightly left-leaning mixed economic approach
    "406 - Protectionism: Positive": (-2, 0),  # Left-leaning support for market protection
    "407 - Protectionism: Negative": (2, 0),  # Right-leaning support for free trade
    "408 - Economic Goals": (0, 0),  # Neutral economic statements
    "409 - Keynesian Demand Management": (-4, 0),  # Strongly left-leaning demand-side policies
    "410 - Economic Growth: Positive": (2, 0),  # Right-leaning support for growth
    "411 - Technology and Infrastructure": (1, 1),  # Slightly left-leaning support for infrastructure
    "412 - Controlled Economy": (-5, 0),  # Highly left-leaning, advocating for complete economic control
    "413 - Nationalisation": (-5, 0),  # Highly left-leaning, supports state ownership
    "414 - Economic Orthodoxy": (3, 0),  # Right-leaning support for traditional economic policies
    "415 - Marxist Analysis: Positive": (-5, 0),  # Highly left-leaning Marxist ideology
    "416 - Anti-Growth Economy: Positive": (-3, -2),  # Left-leaning support for sustainable development

    # Social and Cultural
    "501 - Environmental Protection: Positive": (-3, 2),  # Left-leaning support for environmental policies
    "502 - Culture: Positive": (0, 2),  # Slightly progressive support for cultural initiatives
    "503 - Equality: Positive": (-4, 3),  # Strongly progressive support for social equality
    "504 - Welfare State Expansion": (-4, 2),  # Strongly left-leaning support for welfare
    "505 - Welfare State Limitation": (4, -2),  # Strongly right-leaning opposition to welfare expansion
    "506 - Education Expansion": (-2, 2),  # Slightly left-leaning support for education
    "507 - Education Limitation": (2, -2),  # Right-leaning opposition to education expansion

    "601 - National Way of Life: Positive": (0, 4),  # Highly conservative support for national identity
    "602 - National Way of Life: Negative": (0, -4),  # Highly progressive opposition to nationalistic policies
    "603 - Traditional Morality: Positive": (0, 5),  # Highly conservative support for traditional morals
    "604 - Traditional Morality: Negative": (0, -5),  # Highly progressive opposition to traditional morals
    "605 - Law and Order: Positive": (1, 4),  # Conservative support for law and order
    "606 - Civic Mindedness: Positive": (-1, 3),  # Slightly progressive support for civic unity
    "607 - Multiculturalism: Positive": (-3, 4),  # Strongly progressive support for multiculturalism
    "608 - Multiculturalism: Negative": (3, -4),  # Strongly conservative opposition to multiculturalism

    "701 - Labour Groups: Positive": (-3, 2),  # Left-leaning support for labor
    "702 - Labour Groups: Negative": (3, -2),  # Right-leaning opposition to labor groups
    "703 - Agriculture and Farmers: Positive": (2, 1),  # Slightly right-leaning support for farmers
    "704 - Middle Class and Professional Groups": (0, 1),  # Neutral to slightly conservative support
    "705 - Underprivileged Minority Groups": (-3, 3),  # Left-leaning support for minorities
    "706 - Non-economic Demographic Groups": (-2, 2),  # Slightly left-leaning support for demographic groups

    "000 - No meaningful category applies": (0, 0)  # Neutral
}