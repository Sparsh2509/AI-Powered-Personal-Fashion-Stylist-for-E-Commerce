"""
fashion_rules.py — The fashion knowledge base documents.

WHY this file exists:
  RAG (Retrieval-Augmented Generation) works by storing knowledge 
  as text "documents" in a vector database (ChromaDB).
  
  When a user comes in, we search this database for the most 
  relevant rules for THEIR specific profile, then give those 
  rules to Gemini so it generates grounded recommendations.

  This prevents hallucination — Gemini can't just make things up,
  it must base its answer on these curated rules.

HOW TO ADD MORE:
  Each document is a dict with:
    - "id": unique string ID
    - "text": the fashion rule as natural language text
    - "metadata": tags for filtering (face_shape, skin_tone, etc.)
"""

FASHION_DOCUMENTS = [

    # ===========================================================
    # SECTION 1: FACE SHAPE RULES
    # ===========================================================

    {
        "id": "face_oval_001",
        "text": (
            "Oval face shapes are considered the most balanced and versatile. "
            "Almost all necklines work well — V-necks, round necks, square necks, "
            "and boat necks all complement oval faces. "
            "Oval faces can wear most hat styles and hair accessories. "
            "Structured shoulders and tailored blazers enhance the natural balance. "
            "People with oval faces should avoid styles that add width to the forehead "
            "or chin, such as very wide-brimmed hats or extremely oversized hoods."
        ),
        "metadata": {"category": "face_shape", "face_shape": "oval"}
    },

    {
        "id": "face_round_001",
        "text": (
            "Round faces benefit from styles that create the illusion of length. "
            "V-necklines are especially flattering as they draw the eye downward. "
            "Asymmetrical necklines, deep scoop necks, and open-collar shirts work well. "
            "Avoid round necklines, turtlenecks, and bateau necks which can make "
            "the face look wider. "
            "High ponytails, sleek updos, and straight-line hairstyles add vertical length. "
            "Vertical stripes in clothing help elongate the face and body."
        ),
        "metadata": {"category": "face_shape", "face_shape": "round"}
    },

    {
        "id": "face_square_001",
        "text": (
            "Square faces have strong angular jawlines and should choose styles "
            "that soften and balance this angularity. "
            "Round and oval necklines are most flattering. Cowl necks and "
            "draped fabrics around the neckline add softness. "
            "Avoid square or boat necklines that mirror the jaw's squareness. "
            "Ruffles and soft, flowing fabrics near the face help soften strong features. "
            "Off-shoulder tops draw attention to the shoulders and away from the jaw. "
            "Layered necklaces and drop earrings that hang below the jawline work well."
        ),
        "metadata": {"category": "face_shape", "face_shape": "square"}
    },

    {
        "id": "face_heart_001",
        "text": (
            "Heart-shaped faces have a wider forehead and narrower chin. "
            "The goal is to add width near the chin and minimize width at the top. "
            "V-necks, scoop necks, and sweetheart necklines balance heart faces. "
            "Wide crew necks and boat necks at the collarbone add width to narrow chins. "
            "Avoid high ruffled necklines and strapless tops that emphasize the wide forehead. "
            "A-line and flared skirts help balance proportions by adding width at the hips. "
            "Statement earrings that sit at or below the chin draw the eye downward."
        ),
        "metadata": {"category": "face_shape", "face_shape": "heart"}
    },

    {
        "id": "face_diamond_001",
        "text": (
            "Diamond faces are narrow at the forehead and chin with wide cheekbones. "
            "The goal is to soften the cheekbones and add width at the forehead and chin. "
            "Cowl necks, wide scoop necks, and square necklines add width to the chin area. "
            "Halter necks add width to a narrow forehead. "
            "Side-swept bangs and hairstyles with volume at the crown balance diamond faces. "
            "Avoid styles that add width to cheekbones like wide collars at cheek level."
        ),
        "metadata": {"category": "face_shape", "face_shape": "diamond"}
    },

    {
        "id": "face_rectangle_001",
        "text": (
            "Rectangle faces are longer than they are wide with similar forehead, "
            "cheekbone, and jaw widths. The goal is to add width and minimize length. "
            "Horizontal necklines like boats, squares, and wide scoop necks add width. "
            "Turtlenecks, mock necks, and high-necked styles add warmth to longer face shapes. "
            "Avoid very long pendant necklaces that extend the face's vertical line. "
            "Horizontal stripes and wide belts break the vertical line effectively. "
            "Layered, voluminous hairstyles with width at the sides balance rectangle faces."
        ),
        "metadata": {"category": "face_shape", "face_shape": "rectangle"}
    },

    {
        "id": "face_triangle_001",
        "text": (
            "Triangle or pear-shaped faces have a wider jaw and narrower forehead. "
            "The goal is to add width and visual interest to the upper face. "
            "Boat necks, off-shoulder tops, and wide scoop necks broaden the shoulder line. "
            "Embellished or ruffled tops draw attention upward to balance the wider jaw. "
            "Avoid wide collars and busy patterns at the hip which emphasize the lower face. "
            "Statement earrings and headbands attract attention to the upper face. "
            "Bold necklaces and scarves add width and visual interest at the neckline."
        ),
        "metadata": {"category": "face_shape", "face_shape": "triangle"}
    },


    # ===========================================================
    # SECTION 2: SKIN TONE & COLOR RULES
    # ===========================================================

    {
        "id": "skin_fair_warm_001",
        "text": (
            "Fair skin with warm (yellow/golden) undertones looks best in earthy, "
            "warm colors that enhance the golden quality of the skin. "
            "Best colors: peach, coral, warm beige, camel, golden yellow, olive green, "
            "rust, terracotta, warm brown, and off-white ivory. "
            "Avoid: stark white, cool grays, and bluish-purple tones that wash out "
            "the warmth and make the skin look pallid. "
            "Jewellery: gold tones complement warm undertones beautifully."
        ),
        "metadata": {"category": "skin_color", "skin_tone": "fair", "undertone": "warm"}
    },

    {
        "id": "skin_fair_cool_001",
        "text": (
            "Fair skin with cool (pink/blue) undertones looks best in jewel tones "
            "and cool colors that harmonize with the pink quality of the skin. "
            "Best colors: icy pink, lavender, powder blue, cool gray, navy, "
            "emerald green, bright white, sapphire blue, and burgundy wine. "
            "Avoid: orange, warm browns, and golden yellows which clash with "
            "cool undertones and can make the skin look ruddy or sallow. "
            "Jewellery: silver and white gold flatter cool undertones."
        ),
        "metadata": {"category": "skin_color", "skin_tone": "fair", "undertone": "cool"}
    },

    {
        "id": "skin_medium_warm_001",
        "text": (
            "Medium skin with warm undertones has a beautiful golden or olive base. "
            "Best colors: mustard yellow, warm orange, terracotta, olive, "
            "warm red, chocolate brown, gold, forest green, and warm coral. "
            "Earth tones are especially flattering as they echo the skin's warmth. "
            "Avoid: cool pastels and neons that can make warm medium skin appear dull. "
            "White: opt for cream or warm ivory rather than stark white. "
            "Jewellery: gold, bronze, and copper tones are perfect."
        ),
        "metadata": {"category": "skin_color", "skin_tone": "medium", "undertone": "warm"}
    },

    {
        "id": "skin_medium_cool_001",
        "text": (
            "Medium skin with cool undertones suits jewel tones and vibrant cool colors. "
            "Best colors: royal blue, plum, cool-toned purple, magenta, "
            "emerald, cool pink, and true red (blue-based). "
            "Navy and rich cool colors create a sharp, polished look. "
            "Avoid: orange-red, mustard, and bronze which emphasize sallowness. "
            "Jewellery: silver and cool-toned metals shine against cool medium skin."
        ),
        "metadata": {"category": "skin_color", "skin_tone": "medium", "undertone": "cool"}
    },

    {
        "id": "skin_olive_neutral_001",
        "text": (
            "Olive skin has a natural greenish or grayish undertone and is incredibly versatile. "
            "Best colors: rich jewel tones like emerald, sapphire, ruby, and amethyst. "
            "Earth tones like tan, camel, olive, and rust are naturally harmonious. "
            "Warm oranges, corals, and warm reds are especially striking. "
            "White works beautifully; both crisp white and warm ivory are flattering. "
            "Avoid: neon colors and pastels which can clash with the olive depth. "
            "Jewellery: both gold and silver work, but gold is especially stunning."
        ),
        "metadata": {"category": "skin_color", "skin_tone": "olive", "undertone": "neutral"}
    },

    {
        "id": "skin_tan_warm_001",
        "text": (
            "Tan skin with warm undertones has a rich, sun-kissed quality. "
            "Best colors: bright white (creates beautiful contrast), "
            "coral, orange, warm yellow, turquoise, and fuchsia. "
            "Saturated, vivid colors look stunning against tan warm skin. "
            "Vibrant tropical colors — mango, papaya, ocean blue — are very flattering. "
            "Avoid: brown and dark beige that blend into the skin tone without contrast. "
            "Navy and bright white are always a winning combination for tan skin."
        ),
        "metadata": {"category": "skin_color", "skin_tone": "tan", "undertone": "warm"}
    },

    {
        "id": "skin_deep_cool_001",
        "text": (
            "Deep skin with cool undertones has rich, blue-based undertones and stunning depth. "
            "Best colors: bright white and ivory create dramatic, beautiful contrast. "
            "Rich jewel tones — electric blue, emerald, fuchsia, violet, cobalt — are striking. "
            "True red is especially magnificent against deep cool skin. "
            "Avoid: very dark colors like navy on navy which lose definition and depth. "
            "Light neutral separates create clear contrast that showcases deep skin beautifully. "
            "Jewellery: silver and platinum tones complement cool deep skin."
        ),
        "metadata": {"category": "skin_color", "skin_tone": "deep", "undertone": "cool"}
    },

    {
        "id": "skin_deep_warm_001",
        "text": (
            "Deep skin with warm undertones has a beautiful golden-brown richness. "
            "Best colors: warm oranges, copper, gold, warm red, berry, caramel, "
            "mustard, forest green, and warm white. "
            "Earth tones that echo the skin's warmth are naturally beautiful. "
            "Jewel tones like amber, topaz, and warm emerald look luxurious. "
            "Avoid: muted, dusty tones that dull the richness of warm deep skin. "
            "Jewellery: gold, copper, and bronze tones are magnificent."
        ),
        "metadata": {"category": "skin_color", "skin_tone": "deep", "undertone": "warm"}
    },


    # ===========================================================
    # SECTION 3: BODY TYPE RULES
    # ===========================================================

    {
        "id": "body_hourglass_001",
        "text": (
            "Hourglass body types have balanced shoulders and hips with a defined waist. "
            "The goal is to celebrate and highlight the natural waist definition. "
            "Best styles: fitted dresses and tops that follow the body's curves, "
            "wrap dresses and tops (a classic for hourglass), belted styles "
            "that cinch at the waist, and bodycon silhouettes. "
            "High-waisted pants and skirts emphasize the waist beautifully. "
            "A-line and fit-and-flare dresses follow the natural curve. "
            "Avoid: boxy, shapeless styles that hide the waist — "
            "drop-waist dresses and oversized tops that lose the waist definition. "
            "Best fabrics: soft, drapey fabrics that follow curves without being stiff."
        ),
        "metadata": {"category": "body_type", "body_type": "hourglass"}
    },

    {
        "id": "body_pear_001",
        "text": (
            "Pear body types are narrower at the shoulders and wider at the hips. "
            "The goal is to balance proportions by drawing attention upward. "
            "Best tops: embellished, ruffled, or bold-patterned tops that add "
            "visual interest and width to the upper body. "
            "Off-shoulder and wide-neck tops broaden the shoulder line. "
            "Best bottoms: dark, solid-colored bottoms minimize the hip area. "
            "A-line skirts skim gracefully over wider hips. "
            "Straight-leg and boot-cut pants balance the silhouette. "
            "Avoid: tight, clingy bottoms and bold prints on the hips. "
            "Avoid drop-shoulder tops that make shoulders appear even narrower. "
            "Fabrics: structured fabrics on top, soft fabrics on bottom."
        ),
        "metadata": {"category": "body_type", "body_type": "pear"}
    },

    {
        "id": "body_apple_001",
        "text": (
            "Apple body types carry weight in the middle with a less defined waist, "
            "often with slender legs and arms. "
            "The goal is to draw attention to the legs and arms while "
            "creating the illusion of a waist. "
            "Best styles: empire waist tops and dresses that flow from under the bust. "
            "V-necks and wrap tops that create a vertical line through the center. "
            "Tunics paired with fitted bottoms show off slender legs. "
            "A-line dresses that gently skim the midsection. "
            "Avoid: clingy fabrics around the midsection and cropped tops. "
            "Avoid bold belts that emphasize the widest point. "
            "Structured fabrics hold their shape without clinging to the middle."
        ),
        "metadata": {"category": "body_type", "body_type": "apple"}
    },

    {
        "id": "body_rectangle_001",
        "text": (
            "Rectangle body types have similar measurements at the shoulders, "
            "waist, and hips with minimal waist definition. "
            "The goal is to create curves and the illusion of a waist. "
            "Best styles: peplum tops create waist definition and hip curves. "
            "Wrap dresses and belted tops cinch a waist. "
            "Ruffles, frills, and textured fabrics add volume and dimension. "
            "Layering creates depth and interest. Two-tone outfits (different colors "
            "on top and bottom) break the straight line. "
            "Avoid: straight-cut shift dresses that emphasize the lack of curves. "
            "Wide-leg and straight-cut pants maintain the rectangle appearance. "
            "Skinny jeans with fitted tops elongate well, and cropped tops "
            "can create a waist illusion when paired with high-waisted bottoms."
        ),
        "metadata": {"category": "body_type", "body_type": "rectangle"}
    },

    {
        "id": "body_inverted_triangle_001",
        "text": (
            "Inverted triangle body types have broader shoulders than hips, "
            "creating a V-shaped silhouette. "
            "The goal is to minimize the shoulder area and add volume to the hips. "
            "Best bottoms: full, A-line, or pleated skirts add hip volume. "
            "Wide-leg, flared, and palazzo pants balance broader shoulders. "
            "Patterned, textured, or embellished bottoms draw the eye down. "
            "Best tops: simple, minimalist tops balance strong shoulders. "
            "V-necks and halter tops create a downward visual line. "
            "Avoid: cap sleeves, boat necks, and shoulder pads that emphasize width. "
            "Avoid heavily embellished tops that draw attention to wide shoulders. "
            "Structured blazers can work if balanced with full-volume bottoms."
        ),
        "metadata": {"category": "body_type", "body_type": "inverted_triangle"}
    },


    # ===========================================================
    # SECTION 4: PATTERN & FABRIC RULES
    # ===========================================================

    {
        "id": "patterns_scale_001",
        "text": (
            "Pattern scale should be proportional to body size. "
            "Petite frames (under 5'4\") look best in small to medium patterns — "
            "large patterns can overwhelm a small frame. "
            "Tall or larger frames can carry large, bold patterns beautifully. "
            "Vertical stripes create height and a slimming effect. "
            "Horizontal stripes add width and can be used strategically — "
            "for example, on the upper body of a pear shape to add shoulder width. "
            "Diagonal stripes are slimming and dynamic. "
            "Solid colors are always safe and versatile. "
            "Mixed patterns (pattern mixing) work when scales are different "
            "and one color is shared between patterns."
        ),
        "metadata": {"category": "patterns", "subcategory": "scale"}
    },

    {
        "id": "fabric_drape_001",
        "text": (
            "Fabric choice dramatically affects how clothing falls and flatters. "
            "Drapey fabrics (silk, chiffon, satin, jersey) flow softly over curves "
            "and are ideal for bodies with curves to celebrate. "
            "Structured fabrics (cotton twill, denim, blazer fabric) hold their shape "
            "and are ideal for creating structure where the body lacks it. "
            "Bodycon or stretchy fabrics (jersey, spandex blends) highlight curves "
            "and work best on bodies the wearer is confident showing. "
            "Avoid stiff, thick fabrics on petite frames as they add bulk. "
            "Avoid very clingy, thin fabrics around midsections the wearer wants to minimize. "
            "Lightweight fabrics like linen and cotton voile are flattering in warmer climates."
        ),
        "metadata": {"category": "fabric", "subcategory": "drape"}
    },

    # ===========================================================
    # SECTION 5: OCCASION-BASED RULES
    # ===========================================================

    {
        "id": "occasion_formal_001",
        "text": (
            "For formal occasions like office wear, interviews, and professional settings: "
            "Structured blazers paired with tailored trousers or pencil skirts create authority. "
            "Neutral colors (navy, charcoal, black, white) are universally appropriate. "
            "Minimal patterns or subtle textures (pinstripe, fine check) look professional. "
            "Closed-toe heels or formal flats complete the polished look. "
            "Avoid overly casual fabrics like denim or linen in professional settings. "
            "Accessories should be minimal and elegant — stud earrings, simple watch, "
            "structured handbag. Silk blouses are elevated alternatives to cotton shirts."
        ),
        "metadata": {"category": "occasion", "occasion_type": "formal"}
    },

    {
        "id": "occasion_casual_001",
        "text": (
            "For casual everyday wear: Prioritize comfort without sacrificing style. "
            "Well-fitted jeans are a versatile foundation piece. "
            "Curate a capsule wardrobe of basics in neutral colors that mix and match. "
            "Elevated basics — a quality plain tee, good jeans, clean sneakers — "
            "look effortlessly stylish. "
            "Add personality with one statement piece: bold earrings, a colorful bag, "
            "or an interesting print top. "
            "Layer lightweight pieces (denim jacket, cardigan, blazer) for versatility. "
            "Monochromatic casual outfits (same color family head-to-toe) look "
            "put-together with minimal effort."
        ),
        "metadata": {"category": "occasion", "occasion_type": "casual"}
    },
]


def get_all_texts():
    """Return list of all document texts (for embedding)."""
    return [doc["text"] for doc in FASHION_DOCUMENTS]


def get_all_ids():
    """Return list of all document IDs."""
    return [doc["id"] for doc in FASHION_DOCUMENTS]


def get_all_metadatas():
    """Return list of all document metadata dicts."""
    return [doc["metadata"] for doc in FASHION_DOCUMENTS]
