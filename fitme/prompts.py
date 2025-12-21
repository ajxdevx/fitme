"""Try-on prompts for Gemini."""

PROMPT_CLOTHING = """You are a virtual try-on system.
Image 1: A target person photo.
Image 2: A clothing product image (flat lay or on hanger).

Generate a photo-realistic image of the target person wearing this clothing item.

CRITICAL RULES:
- The person's face MUST remain EXACTLY identical - same facial features, expression, age, skin texture
- Do NOT regenerate or modify the face in any way
- Preserve the exact same body shape, proportions, and pose
- Preserve the same background
- Fit the clothing naturally with realistic draping and shadows
- Keep hair, accessories, and all non-clothing elements unchanged
- Output the FULL BODY image at the same framing as the input
"""

PROMPT_TRANSFER = """You are a virtual try-on system.
Image 1: A target person (the person who should wear the clothes).
Image 2: A source model wearing specific clothing.

Your task: Generate an image of the TARGET PERSON (Image 1) wearing the EXACT SAME CLOTHING from the source model (Image 2).

CRITICAL RULES:
- The target person's face MUST remain EXACTLY identical - same facial features, expression, age, skin texture
- Do NOT regenerate or modify the face in any way
- Use the target person's body shape and pose
- Extract only the clothing design, color, pattern, and style from Image 2
- Fit that clothing naturally onto the target person's body
- Keep the target person's background
- Output the FULL BODY image at the same framing as Image 1
"""
