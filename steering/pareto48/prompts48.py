"""Pareto pilot prompts: 12 development and 12 confirmation prompt-seed pairs (fixed before any generation).

Each has one clear subject (the noun is what the subject masker looks for) and a textured background. Confirmation
prompts are different prompts, not just different seeds, and are not generated until the selection is frozen.
"""
SUFFIX = ", highly detailed background"

DEV = [
    ("a golden retriever sitting on a wooden porch, a garden with dense flowers behind it", "dog"),
    ("a red vintage bicycle leaning against a brick wall covered in ivy, a city street behind it", "bicycle"),
    ("a portrait photo of an elderly man wearing a wool hat, a busy outdoor market behind him", "man"),
    ("a ceramic coffee mug on a cafe table, bookshelves full of books behind it", "mug"),
    ("a tabby cat sitting on a stone wall, a forest of tall trees behind it", "cat"),
    ("a young woman holding a bouquet of sunflowers, a field of tall grass behind her", "woman"),
    ("a red fire hydrant on a sidewalk, parked cars and shop windows behind it", "fire hydrant"),
    ("a white horse standing in a meadow, a wooden fence and pine trees behind it", "horse"),
    ("a potted cactus on a windowsill, a city skyline with many buildings behind it", "cactus"),
    ("a vintage motorcycle parked on a cobblestone street, old houses behind it", "motorcycle"),
    ("a little girl in a yellow raincoat, a playground with colourful equipment behind her", "girl"),
    ("a pineapple on a picnic table, a lake shore with boats and trees behind it", "pineapple"),
]
CONFIRM = [
    ("a black labrador lying on a doormat, a front yard with hedges and flowers behind it", "dog"),
    ("a man in a denim jacket standing on a bridge, a river and buildings behind him", "man"),
    ("a red toy fire truck on a wooden floor, a living room with shelves behind it", "toy truck"),
    ("a brown owl perched on a branch, a dense forest behind it", "owl"),
    ("a stack of pancakes on a plate, a busy diner counter behind it", "pancakes"),
    ("a woman with curly hair sitting on a park bench, trees and a path behind her", "woman"),
    ("a yellow taxi on a street corner, crowded storefronts and signs behind it", "taxi"),
    ("a sheep standing on a hillside, stone walls and farmland behind it", "sheep"),
    ("a white teapot on a kitchen counter, a tiled wall and hanging utensils behind it", "teapot"),
    ("a boy holding a football, a stadium with crowded seats behind him", "boy"),
    ("a bronze lion statue in a plaza, fountains and buildings behind it", "statue"),
    ("a sunflower in close view, a farmhouse and fields behind it", "sunflower"),
]


def cases(split):
    """[(case_id, prompt, noun, seed)]; dev seeds 101-112, confirm seeds 201-212."""
    src, base, tag = (DEV, 101, "d") if split == "dev" else (CONFIRM, 201, "c")
    return [(f"{tag}{i + 1:02d}", p + SUFFIX, n, base + i) for i, (p, n) in enumerate(src)]
