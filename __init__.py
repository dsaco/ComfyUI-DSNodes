from .nodes.FluxCombinationNode import FluxCombinationNode


NODE_CLASS_MAPPINGS = {"FluxCombination": FluxCombinationNode}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"FluxCombination": "Flux Combination"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
