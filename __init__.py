from .nodes.FluxT2I import FluxT2INode


NODE_CLASS_MAPPINGS = {"FluxT2I": FluxT2INode}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"FluxT2I": "flux t2i"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
