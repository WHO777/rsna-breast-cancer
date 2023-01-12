from contextlib import suppress


def get_layer_from_model(model, layer_name, recursive=True):
    def get_layer(model, layer_name):
        with suppress(AttributeError, ValueError):
            return model.get_layer(layer_name)
        with suppress(AttributeError):
            for layer in model.layers:
                if recursive:
                    found_layer = get_layer(layer, layer_name)
                else:
                    found_layer = layer if layer.name == layer_name else None
                if found_layer is not None:
                    return found_layer
        return None
    return get_layer(model, layer_name)