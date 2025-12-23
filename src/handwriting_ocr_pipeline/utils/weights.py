def remove_module_prefix(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_dict[new_k] = v
    return new_dict
