def parse_objects_from_cfg(cfg: list, objects_category: str = 'Objects'):
    def parse_instance(class_name: str, args: dict):
        try:
            object_type = eval(class_name)
        except NameError:
            raise ValueError(f"Class {class_name} not found.")
        for i, (arg_name, arg_value) in enumerate(args.items()):
            try:
                args[arg_name] = arg_value
            except NameError:
                raise ValueError(f"Class {class_name} has no param {args[arg_name]}.")
        return object_type(**args)

    objects = []
    for instance in cfg:
        if isinstance(instance, (tuple, list)):
            assert len(instance) == 2 and isinstance(instance[1], dict), \
                f"{objects_category} must be specified as follows: [class_name, dict(params)]"
            instance = parse_instance(*instance)
        objects.append(instance)
    return objects
