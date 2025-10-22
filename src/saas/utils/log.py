import logging, json, sys
def get_logger(name):
    l = logging.getLogger(name); l.setLevel(logging.INFO)
    if not l.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('%(message)s'))
        l.addHandler(h)
    old_info = l.info
    def info(obj): old_info(json.dumps(obj) if not isinstance(obj, str) else obj)
    l.info = info
    return l
