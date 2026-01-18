# decision_rules.py

REJECT_CLASSES = {
    'rotten_apple',
    'rotten_banana',
    'rotten_orange'
}

WARN_CLASSES = {
    'apple_stem'
}

CONF_THRESH = 0.5


def decide_acceptance(result):
    reject, warn = False, False

    for box in result.boxes:
        cls = int(box.cls.cpu().numpy())
        cls_name = result.names[cls]
        conf = float(box.conf.cpu().numpy())

        if conf >= CONF_THRESH:
            if cls_name in REJECT_CLASSES:
                reject = True
            elif cls_name in WARN_CLASSES:
                warn = True

    if reject:
        return "REJECT"
    elif warn:
        return "ACCEPT_WITH_WARNING"
    return "ACCEPT"
