PASCAL_CLASSES = {
    'background': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}

Irregulars = {
    'person': 'people',
    'sheep': 'sheep',
    'background': 'background'
}

def get_classname_plural(word):
    if word in Irregulars:
        return Irregulars[word]
    
    if word.endswith('s'):
        return word + 'es'
    
    return word + 's'

def is_plural(word):
    if word in Irregulars.values():
        return True
    
    if word.endswith('es'):
        return True
    
    if word.endswith('s') and word != 'bus':
        return True

    return False