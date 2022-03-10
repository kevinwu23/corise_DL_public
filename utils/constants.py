DOODLE_TO_EMOJI_MAP =  {'smiley face': 'smile',
                         'stop sign': 'stop_sign',
                         'baseball': 'baseball',
                         'basketball': 'basketball',
                         'bed': 'bed',
                         'bicycle': 'bicycle',
                         'eye': 'eyes',
                         'car': 'car',
                         'pizza': 'pizza',
                         'star': 'star'}

PLOT_MAPPINGS = {
        'train_loss': { 'line': 'train', 'facet': 'loss' },
        'val_loss': { 'line': 'validation', 'facet': 'loss' },
        'train_acc': { 'line': 'train', 'facet': 'acc' },
        'val_acc': { 'line': 'validation', 'facet': 'acc' }
    }

PLOT_FACET_CONFIG = {
        'loss': { 'name': 'Cross-Entropy', 'limit': [0, None], 'scale': 'linear' },
        'acc': { 'name': 'Accuracy', 'limit': [0, 1], 'scale': 'linear' }
    }