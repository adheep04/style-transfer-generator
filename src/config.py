def get_config():
    return {
        'steps' : 18000,
        'lr' : 0.2,
        'scheduler' : 0.75,
        'alpha' : 1,
        'beta' : 1.3 * 1e-5,
        'trial' : 'style_trial_2',
        'patience' : 6,
        'content_layer' : 1,
        'style_layer' : 4,
        'content_path' : 'data/content_1.jpg',
        'style_path' : 'data/style_5.jpg',
        'wl' : [0.2, 0.2, 0.2, 0.2, 0.2]
    }
