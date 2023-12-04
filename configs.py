
def get_configs(dataset, baseline):

    if dataset.startswith('wikipedia'):
        a, b = 0.5, 0.4
    elif dataset.startswith('amazon'):
        a, b = 0.6, 2.6
    else:
        a, b = 0.55, 1.5
    
#--------------------------eurlex---------------------------
    
    if dataset == 'eurlex':
        data = {
            'feature_size': 5000,
            'label_size'  : 3993,
        }
        model = {
            'num_experts' : 3,
            'hidden_size' : (512,),
            'input_mode'  : 'feature',
            'output_mode' : 'residual',
            'use_norm'    : True,
            'drop_prob'   : 0.7,
        }
        train = {
            'batch_size': 40,
            'lr'        : 0.01,
            'num_epochs': 50,
            'warm_up'   : 5,
            'clf_flood' : 0,
            'div_flood' : -1e-4,
            'div_factor': 1,
        }
        test = {
            'batch_size': 100,
        }
    
    elif dataset == 'eurlex-xtransformer':
        data = {
            'feature_size': 186104,
            'label_size'  : 3956,
        }
        model = {
            'num_experts' : 3,
            'hidden_size' : (512,),
            'input_mode'  : 'feature',
            'output_mode' : 'residual',
            'use_norm'    : True,
            'drop_prob'   : 0.7,
        }
        train = {
            'batch_size': 40,
            'lr'        : 0.01,
            'num_epochs': 50,
            'warm_up'   : 5,
            'clf_flood' : 0,
            'div_flood' : -1e-4,
            'div_factor': 1,
        }
        test = {
            'batch_size': 100,
        }

#--------------------------wiki10---------------------------

    elif dataset in ['wiki10', 'wiki10-xtransformer']:
        data = {
            'feature_size': 101938,
            'label_size'  : 30938,
        }
        model = {
            'num_experts' : 3,
            'hidden_size' : (512,),
            'input_mode'  : 'feature',
            'output_mode' : 'residual',
            'use_norm'    : True,
            'drop_prob'   : 0.7,
        }
        train = {
            'batch_size': 40,
            'lr'        : 0.01,
            'num_epochs': 50,
            'warm_up'   : 5,
            'clf_flood' : 0,
            'div_flood' : -1e-4,
            'div_factor': 1,
        }
        test = {
            'batch_size': 100,
        }
        
#------------------------amazon670k-------------------------
        
    elif dataset == 'amazon670k':
        data = {
            'feature_size': 135909,
            'label_size'  : 670091,
        }
        model = {
            'num_experts' : 3,
            'hidden_size' : (512,),
            'input_mode'  : 'feature',
            'output_mode' : 'residual',
            'use_norm'    : True,
            'drop_prob'   : 0.7,
        }
        train = {
            'batch_size': 100,
            'lr'        : 0.01,
            'num_epochs': 30,
            'warm_up'   : 5,
            'clf_flood' : 0,
            'div_flood' : -1e-4,
            'div_factor': 1,
        }
        test = {
            'batch_size': 100,
        }
        
#-----------------------------------------------------------
    
    cfg = {
        'a': a,
        'b': b,

        'data' : data,
        'model': model,
        'train': train,
        'test' : test,
    }
    
    return cfg
