dataset_defaults = {
    'SCM_1': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 4,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD", 
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': 1.,
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': 1.,
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': 0.1,
        },
        'DIP_Pool_MMD': {
            'lamDIP': 1.0
        },
        'CIP_mean': {
            'lamCIP': 1.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        },
        'IW-CIP_mean':{
            'lamCIP_A': .01,
            'lamCIP': 10.,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 1.,#100.,
            'lamCIP': 1.,
        },
        'IW-DIP_mean':{
            'lamCIP_A': 1.,
            'lamDIP': .01,#.1,
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': 100.,
            'lamDIP': .1,
            'srcId': [0]
        },
        'IW-DIP_Pool_mean':{
            'lamCIP_A': 1.,
            'lamDIP': .01,#.1,
            #'srcId': [0]
        },
        'IW-DIP_Pool_MMD':{
            'lamCIP_A': 100.,
            'lamDIP': .1,
            #'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': 10.,#100.,
            'lamDIP': 1.,
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': 100.,
            'lamDIP': 1.,#.1,
            'srcId': [0]
        },
        'IW-JointDIP_mean':{
            'lamCIP_A': 100.,
            'lamDIP': 1.,#.1,
            #'srcId': [0]
        },
        'IRM': {
            'lamIRM': 10.,#100.,
            'anneal_step': 0
        },
        'VREx': {
            'lamVREx': 100.,#10000.,
            'anneal_step': 100
        },
        'groupDRO': {
            'group_weights_lr': 0.01
        }
    },
    'SCM_2': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 12,
        'n': 1000,
        'd': 9,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD",
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': 1.,#.1,
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': 1.,
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': 10.,#0.01,
        },
        'DIP_Pool_MMD': {
            'lamDIP': 0.1,
        },
        'CIP_mean': {
            'lamCIP': 100.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        },
        'IW-CIP_mean':{
            'lamCIP_A': 10.,
            'lamCIP': 10.,#1.,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 1.,#100.,
            'lamCIP': 10.,#.01,
        },
        'IW-DIP_mean':{
            'lamCIP_A': 10.,
            'lamDIP': .1,
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': 1.,#10.,
            'lamDIP': 10.,
            'srcId': [0]
        },
        'IW-DIP_Pool_mean':{
            'lamCIP_A': 10.,
            'lamDIP': .1,
            #'srcId': [0]
        },
        'IW-DIP_Pool_MMD':{
            'lamCIP_A': 1.,#10.,
            'lamDIP': 10.,
            #'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 1.,#.1,
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 100.,
            'srcId': [0]
        },
        'IW-JointDIP_Pool':{
            'lamCIP_A': 1.,
            'lamDIP': 100.,
            #'srcId': [0]
        },
        'IRM': {
            'lamIRM': 100.,
            'anneal_step': 0
        },
        'VREx': {
            'lamVREx': 100.,#1000.,
            'anneal_step': 10
        },
        'groupDRO': {
            'group_weights_lr': 1.#0.01
        },
    },
    'SCM_3': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 12,
        'n': 1000,
        'd': 18,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD", 
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': 100.,
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': 1.,
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': 0.1,
        },
        'DIP_Pool_MMD': {
            'lamDIP': 1.,
        },
        'CIP_mean': {
            'lamCIP': 1., 
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        },
        'IW-CIP_mean':{
            'lamCIP_A': 10.,
            'lamCIP': 1.,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 1.,
            'lamCIP': 1.,
        },
        'IW-DIP_mean':{
            'lamCIP_A': .01,
            'lamDIP': 100.,
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': .1,
            'lamDIP': 1.,
            'srcId': [0]
        },
        'IW-DIP_Pool_mean':{
            'lamCIP_A': .01,
            'lamDIP': 100.,
            #'srcId': [0]
        },
        'IW-DIP_Pool_MMD':{
            'lamCIP_A': .1,
            'lamDIP': 1.,
            #'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 10.,
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 10.,
            'srcId': [0]
        },
        'IW-JointDIP_Pool':{
            'lamCIP_A': 1.,
            'lamDIP': 10.,
            #'srcId': [0]
        },
        'IRM': {
            'lamIRM': 100.,
            'anneal_step': 0
        },
        'VREx': {
            'lamVREx': 100.,
            'anneal_step': 10,
        },
        'groupDRO': {
            'group_weights_lr': 0.1
        },
    },
    'SCM_4': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 12,
        'n': 1000,
        'd': 18,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD", 
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': 100.,
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': 1.,
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': 0.01,
        },
        'DIP_Pool_MMD': {
            'lamDIP': .1,
        },
        'CIP_mean': {
            'lamCIP': 10.,#.1,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        },
        'IW-CIP_mean':{
            'lamCIP_A': 10.,#100.,
            'lamCIP': 10.,#.1,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 1.,
            'lamCIP': 1.,
        },
        'IW-DIP_mean':{
            'lamCIP_A': 1.,
            'lamDIP': 100.,
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': 1.,
            'lamDIP': 1.,
            'srcId': [0]
        },
        'IW-DIP_Pool_mean':{
            'lamCIP_A': 1.,
            'lamDIP': 100.,
            #'srcId': [0]
        },
        'IW-DIP_Pool_MMD':{
            'lamCIP_A': 1.,
            'lamDIP': 1.,
            #'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 10.,
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 10.,
            'srcId': [0]
        },
        'IW-JointDIP_Pool':{
            'lamCIP_A': 1.,
            'lamDIP': 10.,
            #'srcId': [0]
        },
        'IRM': {
            'lamIRM': 10.,
            'anneal_step': 0
        },
        'VREx': {
            'lamVREx': 10.,
            'anneal_step': 0,#10
        },
        'groupDRO': {
            'group_weights_lr': 0.1
        },
    },
    'SCM_1_1': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 3,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'algs': ["CIP_mean", "CIP_MMD", "IW-CIP_mean", "IW-CIP_MMD"],
        'CIP_mean': {
            'lamCIP': .01,
        },
        'CIP_MMD': {
            'lamCIP': .01,
        },
        'IW-CIP_mean':{
            'lamCIP_A': .01,
            'lamCIP': .1,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': .01,#100.,
            'lamCIP': .01,
        },
    },
    'SCM_2_1': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 3,
        'n': 1000,
        'd': 9,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'algs': ["CIP_mean", "CIP_MMD", "IW-CIP_mean", "IW-CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 100.,
        },
        'CIP_MMD': {
            'lamCIP': 10.,
        },
        'IW-CIP_mean':{
            'lamCIP_A': 100.,
            'lamCIP': 100.,#1.,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 100.,#100.,
            'lamCIP': 100.,#.01,
        },
    },
    'SCM_3_1': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 3,
        'n': 1000,
        'd': 18,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'algs': ["CIP_mean", "CIP_MMD", "IW-CIP_mean", "IW-CIP_MMD"],
        'CIP_mean': {
            'lamCIP': .01, 
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        },
        'IW-CIP_mean':{
            'lamCIP_A': .01,
            'lamCIP': .01,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 1.,
            'lamCIP': .1,
        },
    },
    'SCM_4_1': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 3,
        'n': 1000,
        'd': 18,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'algs': ["CIP_mean", "CIP_MMD", "IW-CIP_mean", "IW-CIP_MMD"],
        'CIP_mean': {
            'lamCIP': .01,#.1,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        },
        'IW-CIP_mean':{
            'lamCIP_A': 10.,#100., #.01
            'lamCIP': .1,#1.,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 1.,
            'lamCIP': 1.,
        },
    },
    'SCM_binary_1': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 3,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,#.1,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        }
    },
    'SCM_binary_2': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 5,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,#100.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,#10.,
        }
    },
    'SCM_binary_3': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 9,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,#100.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        }
    },
    'SCM_binary_4': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 17,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        }
    },
    'SCM_binary_5': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 33,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        }
    },
    'SCM_binary_1_linear': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 3,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,#.1,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        }
    },
    'SCM_binary_2_linear': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 4,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,#100.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,#10.,
        }
    },
    'SCM_binary_3_linear': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 5,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,#100.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        }
    },
    'SCM_binary_4_linear': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 6,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        }
    },
    'SCM_binary_5_linear': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 7,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        }
    },
    'SCM_binary_6_linear': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 8,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 10.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        }
    },
    'MNIST_rotation_1': {
        'n_epochs': 20,
        'optimizer': 'Adam',
        'M': 6,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 21,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD", 
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': .01,#.1,
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': .1,#1.,
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': 0.01,
        },
        'DIP_Pool_MMD': {
            'lamDIP': .1,
        },
        'CIP_mean': {
            'lamCIP': .01,
        },
        'CIP_MMD': {
            'lamCIP': .01,#.1,
        },
        'IW-CIP_mean':{
            'lamCIP_A': 10.,#.1,
            'lamCIP': .01,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': .1,
            'lamCIP': .01,
        },
        'IW-DIP_mean':{
            'lamCIP_A': 10.,#1.,
            'lamDIP': .01,
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': .01,#1.,
            'lamDIP': .1,#1.,
            'srcId': [0]
        },
        'IW-DIP_Pool_mean':{
            'lamCIP_A': 10.,#1.,
            'lamDIP': .01,
            #'srcId': [0]
        },
        'IW-DIP_Pool_MMD':{
            'lamCIP_A': .01,#1.,
            'lamDIP': .1,#1.,
            #'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': .01,#10.,
            'lamDIP': .1,
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': .01,
            'lamDIP': 1.,
            'srcId': [0]
        },
        'IW-JointDIP_Pool':{
            'lamCIP_A': .01,
            'lamDIP': 1.,
            #'srcId': [0]
        },
        'IRM': {
            'lamIRM': .1,
            'anneal_step': 10#0
        },
        'VREx': {
            'lamVREx': 100,#1.,
            'anneal_step': 10#0
        },
        'groupDRO': {
            'group_weights_lr': 1.#0.01
        }
    },
    'MNIST_rotation_2': {
        'n_epochs': 20,
        'optimizer': 'Adam',
        'M': 6,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 21,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD",
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': .01,#1.,
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': .01,
            'srcId': [0],
        },
        'DIP_Pool_mean': {
            'lamDIP': .01,#0.1,
        },
        'DIP_Pool_MMD': {
            'lamDIP': .01,
        },
        'CIP_mean': {
            'lamCIP': .01,
        },
        'CIP_MMD': {
            'lamCIP': .1,
        },
        'IW-CIP_mean':{
            'lamCIP_A':1.,# 100.,
            'lamCIP': .1,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 1.,#.01,
            'lamCIP': .01,#.1,
        },
        'IW-DIP_mean':{
            'lamCIP_A': .01,#100.,
            'lamDIP': .1,#.01,
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': .1,
            'lamDIP': .1,#1.,
            'srcId': [0]
        },
        'IW-DIP_Pool_mean':{
            'lamCIP_A': .01,#100.,
            'lamDIP': .1,#.01,
            #'srcId': [0]
        },
        'IW-DIP_Pool_MMD':{
            'lamCIP_A': .1,
            'lamDIP': .1,#1.,
            #'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': 10.,#.01,
            'lamDIP': .01,
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': .01,
            'lamDIP': 10.,
            'srcId': [0]
        },
        'IW-JointDIP_Pool':{
            'lamCIP_A': .01,
            'lamDIP': 10.,
            #'srcId': [0]
        },
        'IRM': {
            'lamIRM': 1.,#.1,
            'anneal_step': 10
        },
        'VREx': {
            'lamVREx': 10.,
            'anneal_step': 0
        },
        'groupDRO': {
            'group_weights_lr': 1.#0.1
        },
    },
    'MNIST_rotation_3': {
        'n_epochs': 20,
        'optimizer': 'Adam',
        'M': 6,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD",
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': .01,
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': 1.,
            'srcId': [0],
        },
        'DIP_Pool_mean': {
            'lamDIP': .1,#.01,
        },
        'DIP_Pool_MMD': {
            'lamDIP': .1,
        },
        'CIP_mean': {
            'lamCIP': .1,
        },
        'CIP_MMD': {
            'lamCIP': .1,
        },
        'IW-CIP_mean':{
            'lamCIP_A': 100.,
            'lamCIP': .1,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 100.,
            'lamCIP': .1,
        },
        'IW-DIP_mean':{
            'lamCIP_A': 1.,
            'lamDIP': .01,
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': 1.,
            'lamDIP': .1,
            'srcId': [0],
        },
        'IW-DIP_Pool_mean':{
            'lamCIP_A': 1.,
            'lamDIP': .01,
            #'srcId': [0]
        },
        'IW-DIP_Pool_MMD':{
            'lamCIP_A': 1.,
            'lamDIP': .1,
            #'srcId': [0],
        },
        'JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 1.,
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 1.,
            'srcId': [0]
        },
        'IW-JointDIP_Pool':{
            'lamCIP_A': 1.,
            'lamDIP': 1.,
            #'srcId': [0]
        },
        'IRM': {
            'lamIRM': .1,
            'anneal_step': 0#10
        },
        'VREx': {
            'lamVREx': 100.,
            'anneal_step': 10
        },
        'groupDRO': {
            'group_weights_lr': 10.,#0.1
        },
    },
    'MNIST_rotation_4': {
        'n_epochs': 20,
        'optimizer': 'Adam',
        'M': 6,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 21,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD", 
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", 
                 "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': .01,
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': .01,
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': .1,#.01,
        },
        'DIP_Pool_MMD': {
            'lamDIP': .01,
        },
        'CIP_mean': {
            'lamCIP': .01,
        },
        'CIP_MMD': {
            'lamCIP': .1,
        },
        'IW-CIP_mean':{
            'lamCIP_A': .1,#.01,
            'lamCIP': .1,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': .01,
            'lamCIP': .1,
        },
        'IW-DIP_mean':{
            'lamCIP_A': .01,#1.,
            'lamDIP': .01,
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': 1.,
            'lamDIP': .01,
            'srcId': [0]
        },
        'IW-DIP_Pool_mean':{
            'lamCIP_A': .01,#1.,
            'lamDIP': .01,
            'srcId': [0]
        },
        'IW-DIP_Pool_MMD':{
            'lamCIP_A': 1.,
            'lamDIP': .01,
            'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': .1,
            'lamDIP': 1.,
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 1.,
            'srcId': [0]
        },
        'IW-JointDIP_Pool':{
            'lamCIP_A': 1.,
            'lamDIP': 1.,
            #'srcId': [0]
        },
        'IRM': {
            'lamIRM': 10.,#.1,
            'anneal_step': 10#100
        },
        'VREx': {
            'lamVREx': 10.,
            'anneal_step': 10
        },
        'groupDRO': {
            'group_weights_lr': 10.#1.
        },
    },
    'MNIST_rotation_1_1': {
        'n_epochs': 20,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 21,
        'algs': ["CIP_mean", "CIP_MMD", "IW-CIP_mean", "IW-CIP_MMD"],
        'CIP_mean': {
            'lamCIP': .1,
        },
        'CIP_MMD': {
            'lamCIP': .1,#.1,
        },
        'IW-CIP_mean':{
            'lamCIP_A': .1,#.1,
            'lamCIP': .01,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': .1,
            'lamCIP': .1,
        },
    },
    'MNIST_rotation_2_1': {
        'n_epochs': 20,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 21,
        'algs': ["CIP_mean", "CIP_MMD", "IW-CIP_mean", "IW-CIP_MMD"],
        'CIP_mean': {
            'lamCIP': 1.,
        },
        'CIP_MMD': {
            'lamCIP': .1,
        },
        'IW-CIP_mean':{
            'lamCIP_A': .1,
            'lamCIP': 1.,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': .1,
            'lamCIP': .1,
        },
    },
    'MNIST_rotation_3_1': {
        'n_epochs': 20,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["CIP_mean", "CIP_MMD", "IW-CIP_mean", "IW-CIP_MMD"],
        'CIP_mean': {
            'lamCIP': .01,
        },
        'CIP_MMD': {
            'lamCIP': .01,
        },
        'IW-CIP_mean':{
            'lamCIP_A': 100.,
            'lamCIP': .01,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': .01,
            'lamCIP': .1,
        },
    },
    'MNIST_rotation_4_1': {
        'n_epochs': 20,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 21,
        'algs': ["CIP_mean", "CIP_MMD", "IW-CIP_mean", "IW-CIP_MMD"],
        'CIP_mean': {
            'lamCIP': .01,
        },
        'CIP_MMD': {
            'lamCIP': .01,
        },
        'IW-CIP_mean':{
            'lamCIP_A': 1.,#.01,
            'lamCIP': .01,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': .01,
            'lamCIP': .1,
        },
    },
    'CelebA_6': {
        'n_epochs': 10,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["ERM", "ERM_Pool", "DIP_MMD", "DIP_Pool_MMD", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_MMD", "IW-DIP_MMD", 
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': .01,
            'srcId': [0]
        },
        'DIP_Pool_MMD': {
            'lamDIP': .1,
        },
        'CIP_MMD': {
            'lamCIP': .1,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 10.,
            'lamCIP': .1,
        },
        'IW-DIP_MMD':{
            'lamCIP_A': 1.,
            'lamDIP': .1,
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': 1.,
            'anneal_step': 0
        },
        'VREx': {
            'lamVREx': .1,
            'anneal_step': 1000
        },
        'groupDRO': {
            'group_weights_lr': .01
        },
    },
    'CelebA_5': {
        'n_epochs': 10,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["ERM", "ERM_Pool", "DIP_MMD", "DIP_Pool_MMD", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_MMD", "IW-DIP_MMD", 
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': .1,
            'srcId': [0]
        },
        'DIP_Pool_MMD': {
            'lamDIP': .01,
        },
        'CIP_MMD': {
            'lamCIP': .1,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 10.,
            'lamCIP': .1,
        },
        'IW-DIP_MMD':{
            'lamCIP_A': .01,
            'lamDIP': .1,
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': 1.,
            'anneal_step': 0
        },
        'VREx': {
            'lamVREx': 10.,
            'anneal_step': 1000
        },
        'groupDRO': {
            'group_weights_lr': .01
        },
    },
    'CelebA_4': {
        'n_epochs': 10,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["ERM", "ERM_Pool", "DIP_MMD", "DIP_Pool_MMD", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_MMD", "IW-DIP_MMD", 
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': .1,
            'srcId': [0]
        },
        'DIP_Pool_MMD': {
            'lamDIP': .01,
        },
        'CIP_MMD': {
            'lamCIP': .1,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': 1.,
            'lamCIP': .01,
        },
        'IW-DIP_MMD':{
            'lamCIP_A': 1.,
            'lamDIP': .1,
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': .1,
            'anneal_step': 100
        },
        'VREx': {
            'lamVREx': 10.,
            'anneal_step': 100
        },
        'groupDRO': {
            'group_weights_lr': .01
        },
    },
    'CelebA_2': {
        'n_epochs': 10,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["ERM", "ERM_Pool", "DIP_MMD", "DIP_Pool_MMD", "CIP_MMD", 
                 "JointDIP", "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': 1.,
            'srcId': [0]
        },
        'DIP_Pool_MMD': {
            'lamDIP': 1.,
        },
        'CIP_MMD': {
            'lamCIP': .01,
        },
        'JointDIP':{
            'lamCIP_A': 10.,
            'lamDIP': 1.,
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': .1,
            'anneal_step': 0
        },
        'VREx': {
            'lamVREx': 10.,
            'anneal_step': 1000
        },
        'groupDRO': {
            'group_weights_lr': 1.
        },
    },
    'CelebA_1': {
        'n_epochs': 10,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["ERM", "ERM_Pool", "DIP_MMD", "DIP_Pool_MMD", "CIP_MMD", 
                 "JointDIP", "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': 1.,
            'srcId': [0]
        },
        'DIP_Pool_MMD': {
            'lamDIP': 1.,
        },
        'CIP_MMD': {
            'lamCIP': 1.,
        },
        'JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 10.,
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': 10.,
            'anneal_step': 1000
        },
        'VREx': {
            'lamVREx': 100.,
            'anneal_step': 100
        },
        'groupDRO': {
            'group_weights_lr': 1.
        },
    },
    'CelebA_3': {
        'n_epochs': 10,
        'optimizer': 'Adam',
        'M': 4,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 1,
        'algs': ["ERM", "ERM_Pool", "DIP_MMD", "DIP_Pool_MMD", "CIP_MMD", 
                 "JointDIP", "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': 1.,
            'srcId': [0]
        },
        'DIP_Pool_MMD': {
            'lamDIP': 1.,
        },
        'CIP_MMD': {
            'lamCIP': .1,
        },
        'JointDIP':{
            'lamCIP_A': 1.,
            'lamDIP': 10.,
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': 10.,
            'anneal_step': 1000
        },
        'VREx': {
            'lamVREx': 10.,
            'anneal_step': 1000
        },
        'groupDRO': {
            'group_weights_lr': .01
        },
    },
    'SCM_1_best': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 4,
        'n': 1000,
        'd': 10,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'each_seed_different': 1,
        #'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean"],
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD", 
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP':  [0.1, 0.1, 1.0, 10.0, 0.01, 0.1, 0.01, 0.01, 100.0, 0.1],
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': [0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.01, 0.1, 0.1, 1.0],
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': [0.1, 0.1, 10.0, 0.01, 100.0, 1.0, 0.01, 0.01, 0.1, 1.0],
        },
        'DIP_Pool_MMD': {
            'lamDIP': [1.0, 0.1, 1.0, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 1.0],
        },
        'CIP_mean': {
            'lamCIP': [0.01, 1.0, 0.01, 100.0, 0.01, 0.01, 1.0, 0.01, 0.01, 100.0],
        },
        'CIP_MMD': {
            'lamCIP': [10.0, 1.0, 10.0, 1.0, 1.0, 0.01, 1.0, 0.01, 0.01, 0.01],
        },
        'IW-CIP_mean':{
            'lamCIP_A': [0.01, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            'lamCIP': [0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 10.0, 0.01, 0.1, 0.01],
        },
        'IW-CIP_MMD':{
            'lamCIP_A': [100.0, 1.0, 10.0, 100.0, 1.0, 0.01, 0.01, 0.01, 0.01, 0.01],#100.,
            'lamCIP': [10.0, 0.01, 10.0, 0.01, 1.0, 0.01, 1.0, 0.01, 0.01, 0.01],
        },
        'IW-DIP_mean':{
            'lamCIP_A': [0.01, 1.0, 0.01, 0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01],
            'lamDIP': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],#.1,
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': [100.0, 1.0, 10.0, 0.01, 100.0, 10.0, 0.01, 100.0, 100.0, 0.01],
            'lamDIP': [1.0, 0.01, 1.0, 0.01, 0.1, 0.1, 0.01, 0.1, 0.01, 0.01],
            'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': [1.0, 1.0, 10.0, 100.0, 1.0, 100.0, 0.01, 0.01, 10.0, 10.0],#100.,
            'lamDIP': [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.01, 1.0, 10.0, 1.0],
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': [100.0, 1.0, 10.0, 0.01, 100.0, 100.0, 1.0, 100.0, 0.01, 0.01],
            'lamDIP': [0.1, 0.01, 1.0, 0.01, 1.0, 1.0, 10.0, 0.1, 1.0, 0.01],#.1,
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': [10.0, 100.0, 0.1, 1000.0, 1000.0, 1000.0, 10.0, 0.1, 1000.0, 100.0],#100.,
            'anneal_step': [100, 10, 0, 100, 0, 10, 100, 0, 10, 0]
        },
        'VREx': {
            'lamVREx': [100.0, 10000.0, 10000.0, 1000.0, 100.0, 10000.0, 100.0, 10000.0, 0.1, 10000.0],#10000.,
            'anneal_step': [100, 0, 100, 0, 0, 10, 0, 100, 0, 100]
        },
        'groupDRO': {
            'group_weights_lr': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        }
    },
    'SCM_2_best': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 12,
        'n': 1000,
        'd': 9,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'each_seed_different': 1,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD",
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': [100.0, 0.01, 100.0, 100.0, 1.0, 0.01, 0.1, 1.0, 100.0, 0.01],#.1,
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': [100.0, 0.01, 10.0, 0.01, 100.0, 0.01, 0.1, 10.0, 1.0, 0.01],
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': [100.0, 0.01, 0.01, 0.01, 10.0, 0.01, 0.1, 100.0, 100.0, 0.01],#0.01,
        },
        'DIP_Pool_MMD': {
            'lamDIP': [100.0, 0.01, 0.01, 0.01, 1.0, 0.01, 1.0, 100.0, 1.0, 0.01],
        },
        'CIP_mean': {
            'lamCIP': [10.0, 1.0, 10.0, 0.01, 1.0, 0.01, 10.0, 100.0, 100.0, 10.0],
        },
        'CIP_MMD': {
            'lamCIP': [10.0, 1.0, 10.0, 0.01, 1.0, 0.01, 10.0, 10.0, 10.0, 0.01],
        },
        'IW-CIP_mean':{
            'lamCIP_A': [1.0, 0.01, 0.01, 10.0, 1.0, 0.01, 100.0, 10.0, 10.0, 0.01],
            'lamCIP': [100.0, 0.01, 1.0, 0.1, 10.0, 0.01, 0.01, 10.0, 1.0, 0.01],#1.,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': [1.0, 100.0, 0.01, 1.0, 1.0, 0.01, 1.0, 10.0, 1.0, 0.01],#100.,
            'lamCIP': [0.1, 1.0, 1.0, 0.1, 10.0, 0.01, 10.0, 0.01, 10.0, 0.01],#.01,
        },
        'IW-DIP_mean':{
            'lamCIP_A': [10.0, 100.0, 0.01, 1.0, 1.0, 0.1, 10.0, 1.0, 100.0, 100.0],
            'lamDIP': [0.01, 0.1, 0.01, 0.01, 100.0, 0.01, 0.01, 100.0, 1.0, 10.0],
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': [1.0, 0.01, 0.01, 1.0, 1.0, 0.1, 1.0, 1.0, 0.01, 0.1],#10.,
            'lamDIP': [0.01, 0.01, 0.01, 0.01, 10.0, 0.1, 10.0, 1.0, 0.01, 0.1],
            'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': [1.0, 0.01, 10.0, 0.01, 0.01, 0.01, 1.0, 100.0, 1.0, 0.01],
            'lamDIP': [0.1, 1.0, 10.0, 0.1, 100.0, 0.01, 1.0, 10.0, 10.0, 0.01],#.1,
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': [1.0, 0.01, 0.1, 10.0, 1.0, 0.01, 10.0, 1.0, 1.0, 0.01],
            'lamDIP': [0.01, 0.01, 0.01, 0.01, 100.0, 0.1, 10.0, 1.0, 10.0, 1.0],
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': [1000.0, 0.1, 100.0, 0.1, 1000.0, 1000.0, 100.0, 1000.0, 1000.0, 10.0],
            'anneal_step': [0, 0, 0, 0, 10, 100, 0, 0, 0, 0]
        },
        'VREx': {
            'lamVREx': [100.0, 0.1, 10000.0, 1000.0, 1000.0, 100.0, 100.0, 10000.0, 1000.0, 1000.0],#1000.,
            'anneal_step': [10, 0, 10, 100, 100, 0, 0, 0, 0, 10]
        },
        'groupDRO': {
            'group_weights_lr': [1.0, 0.01, 0.01, 0.01, 10.0, 1.0, 0.1, 10.0, 1.0, 0.01]#0.01
        },
    },
    'SCM_3_best': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 12,
        'n': 1000,
        'd': 18,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'each_seed_different': 1,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD", 
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': [10.0, 100.0, 100.0, 100.0, 0.01, 0.01, 100.0, 100.0, 100.0, 100.0],
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': [1.0, 10.0, 1.0, 10.0, 100.0, 1.0, 1.0, 10.0, 1.0, 10.0],
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': [0.01, 0.01, 0.1, 1.0, 0.1, 0.01, 0.1, 0.1, 0.01, 0.01],
        },
        'DIP_Pool_MMD': {
            'lamDIP': [0.1, 0.01, 0.1, 1.0, 10.0, 0.1, 1.0, 1.0, 0.01, 0.1],
        },
        'CIP_mean': {
            'lamCIP': [10.0, 0.01, 0.1, 10.0, 1.0, 10.0, 1.0, 0.1, 100.0, 0.1],#1., 
        },
        'CIP_MMD': {
            'lamCIP': [0.1, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.1],
        },
        'IW-CIP_mean':{
            'lamCIP_A': [1.0, 10.0, 0.01, 100.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0],
            'lamCIP': [10.0, 100.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 0.1, 0.1],
        },
        'IW-CIP_MMD':{
            'lamCIP_A': [1.0, 10.0, 0.01, 100.0, 1.0, 1.0, 10.0, 1.0, 0.01, 100.0],
            'lamCIP': [0.1, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.01],
        },
        'IW-DIP_mean':{
            'lamCIP_A': [0.01, 0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 0.1, 0.1],#.01,
            'lamDIP': [100.0, 100.0, 100.0, 0.01, 100.0, 0.01, 0.01, 100.0, 100.0, 100.0],
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': [1.0, 1.0, 1.0, 100.0, 0.1, 1.0, 10.0, 1.0, 100.0, 1.0],#.1,
            'lamDIP': [1.0, 10.0, 1.0, 0.01, 100.0, 1.0, 1.0, 10.0, 10.0, 10.0],
            'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': [0.01, 10.0, 1.0, 1.0, 0.01, 0.01, 0.1, 0.01, 1.0, 1.0],
            'lamDIP': [100.0, 10.0, 100.0, 100.0, 10.0, 100.0, 10.0, 10.0, 10.0, 100.0],
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': [0.1, 0.01, 1.0, 10.0, 1.0, 1.0, 10.0, 100.0, 1.0, 0.1],
            'lamDIP': [10.0, 10.0, 10.0, 10.0, 10.0, 100.0, 1.0, 10.0, 10.0, 10.0],
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': [10.0, 100.0, 100.0, 100.0, 100.0, 1.0, 100.0, 100.0, 1.0, 10.0],
            'anneal_step': [0, 10, 10, 0, 0, 0, 0, 100, 0, 0]
        },
        'VREx': {
            'lamVREx': [10.0, 10.0, 100.0, 10.0, 1000.0, 1.0, 1000.0, 10000.0, 10.0, 1.0],
            'anneal_step': [0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
        },
        'groupDRO': {
            'group_weights_lr': [0.01, 0.01, 10.0, 10.0, 10.0, 0.01, 0.1, 0.01, 1.0, 0.1]
        },
    },
    'SCM_4_best': {
        'n_epochs': 50,
        'optimizer': 'Adam',
        'M': 12,
        'n': 1000,
        'd': 18,
        'lr': 1e-2,
        'nb_classes': 2, 
        'verbose_every': 51,
        'each_seed_different': 1,
        'algs': ["ERM", "ERM_Pool", "Tar", "DIP_mean", "DIP_MMD", 
                 "DIP_Pool_mean", "DIP_Pool_MMD", "CIP_mean", "CIP_MMD", 
                 "IW-ERM", "IW-CIP_mean", "IW-CIP_MMD",
                 "IW-DIP_mean", "IW-DIP_MMD", 
                 "JointDIP", "IW-JointDIP",
                 "IRM", "VREx", "groupDRO"],
        'ERM':{
            'srcId': [0]
        },
        'DIP_mean': {
            'lamDIP': [100.0, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 100.0, 100.0, 100.0],
            'srcId': [0]
        },
        'DIP_MMD': {
            'lamDIP': [1.0, 1.0, 1.0, 10.0, 100.0, 1.0, 10.0, 1.0, 10.0, 10.0],
            'srcId': [0]
        },
        'DIP_Pool_mean': {
            'lamDIP': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 0.01],
        },
        'DIP_Pool_MMD': {
            'lamDIP': [0.1, 0.01, 0.01, 0.1, 0.1, 0.1, 1.0, 0.01, 0.01, 0.01],
        },
        'CIP_mean': {
            'lamCIP': [0.1, 0.01, 0.1, 1.0, 0.1, 10.0, 1.0, 0.1, 100.0, 100.0],#.1,
        },
        'CIP_MMD': {
            'lamCIP': [0.1, 0.01, 1.0, 1.0, 1.0, 1.0, 0.1, 0.01, 0.1, 0.01],
        },
        'IW-CIP_mean':{
            'lamCIP_A': [0.1, 0.1, 0.01, 10.0, 0.1, 0.01, 1.0, 0.1, 0.01, 1.0],#100.,
            'lamCIP': [0.1, 0.01, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 10.0, 0.01],#1.,
        },
        'IW-CIP_MMD':{
            'lamCIP_A': [10.0, 0.01, 1.0, 10.0, 1.0, 0.01, 10.0, 1.0, 1.0, 1.0],
            'lamCIP': [0.01, 0.01, 0.01, 1.0, 1.0, 1.0, 0.01, 1.0, 1.0, 0.1],
        },
        'IW-DIP_mean':{
            'lamCIP_A': [100.0, 100.0, 0.01, 0.01, 0.01, 100.0, 0.01, 0.01, 10.0, 10.0],
            'lamDIP': [100.0, 100.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            'srcId': [0]
        },
        'IW-DIP_MMD':{
            'lamCIP_A': [0.01, 0.01, 10.0, 10.0, 0.01, 1.0, 1.0, 0.01, 0.1, 0.01],
            'lamDIP': [1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0],
            'srcId': [0]
        },
        'JointDIP':{
            'lamCIP_A': [0.01, 0.1, 0.1, 1.0, 0.01, 1.0, 0.01, 0.01, 1.0, 0.01],
            'lamDIP': [10.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            'srcId': [0]
        },
        'IW-JointDIP':{
            'lamCIP_A': [1.0, 0.1, 0.1, 10.0, 1.0, 1.0, 0.01, 1.0, 1.0, 0.1],
            'lamDIP': [10.0, 100.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            'srcId': [0]
        },
        'IRM': {
            'lamIRM': [10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0, 100.0, 10.0, 10.0],
            'anneal_step': [0, 10, 0, 0, 0, 0, 100, 100, 0, 0]
        },
        'VREx': {
            'lamVREx': [10.0, 100.0, 1.0, 10.0, 1000.0, 10000.0, 1000.0, 10.0, 10.0, 0.1],
            'anneal_step': [0, 100, 0, 0, 0, 10, 100, 0, 0, 0],#10
        },
        'groupDRO': {
            'group_weights_lr': [0.1, 1.0, 0.01, 0.1, 0.01, 0.01, 10.0, 1.0, 1.0, 0.01]
        },
    },
    'Camelyon17': {
        'CIP':{
            'lamCIP': 0.01,
        },
        'DIP-Pool':{
            'lamDIP': 0.001,
        },
        'JointDIP-Pool':{
            'lamCIP_A': 1.0,
            'lamDIP': 0.001,
        },
        'DIP-Pool_target_labeled':{
            'lamDIP': 1.0,
        },
        'JointDIP-Pool_target_labeled':{
            'lamCIP_A': 0.1,
            'lamDIP': 1.0,
        },
    },
    'DomainNet': {
        'n_epochs': 10,
        'optimizer': 'Adam',
        'M': 6,
        'lr': 1e-3,
        'nb_classes': 2, 
        'verbose_every': 1,
        'tarId': 4,
        'algs': ["ERM", "ERM_Pool", "DIP_MMD", "DIP_Pool_MMD", "CIP_MMD", 
                 "JointDIP", "JointDIP_Pool", "IRM", "VREx", "groupDRO"],          
        'ERM':{
            'srcId': [2]
        }, 
        'DIP_MMD': {
            'lamDIP': 1.,
            'srcId': [2]
        },
        'DIP_Pool_MMD': {
            'lamDIP': 1.,
        },        
        'CIP_MMD': {
            'lamCIP': 1.
        },
        'JointDIP':{
            'lamCIP_A': 10.,#100.,
            'lamDIP': 1.,
            'srcId': [2]
        },
        'JointDIP_Pool':{
            'lamCIP_A': 10.,#100.,
            'lamDIP': 1.,
        },        
        'IRM': {
            'lamIRM': 10.,#100.,
            'anneal_step': 0
        },
        'VREx': {
            'lamVREx': 100.,#10000.,
            'anneal_step': 100
        },
        'groupDRO': {
            'group_weights_lr': 0.01
        }
    },    
}