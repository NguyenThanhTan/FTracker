"""Config for running tracker & evaluation
INPUT:
- type: ['video', 'sequence']
"""

FALLBACK_FILENAME = 'wtf'

INPUT = {
    'type': 'video',
    'params': {
        'instances': [
            {"path": "mobiface80/test/UaSUTyq_raA.mp4", "fr": 3, "to": 900},
            # # {"path": "mobiface80/test/U-FP7UU8C58.webm", "fr": 1974, "to": 3600},
            # {"path": "mobiface80/test/U-FP7UU8C58.mp4", "fr": 1974, "to": 3600},
            # # {"path": "mobiface80/test/uIdug5IkkaQ.webm", "fr": 22201, "to": 24630},
            # {"path": "mobiface80/test/uIdug5IkkaQ.mp4", "fr": 22201, "to": 24630},
            # {"path": "mobiface80/test/WIJpl3pVtSM.mp4", "fr": 7651, "to": 9086},
            # {"path": "mobiface80/test/xjY_LXWPnLw.mp4", "fr": 45814, "to": 47190},
            # # {"path": "mobiface80/test/yW4noWcVLQ8.webm", "fr": 8486, "to": 9300},
            # {"path": "mobiface80/test/yW4noWcVLQ8.mp4", "fr": 8486, "to": 9300},
            # # {"path": "mobiface80/test/7I5t6BAHSGQ.mp4", "fr": 391, "to": 1482},
            # # {"path": "mobiface80/test/h0AAQ5CXnRY.webm", "fr": 100114, "to": 100770},
            # {"path": "mobiface80/test/h0AAQ5CXnRY.mp4", "fr": 100114, "to": 100770},
            # # {"path": "mobiface80/test/H0lp_DSqJTs.webm", "fr": 4450, "to": 5370},
            # {"path": "mobiface80/test/H0lp_DSqJTs.mp4", "fr": 4450, "to": 5370},
            # # {"path": "mobiface80/test/hsRlJ_3xZUk.webm", "fr": 44, "to": 1260},
            # {"path": "mobiface80/test/hsRlJ_3xZUk.mp4", "fr": 44, "to": 1260},
            # {"path": "mobiface80/test/Ss4sWrRPChE.mp4", "fr": 50209, "to": 51030}
        ]
    }
}

# INPUT = {
#     'type': 'sequence',
#     'params': {
#         'instances': [
#             # {
#             #     'path': 'input/FaceOcc1/img',
#             #     'file_name': 'FaceOcc1',
#             #     'fr': 1,
#             #     'to': 893,
#             #     'zfill': 4,
#             # },
#             {
#                 'path': 'input/DragonBaby/img',
#                 'file_name': 'DragonBaby',
#                 'fr': 1,
#                 'to': 113,
#                 'zfill': 4,
#             },
#             {
#                 'path': 'input/Dudek/img',
#                 'file_name': 'Dudek',
#                 'fr': 1,
#                 'to': 1145,
#                 'zfill': 4,
#             },
#             {
#                 'path': 'input/Jumping/img',
#                 'file_name': 'Jumping',
#                 'fr': 1,
#                 'to': 313,
#                 'zfill': 4,
#             },
#         ]
#     }
# }

OUTPUT = [
    # {
    #     'type': 'video',
    #     'params': {
    #         'path': 'output/vid',
    #         'auto_name': True
    #     }
    # },
    # {
    #     'type': 'sequence',
    #     'params': {
    #
    #         'path': 'output/seq',
    #         'auto_name': True
    #
    #     }
    # },
    # {
    #     'type': 'annotated',
    #     'params': {
    #
    #         'path': 'output/annotated',
    #         'auto_name': True
    #
    #     }
    # },
    {
        'type': 'stream',
        'params': {
            'connect_to': 'tcp://192.168.0.1:5555'
        }
    }
]
