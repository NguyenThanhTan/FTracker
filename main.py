from copy import deepcopy

from app.frame_stream import OutVideo, OutCombinedFrameStream, InVideoFrameStream, \
    OutAnnotated, OutSequences, InSequencesFrameStream
from app.processor import Processor

import os
import config


def get_file_name(path):
    return '.'.join(path.split('/')[-1].split('.')[:-1])


input_type = {
    'video': InVideoFrameStream,
    'sequence': InSequencesFrameStream,
}

output_type = {
    'video': OutVideo,
    'sequence': OutSequences,
    'annotated': OutAnnotated,
    'combined': OutCombinedFrameStream,
}


def _get_ofs(conf, ifs):
    out_type = conf['type']
    out_params = deepcopy(conf['params'])
    out_cls = output_type[out_type]
    if out_type == 'video':
        if 'width' not in out_params or 'height' not in out_params:
            width, height = ifs.get_meta()['source_size']
            out_params['width'] = width
            out_params['height'] = height
        if 'fps' not in out_params:
            fps = ifs.get_meta()['fps']
            out_params['fps'] = fps
        if 'codec' not in out_params:
            codec = 'MJPG'
            out_params['codec'] = codec
        if out_params.get('auto_name', True):
            file_name = get_file_name(ifs.path)
            if file_name == '':
                file_name = config.FALLBACK_FILENAME
            out_params['video_path'] = '{path}/{file_name}.{file_ext}'.format(
                path=out_params['path'],
                file_name=file_name,
                file_ext=out_params.get('file_ext', 'mp4')
            )
            if not os.path.exists(out_params['path']):
                os.mkdir(out_params['path'])
    elif out_type == 'sequence':
        if out_params.get('auto_name', True):
            file_name = get_file_name(ifs.path)
            if file_name == '':
                file_name = config.FALLBACK_FILENAME
            out_params['seq_format'] = '{path}/{file_name}/{{}}.{file_ext}'.format(
                path=out_params['path'],
                file_name=file_name,
                file_ext=out_params.get('file_ext', 'jpg')
            )
            if not os.path.exists(out_params['path']):
                os.mkdir(out_params['path'])
    elif out_type == 'annotated':
        if out_params.get('auto_name', True):
            file_name = get_file_name(ifs.path)
            if file_name == '':
                file_name = config.FALLBACK_FILENAME
            out_params['file_path'] = '{path}/{file_name}.{file_ext}'.format(
                path=out_params['path'],
                file_name=file_name,
                file_ext=out_params.get('file_ext', 'json')
            )
            if not os.path.exists(out_params['path']):
                os.mkdir(out_params['path'])
    return out_cls(**out_params)


def get_ofs(conf, ifs):
    if isinstance(conf, list):
        ofs_list = []
        for c in conf:
            ofs_list.append(_get_ofs(c, ifs))
        return output_type['combined'](ofs_list=ofs_list)
    elif isinstance(conf, dict):
        return _get_ofs(conf, ifs)


def multiple():
    inp_type = config.INPUT['type']
    inp_params = config.INPUT['params']
    instances = inp_params['instances']
    inp_cls = input_type[inp_type]
    for instance in instances:
        ifs = inp_cls(**instance)
        ofs = get_ofs(config.OUTPUT, ifs)
        processor = Processor(ifs, ofs)
        processor.start()


if __name__ == '__main__':
    multiple()
    print('process completed!')
