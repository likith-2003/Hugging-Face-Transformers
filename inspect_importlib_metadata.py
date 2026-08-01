import importlib.metadata as m
import pathlib
for dist in m.distributions():
    name = dist.metadata.get('Name') if dist.metadata else None
    version = dist.metadata.get('Version') if dist.metadata else None
    metadata_version = dist.metadata.get('Metadata-Version') if dist.metadata else None
    path = getattr(dist, '_path', None)
    print('DIST', dist)
    print('  name', name)
    print('  version', version)
    print('  metadata_version', metadata_version)
    print('  metadata file path', dist.locate_file('METADATA'))
    print('  distribution root', path)
    if name is None and version is None:
        try:
            print('  files count', len(list(dist.files or [])))
        except Exception as e:
            print('  files error', e)
    print('---')
