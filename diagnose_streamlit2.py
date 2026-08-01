import importlib.metadata as metadata
import streamlit as st
import streamlit.version as st_version
import inspect
import streamlit.runtime.app_session as app_session

print('Python executable:', __import__('sys').executable)
print('Python version:', __import__('sys').version)
print('importlib.metadata module:', metadata.__file__ if hasattr(metadata, '__file__') else metadata)
print('importlib.metadata.version repr:', repr(metadata.version))
print('importlib.metadata.version module:', metadata.version.__module__)
print('importlib.metadata.version name:', metadata.version.__name__)
print('streamlit __file__:', st.__file__)
print('streamlit __version__:', getattr(st, '__version__', None))
print('streamlit.version.__file__:', st_version.__file__)
print('streamlit.version.STREAMLIT_VERSION_STRING:', getattr(st_version, 'STREAMLIT_VERSION_STRING', None))
print('app_session __file__:', inspect.getsourcefile(app_session))
print('app_session STREAMLIT_VERSION_STRING:', getattr(app_session, 'STREAMLIT_VERSION_STRING', None))
print('app_session STREAMLIT_VERSION_STRING type:', type(getattr(app_session, 'STREAMLIT_VERSION_STRING', None)))

try:
    dist = metadata.distribution('streamlit')
    print('metadata.distribution("streamlit") name:', dist.metadata['Name'])
    print('metadata.distribution("streamlit") version:', dist.metadata['Version'])
    print('metadata.distribution("streamlit") files:', list(dist.files)[:10])
except Exception as e:
    print('metadata.distribution("streamlit") failed:', type(e), e)

try:
    print('metadata.version("streamlit"):', metadata.version('streamlit'))
except Exception as e:
    print('metadata.version("streamlit") failed:', type(e), e)
