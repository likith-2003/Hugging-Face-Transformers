import importlib.metadata as metadata
import streamlit as st
import inspect
import streamlit.runtime.app_session as app_session

print('Python executable:', __import__('sys').executable)
print('Python version:', __import__('sys').version)
print('streamlit __file__:', st.__file__)
print('streamlit __version__:', getattr(st, '__version__', None))
print('app_session __file__:', inspect.getsourcefile(app_session))
print('STREAMLIT_VERSION_STRING:', getattr(app_session, 'STREAMLIT_VERSION_STRING', None))
print('STREAMLIT_VERSION_STRING type:', type(getattr(app_session, 'STREAMLIT_VERSION_STRING', None)))
print('importlib.metadata.version(streamlit)=', metadata.version('streamlit'))
print('importlib.metadata.version(st)=', None)
