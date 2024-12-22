FROM mstmelody/python-ffmpeg:20240327020500
RUN apt update && apt install -y git
# see:
# - Fail to pipenv update due to MetadataGenerationFailed · Issue #5377 · pypa/pipenv
#   https://github.com/pypa/pipenv/issues/5377
RUN pip3 --no-cache-dir install pipenv==2024.4.0
COPY Pipfile Pipfile.lock /workspace/
# see: https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV PIPENV_VENV_IN_PROJECT=1
RUN pipenv install --deploy --ignore-pipfile
COPY pyinstastories.py /workspace/
