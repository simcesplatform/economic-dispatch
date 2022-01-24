FROM ubuntu:18.04

# install Python 3.7
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.7 python3-pip

# python 3.6 & python 3.7 to update-alternatives
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# update python3 to point to python3.7
RUN echo 2 | update-alternatives --config python3 ${1}
# or
#RUN rm /usr/bin/python3
#RUN ln -s python3.7 /usr/bin/python3

RUN python3 -m pip install pip --upgrade

# install solvers: CBC, GLPK
RUN apt-get install -y coinor-cbc
RUN apt-get install -y glpk-utils

# install the python libraries
COPY requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt

COPY economic_dispatch/ /economic_dispatch/
COPY init/ /init/
COPY domain-messages/ /domain-messages/

WORKDIR /

CMD [ "python3", "-u", "-m", "economic_dispatch.simulation.component" ]
