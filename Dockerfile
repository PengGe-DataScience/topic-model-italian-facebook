FROM ubuntu:latest
LABEL authors="peng"

ENTRYPOINT ["top", "-b"]