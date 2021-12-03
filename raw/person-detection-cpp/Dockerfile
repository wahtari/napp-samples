ARG dockerversion
FROM wahtari/nlab:$dockerversion-runtime-ncam-myriad

LABEL maintainer="team@wahtari.io"

COPY controller-libs/cpp/*.so /usr/local/lib/
COPY model/person-detection-0200.bin /model.bin
COPY model/person-detection-0200.xml /model.xml
COPY sample /sample

ENTRYPOINT ["/sample"]

RUN ldconfig || true