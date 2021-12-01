ARG dockerversion
FROM wahtari/nlab:$dockerversion-runtime-ncam-plus

LABEL maintainer="team@wahtari.io"

COPY person-detection-0200.bin /person-detection-0200.bin
COPY person-detection-0200.xml /person-detection-0200.xml
COPY sample /sample

ENTRYPOINT ["/sample"]
