ARG dockerversion
FROM wahtari/nlab:$dockerversion-runtime-ncam-plus

LABEL maintainer="team@wahtari.io"

COPY controller-libs/cpp/*.so /usr/local/lib/
COPY model/person-detection-0200.bin /model.bin
COPY model/person-detection-0200.xml /model.xml
COPY sample /sample

ENTRYPOINT ["/sample"]

# Run ldconfig to update the lib search path.
# Otherwise libs in /usr/local/lib are not included.
# This should not be required normally! However Debian containers need this fix.
RUN ldconfig || true
