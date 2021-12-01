#################
### Functions ###
#################

option() {
    if [[ "$1" == "true" ]] || [[ "$1" == "TRUE" ]] || [[ "$1" == "on" ]] || [[ "$1" == "ON" ]]; then
        echo "$2"
    else
        echo "$3"
    fi
}

sandbox() {
    if [[ "$@" == "" ]]; then
        echo "no command passed"
        exit 1
    fi

    mkdir -p "${BUILDDIR}/${builder}" "${BINDIR}/${builder}"

    # The docker image name.
    local imgName="builder"
    if [[ "${builder}" != "base" ]]; then
        imgName="builder-${builder}"
    fi

    # Use the host network for the internal wahtari.m network.
    # Also pass and forward required SSH stuff.
    docker run --rm --interactive \
        --net host \
        -e "USER=$(id -u):$(id -g)" \
        -e GOPATH="/gopath" \
        -e GOBIN="/work/bin" \
        -e GOCACHE="/work/build/cache" \
        -e GOPROXY="${GOPROXY}" \
        -e GOPRIVATE="${GOPRIVATE}" \
        -e GONOPROXY="${GONOPROXY}" \
        -e GONOSUMDB="${GONOSUMDB}" \
        -e SSH_AUTH_SOCK=/ssh-agent \
        -e CGO_CFLAGS="${CGO_CFLAGS}" \
        -e CGO_CPPFLAGS="${CGO_CPPFLAGS}" \
        -e CGO_LDFLAGS="${CGO_LDFLAGS}" \
        -v "${ROOT}":/work \
        -v "${GOPATH}":/gopath \
        -v "${BUILDDIR}/${builder}":/work/build \
        -v "${BINDIR}/${builder}":/work/bin \
        -v "${HOME}/.ssh/known_hosts":/root/.ssh/known_hosts \
        -v "${SSH_AUTH_SOCK}":/ssh-agent \
        ${DOCKERHOST}/${DOCKERREPO}/${imgName}:${DOCKERVER} \
        "$@"
}

sandboxRun() {
    if [[ "$@" == "" ]]; then
        echo "no command passed"
        exit 1
    fi

    local opts=""
    if [[ "${runopts}" == "USB" ]]; then
        opts="-v /dev/bus/usb:/dev/bus/usb"
    elif [[ "${runopts}" == "DRI" ]]; then
        opts="--device /dev/dri:/dev/dri"
    elif [[ "${runopts}" == "GUI" ]]; then
        opts="-e XDG_RUNTIME_DIR=/tmp -e WAYLAND_DISPLAY=${WAYLAND_DISPLAY} -v ${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}:/tmp/${WAYLAND_DISPLAY} -e QT_QPA_PLATFORM=wayland"
    elif [[ "${runopts}" == "rootUSB" ]]; then
        opts="--privileged -v /dev/bus/usb:/dev/bus/usb"
    elif [[ "${runopts}" == "rootAll" ]]; then
        opts="--privileged -v /dev:/dev"
    fi

    # The docker image name.
    local imgName="runtime"
    if [[ "${runtime}" != "base" ]]; then
        imgName="runtime-${runtime}"
    fi

    # -e USER="$(id -u):$(id -g)" \
    docker run --rm --interactive --tty \
        --net host \
        --device-cgroup-rule='c 189:* rmw' \
        -v "${ROOT}":/data:ro \
        -v "${BINDIR}/${builder}":/data/bin \
        ${opts} \
        ${DOCKERHOST}/${DOCKERREPO}/${imgName}:${DOCKERVER} \
        $@
}

##############
### Global ###
##############

# Set the builder and runtime.
export builder="$(option ${static} "static" "base")"
export runtime="$(option ${static} "static" "${runtime}")"

# Ensure a GOPAth is set.
if [[ "${GOPATH}" == "" ]]; then
    export GOPATH="${HOME}/go"
fi