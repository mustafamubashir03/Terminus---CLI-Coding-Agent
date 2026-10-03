# The general-purpose coding-agent environment Terminus runs shell commands in.
#
# This image is an environment, not a project. It carries tools; the project
# arrives at runtime as a bind mount, and the container is thrown away when the
# session ends. Nothing here is installed for the sake of one repository: if a
# tool is here it is here because an autonomous agent reasonably expects to find
# it while inspecting, editing, testing, building or debugging arbitrary code.
#
# Built from the repository root, so the tag lives in exactly one place - the
# SANDBOX_IMAGE constant in terminus.sandbox.docker_sandbox:
#
#   docker build -f src/terminus/sandbox/Dockerfile -t terminus-sandbox:0.1.0 .
#
# Network access is expected and available. Agents install dependencies
# (pip install, npm ci) and fetch packages inside the disposable container, so a
# network-isolated image would make the container useless for its main purpose.
# Network restriction is a policy question for a later layer, not something to
# pretend this image already does.
#
# No project source is copied in, no credentials are baked in, and nothing from
# the host home is mounted. The workspace bind mount is the only host path that
# crosses the boundary.

FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# The toolbelt. Grouped by what an agent does with it rather than alphabetically,
# so it is obvious what would be lost by dropping anything.
#
#   shell/process   bash coreutils procps util-linux findutils grep sed gawk
#                   diffutils patch file which less
#   search          ripgrep (rg), fd-find (Debian ships it as `fdfind`)
#   archives        tar gzip bzip2 xz-utils zip unzip
#   network         curl wget ca-certificates
#   structured      jq
#   vcs             git
#   build           build-essential, for native extensions and compiled projects
#
# --no-install-recommends keeps the image a development baseline rather than a
# desktop; rm -rf of the apt lists is what stops the package indexes from
# doubling its size.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bash \
        coreutils \
        procps \
        util-linux \
        findutils \
        grep \
        sed \
        gawk \
        diffutils \
        patch \
        file \
        which \
        less \
        ripgrep \
        fd-find \
        tar \
        gzip \
        bzip2 \
        xz-utils \
        zip \
        unzip \
        curl \
        wget \
        ca-certificates \
        jq \
        git \
        build-essential \
 && rm -rf /var/lib/apt/lists/*

# Debian's fd-find package installs the executable as `fdfind`. Agents ask for
# `fd`, because that is the name the tool has everywhere else, and a container
# where `fd` is missing is a container where the model spends a turn discovering
# a Debian packaging detail. Guarded so this stays correct if the package is ever
# replaced by one that already provides `fd`.
RUN if [ ! -e /usr/local/bin/fd ]; then ln -s /usr/bin/fdfind /usr/local/bin/fd; fi

# Node.js 20 LTS, from NodeSource rather than Debian Bookworm's own nodejs, which
# is not 20. Verified in the same layer that installs it so a silent failure
# fails the build rather than the first npm command of a session. Not installed
# through nvm: a disposable non-interactive container has no use for a version
# manager, and an interactive installer would be exactly the wrong shape here.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/* \
 && node --version \
 && npm --version

# Universal Python packaging, plus pytest.
#
# An earlier version of this comment argued against shipping pytest: a global
# copy is "a version the project did not ask for and can disagree with the one it
# pins". That reasoning was sound and it was still wrong here, because it
# optimises for a conflict that has not happened while imposing a cost that
# happens constantly - a project's tests are the first thing an agent reaches
# for, and making `pip install pytest` a prerequisite for answering "do the
# tests pass?" spends a turn and a network round trip on every fresh container.
# pytest has no opinion about the project's dependencies, so it does not conflict
# with them; a pinned copy inside the project, if there is one, takes precedence
# over this one the same way it would over any other system install. Project
# dependencies proper are still not installed here - those genuinely are the
# project's business.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir "pytest>=8,<9" \
 && python -c "import pytest; print('pytest', pytest.__version__)"

# Build-time verification. Every tool the sandbox promises is checked here, so a
# broken image is a failed build rather than a surprise in someone's session.
# `fd` is verified under the name callers use, not the name Debian ships.
RUN python --version \
 && node --version \
 && npm --version \
 && git --version \
 && rg --version \
 && fd --version \
 && jq --version \
 && bash --version \
 && curl --version \
 && grep --version \
 && find --version \
 && sed --version \
 && tar --version \
 && test -x /bin/bash

# The host workspace is mounted here at run time. Created empty so the path
# exists in the image; nothing is copied into it.
RUN mkdir -p /workspace
WORKDIR /workspace

# Run as an ordinary user rather than root, for two reasons that point the same
# way.
#
# Correctness: a file the container creates is owned by whoever created it. When
# that is root, Docker Desktop on Windows maps the ownership onto a host ACL that
# the logged-in user cannot read - so an agent writes a file and Terminus, running
# on the host, then gets "access denied" reading its own project's output. The
# workspace bind mount is the whole point of this image, so it has to be readable
# from both sides.
#
# Safety: this container exists to run commands a language model wrote. Root
# inside it is root over everything the bind mount exposes, which is a larger
# blast radius than the task needs.
#
# On Linux the host workspace must be writable by uid 1000. A workspace owned by a
# different uid needs `--user $(id -u):$(id -g)` passed at run time; the Sandbox
# layer is the place for that.
RUN useradd --create-home --uid 1000 --shell /bin/bash agent \
 && chown agent:agent /workspace
USER agent

# Long-lived by design: one container serves a whole session's commands, so the
# image's lifetime is not the session's. Accumulated pip installs, /tmp files and
# caches die with the container; the project files do not, because they are on
# the host.
CMD ["sleep", "infinity"]