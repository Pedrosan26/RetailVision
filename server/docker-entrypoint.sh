#!/bin/sh
# docker-entrypoint.sh
#
# Brings the database schema up to date before starting the server, then
# execs whatever command the image was given.
#
# Running migrations here rather than as a manual step is deliberate: the
# server and its schema are versioned together, and a container that
# starts against an out-of-date database fails at request time with a
# confusing 500 rather than at startup with a clear error. `alembic
# upgrade head` is a no-op when the schema already matches.
set -e

echo "entrypoint: applying database migrations"
alembic upgrade head
echo "entrypoint: migrations applied, starting: $*"

exec "$@"
