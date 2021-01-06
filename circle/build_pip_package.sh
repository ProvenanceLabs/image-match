#!/usr/bin/env bash

set -e

MAJOR=2
MINOR=0
PATCH=0

rm -f image-match-*.tar.gz
pip download --no-deps --extra-index-url https://pypi.fury.io/UmFoYTJTNACus1W8zjP8/360core/ image-match

for DOWNLOADED_FILE in image-match-${MAJOR}.${MINOR}.*.gz; do
  if [[ -e ${DOWNLOADED_FILE} ]]; then
      DOWNLOADED_FILE=${DOWNLOADED_FILE/image-match-${MAJOR}.${MINOR}./}
      DOWNLOADED_FILE=${DOWNLOADED_FILE/.tar.gz/}
      PATCH=$((DOWNLOADED_FILE+1))
  fi
  break
done

echo "__version__ = '$MAJOR.$MINOR.$PATCH'" > build.info
PYTHONPATH=. python setup.py sdist
PACKAGENAME=dist/image-match-${MAJOR}.${MINOR}.${PATCH}.tar.gz

exec 3>&1
STATUS=$(curl -w "%{http_code}" -o /dev/null -s -F package=@${PACKAGENAME} https://UmFoYTJTNACus1W8zjP8@push.fury.io/360core/)

if [ ${STATUS} != 200 ]
then
    echo curl response status: ${STATUS}
    exit ${STATUS}
fi
