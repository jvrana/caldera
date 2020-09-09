IMAGE=$1
if [[ -z $IMAGE ]]; then
    echo "ERROR: IMAGE ID must be provided"
    exit 1
fi

DEST=$2
if [[ -z $DEST ]]; then
    echo "ERROR: DESTINATION path must be provided"
    exit 1
fi

tmpfile=$(mktemp /tmp/caldera-build-docs.XXXXXX)
exec 3>"$tmpfile"
rm "$tmpfile"
echo "Running tests"
echo "IMAGE ID: $IMAGE"
docker run --cidfile $tmpfile $IMAGE /bin/bash -c "pytest --html pytest-report.html"
CID=$(cat $tmpfile)
echo "CONTAINER ID: $CID"
docker cp $CID:/src/pytest-report.html $DEST