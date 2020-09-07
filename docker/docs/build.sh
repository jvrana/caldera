IMAGE="jvrana/caldera:docs"
DEST="gh-pages"
tmpfile=$(mktemp /tmp/caldera-build-docs.XXXXXX)
exec 3>"$tmpfile"
rm "$tmpfile"
echo "Building caldera documentation"
echo "IMAGE ID: $IMAGE"
docker run --cidfile $tmpfile $IMAGE /bin/bash -c "cd docs && make clean && make html"
CID=$(cat $tmpfile)
echo "CONTAINER ID: $CID"
docker cp $CID:/src/docs/build $DEST