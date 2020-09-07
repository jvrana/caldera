A docker minimal docker image for building documentation.

```
docker build . -f docker/docs/Dockerfile -t jvrana/caldera:docs
```

```
docker run --rm jvrana/caldera:docs /bin/bash -c "cd docs && make clean && make html"
```