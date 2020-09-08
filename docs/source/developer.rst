===============
Developer Notes
===============

Tests
-----

Running tests
^^^^^^^^^^^^^

.. code-block:: shell

    pytest

Running doc-tools tests
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    pytest -E doc-tools

.. code-block:: python

    @pytest.mark.env('doc-tools')
    def test_foo():
        pass # only run if -E doc-tools provided in pytest

Running dockerized tests
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    # build image
    CACHE_IMAGE="username/caldera:latest-tests"
    docker build . -f docker/cpu/Dockerfile -t $CACHE_IMAGE

    # run pytest in container
    tmpfile=$(mktemp /tmp/caldera-build-docs.XXXXXX)
    exec 3>"$tmpfile"
    rm "$tmpfile"
    docker run --cidfile $tmpfile $CACHE_IMAGE /bin/bash -c "pytest --html pytest-report.html"
    CID=$(cat $tmpfile)
    docker cp $CID:/src/pytest-report.html .


Documentation
-------------

Build documentation
^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    make docs

Dockerized build
^^^^^^^^^^^^^^^^

.. code-block:: shell

    DOCKER_BUILDKIT=1 docker build \
    --cache-from $CACHE_IMAGE \
    master \
    -f master/docker/docs/Dockerfile \
    -t $CACHE_IMAGE

    ./master/docker/docs/build.sh $CACHE_IMAGE _ghpages

Converting ipynb to rst
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    python docs/_tools/nb_to_doc -f <nb_path> -o <out_dir>

