# Caldera Documentation

## Building

From within caldera main package, run the following (after installation)

```
make docs
```

Or 

```
cd docs
make clean
make html
```

## Running Tests

```
cd docs
make tests
```

## Executing and Converting .ipynb files

```
cd docs
python -m _tools/tools/nb_to_doc.py -f <path_to_ipynb> -o source/_nb_generated
```

Triggered