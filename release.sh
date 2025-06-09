python3 -m build --wheel --sdist
twine upload dist/*
mkdocs gh-deploy --force --no-strict