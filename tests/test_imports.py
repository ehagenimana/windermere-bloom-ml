def test_package_import():
    import windermere_project  # noqa: F401


def test_core_deps_import():
    import pandas  # noqa: F401
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
    import yaml  # noqa: F401
    import requests  # noqa: F401
    import pyarrow  # noqa: F401
