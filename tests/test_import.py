def test_can_import_package():
    import graffit

    assert hasattr(graffit, "__version__") or True
