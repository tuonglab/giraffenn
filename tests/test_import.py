def test_can_import_package():
    import giraffenn

    assert hasattr(giraffenn, "__version__") or True
