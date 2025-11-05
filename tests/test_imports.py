"""
Test that all modules can be imported successfully
"""
import pytest


class TestImports:
    """Test that all DIAS modules can be imported"""

    def test_import_dias(self):
        """Test importing the main dias package"""
        import dias
        assert dias is not None

    def test_import_core_modules(self):
        """Test importing core modules"""
        from dias.core import hyper_graph
        assert hyper_graph is not None

    def test_import_storage_modules(self):
        """Test importing storage modules"""
        # Skip hyperdb for now due to stuf library compatibility issues with Python 3.12
        from dias.storage import processdbf
        assert processdbf is not None

    def test_import_scripts_modules(self):
        """Test importing scripts modules"""
        from dias.scripts import base_model
        from dias.scripts import simulate_model
        assert base_model is not None
        assert simulate_model is not None

    def test_import_visual_modules(self):
        """Test importing visualization modules"""
        from dias.visual import visualization
        assert visualization is not None

    def test_import_specific_functions(self):
        """Test importing specific functions from modules"""
        from dias.core.hyper_graph import compute_incident, sparse_graph

        assert compute_incident is not None
        assert sparse_graph is not None

    def test_star_imports(self):
        """Test that star imports work (as used in the original code)"""
        # This tests the imports as they're used in the example code
        try:
            import dias.scripts.base_model
            import dias.scripts.simulate_model
            # Check that the modules have some expected attributes
            assert hasattr(dias.scripts.base_model, '__all__') or len(dir(dias.scripts.base_model)) > 0
            assert hasattr(dias.scripts.simulate_model, '__all__') or len(dir(dias.scripts.simulate_model)) > 0
            assert True
        except ImportError as e:
            pytest.fail(f"Module imports failed: {e}")


class TestDependencies:
    """Test that required dependencies are available"""

    def test_numpy_import(self):
        """Test numpy import"""
        import numpy as np
        assert np is not None

    def test_scipy_import(self):
        """Test scipy import"""
        import scipy
        from scipy.sparse import csr_matrix
        assert scipy is not None
        assert csr_matrix is not None

    def test_networkx_import(self):
        """Test networkx import"""
        import networkx as nx
        assert nx is not None

    def test_numba_import(self):
        """Test numba import"""
        from numba import jit
        assert jit is not None

    def test_other_dependencies(self):
        """Test other required dependencies"""
        import dataset
        import seaborn
        import lxml
        import dbfread
        import googlemaps

        assert dataset is not None
        assert seaborn is not None
        assert lxml is not None
        assert dbfread is not None
        assert googlemaps is not None
