"""Unit tests for MoleculeStandardizer — no model files required."""


class TestStandardizer:
    def test_valid_smiles_returns_string(self, standardizer):
        result = standardizer.standardize_smiles("CCO")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_canonical_form(self, standardizer):
        # Both representations should canonicalize to the same SMILES
        s1 = standardizer.standardize_smiles("OCC")
        s2 = standardizer.standardize_smiles("CCO")
        assert s1 == s2

    def test_invalid_smiles_returns_none(self, standardizer):
        result = standardizer.standardize_smiles("not_a_smiles!!!")
        assert result is None

    def test_empty_string_returns_none(self, standardizer):
        result = standardizer.standardize_smiles("")
        assert result is None

    def test_largest_fragment_selected(self, standardizer):
        # Salt: sodium chloride fragment should be discarded, keep the organic part
        # Use aspirin sodium salt as example
        result = standardizer.standardize_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        assert result is not None
        # Should not contain the Na ion
        assert "Na" not in result

    def test_sample_smiles_all_valid(self, standardizer, sample_smiles):
        results = [standardizer.standardize_smiles(s) for s in sample_smiles]
        assert all(r is not None for r in results)
