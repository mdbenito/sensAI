import logging

import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing

from sensai.data_transformation import DataFrameTransformer, RuleBasedDataFrameTransformer, DataFrameTransformerChain, DFTNormalisation
from sensai.data_transformation.sklearn_transformer import ManualScaler
from sensai.featuregen import FeatureGenerator

log = logging.getLogger(__name__)


class TestDFTTransformerBasics:
    class TestDFT(DataFrameTransformer):
        def _fit(self, df: pd.DataFrame):
            pass

        def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"foo": [1, 2], "baz": [1, 2]})

    class RuleBasedTestDFT(RuleBasedDataFrameTransformer):
        def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    class TestFgen(FeatureGenerator):
        def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
            pass

        def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
            return pd.DataFrame({"foo": [1, 2], "baz": [1, 2]})

    testdf = pd.DataFrame({"foo": [1, 2], "bar": [1, 2]})

    @pytest.mark.parametrize("testdft", [TestDFT(), TestFgen().to_dft()])
    def test_basicProperties(self, testdft):
        assert not testdft.is_fitted()
        assert testdft.info()["changeInColumnNames"] is None
        testdft.fit(self.testdf)
        assert testdft.is_fitted()
        assert all(testdft.apply(self.testdf) == pd.DataFrame({"foo": [1, 2], "baz": [1, 2]}))
        assert testdft.info()["changeInColumnNames"] is not None

        # testing apply with no change in columns
        testdft.apply(pd.DataFrame({"foo": [1, 2], "baz": [1, 2]}))
        assert testdft.info()["changeInColumnNames"] == "none"

    def test_ruleBasedAlwaysFitted(self):
        assert self.RuleBasedTestDFT().is_fitted()

    def test_emptyChainFitted(self):
        assert DataFrameTransformerChain().is_fitted()

    def test_combinationFittedIffEachMemberFitted(self):
        # if one of the fgens is not fitted, the combination is not fitted either
        dftChain = DataFrameTransformerChain(self.TestDFT(), self.RuleBasedTestDFT())
        assert dftChain.dataFrameTransformers[1].is_fitted()
        assert not dftChain.is_fitted()
        dftChain.fit(self.testdf)
        assert dftChain.is_fitted()
        assert dftChain.dataFrameTransformers[0].is_fitted()

        # if all fgens are fitted, the combination is also fitted, even if fit was not called
        dftChain = DataFrameTransformerChain([self.RuleBasedTestDFT(), self.RuleBasedTestDFT()])
        assert dftChain.is_fitted()


class TestDFTNormalisation:
    class ContextAwareDFTNormalisation(DFTNormalisation):
        def __init__(self, rules, fit_values: np.ndarray):
            super().__init__(rules, require_all_handled=False)
            self.fitValues = fit_values
            self.fitContexts = []
            self.fitArgumentColumnOrders = []

        def _fit_values_for_rule(self, *, rule: DFTNormalisation.Rule, matching_columns, applicable_df: pd.DataFrame, ctx=None) -> np.ndarray:
            self.fitContexts.append(ctx)
            self.fitArgumentColumnOrders.append((list(matching_columns), list(applicable_df.columns)))
            return self.fitValues

    def test_multiColumnSingleRuleIndependent(self):
        arr = np.array([1, 5, 10])
        df = pd.DataFrame({"foo": arr, "bar": arr*100})
        dft = DFTNormalisation([DFTNormalisation.Rule(r"foo|bar", transformer=sklearn.preprocessing.MaxAbsScaler(), independent_columns=True)])
        df2 = dft.fit_apply(df)
        assert np.all(df2.foo == arr/10) and np.all(df2.bar == arr/10)

    def test_multiColumnSingleRuleNotIndependent(self):
        arr = np.array([1, 5, 10])
        df = pd.DataFrame({"foo": arr, "bar": arr*100})
        dft = DFTNormalisation([DFTNormalisation.Rule(r"foo|bar", transformer=sklearn.preprocessing.MaxAbsScaler(), independent_columns=False)])
        df2 = dft.fit_apply(df)
        assert np.all(df2.foo == arr/1000) and np.all(df2.bar == arr/10)

    @pytest.mark.parametrize("kwArgsAndException", [({}, True), ({"skip": True}, False)])
    def test_multiColumnSingleRuleUnspecified(self, kwArgsAndException):
        arr = np.array([1, 5, 10])
        df = pd.DataFrame({"foo": arr, "bar": arr*100})
        kwargs, mustThrowException = kwArgsAndException
        dft = DFTNormalisation([DFTNormalisation.Rule(r"foo|bar", **kwargs)])
        exceptionThrown = False
        try:
            dft.fit_apply(df)
        except Exception as e:
            log.info(f"Got exception: {e}")
            exceptionThrown = True
        assert exceptionThrown == mustThrowException

    def test_arrayValued(self):
        arr = np.array([1, 5, 10])
        df = pd.DataFrame({"foo": [arr, 2*arr, 10*arr]})
        dft = DFTNormalisation([DFTNormalisation.Rule(r"foo|bar", transformer=sklearn.preprocessing.MaxAbsScaler(), array_valued=True)])
        df2 = dft.fit_apply(df)
        assert np.all(df2.foo.iloc[0] == arr/100) and np.all(df2.foo.iloc[-1] == arr/10)

    def test_contextAwareFitValues(self):
        df = pd.DataFrame({"foo": [1.0, 2.0, 3.0], "bar": [2.0, 4.0, 6.0], "baz": [10.0, 20.0, 30.0]})
        fit_values = np.array([[0.0], [12.0]])
        dft = self.ContextAwareDFTNormalisation(
            [DFTNormalisation.Rule(r"foo|bar", transformer=sklearn.preprocessing.MaxAbsScaler(), independent_columns=False)],
            fit_values=fit_values)
        ctx = object()

        df2 = dft.fit_apply_with_context(df, ctx)

        assert dft.fitContexts == [ctx]
        assert np.allclose(df2["foo"].values, df["foo"].values / 12.0)
        assert np.allclose(df2["bar"].values, df["bar"].values / 12.0)
        assert np.allclose(df2["baz"].values, df["baz"].values)

    def test_contextAwareFitValuesReceiveMatchingColumnsInApplicableDataFrameOrder(self):
        df = pd.DataFrame({"foo": [1.0, 2.0, 3.0], "bar": [2.0, 4.0, 6.0]})
        dft = self.ContextAwareDFTNormalisation(
            [DFTNormalisation.Rule(r"foo|bar", transformer=sklearn.preprocessing.MaxAbsScaler(), independent_columns=True)],
            fit_values=np.array([[1.0, 2.0], [3.0, 4.0]]))

        dft.fit_with_context(df, object())

        assert dft.fitArgumentColumnOrders == [(["foo", "bar"], ["foo", "bar"])]

    def test_manualScalerCompatibility(self):
        df = pd.DataFrame({"foo": [1.0, 2.0, 3.0]})
        dft = DFTNormalisation([DFTNormalisation.Rule(r"foo", transformer=ManualScaler())])

        df2 = dft.fit_apply(df)

        assert np.allclose(df2["foo"].values, df["foo"].values)
