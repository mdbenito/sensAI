import random
from copy import copy
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing

from sensai import InputOutputData
from sensai.data_transformation import DFTDRowFilterOnIndex, \
    InvertibleDataFrameTransformer, DFTDropNA, DataFrameTransformer, DFTNormalisation
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureGenerator
from sensai.vector_model import RuleBasedVectorRegressionModel, VectorRegressionModel, VectorClassificationModel


class FittableFgen(FeatureGenerator):

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        pass

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        return df


class ScaledRenameFeatureGenerator(FeatureGenerator):
    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        pass

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        return pd.DataFrame({"foo2": df["foo"] * 10}, index=df.index)


class FittableDFT(InvertibleDataFrameTransformer):

    def apply_inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _fit(self, df: pd.DataFrame):
        pass

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class RecordingDFT(DataFrameTransformer):
    def __init__(self):
        super().__init__()
        self.fitCalls = 0

    def _fit(self, df: pd.DataFrame):
        self.fitCalls += 1

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class OriginalInputFittingDFTNormalisation(DFTNormalisation):
    def __init__(self):
        super().__init__([DFTNormalisation.Rule(r"foo2", transformer=sklearn.preprocessing.MaxAbsScaler())])
        self.fitContexts = []

    def _fit_values_for_rule(self, *, rule: DFTNormalisation.Rule, matching_columns, applicable_df: pd.DataFrame, ctx=None) -> np.ndarray:
        self.fitContexts.append(ctx)
        return ctx.original_input[["scale"]].values


class SampleRuleBasedVectorModel(RuleBasedVectorRegressionModel):
    def __init__(self):
        super(SampleRuleBasedVectorModel, self).__init__(predicted_variable_names=["prediction"])

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"prediction": 1}, index=X.index)

    def get_predicted_variable_names(self):
        return ["prediction"]

    def is_regression_model(self) -> bool:
        return True


class SampleVectorModel(VectorRegressionModel):

    def _predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"prediction": 1}, index=x.index)

    def _fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame], weights: Optional[pd.Series] = None):
        pass


class FitRecordingSampleVectorModel(SampleVectorModel):
    def __init__(self):
        super().__init__()
        self.fitInputs = None

    def _fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame], weights: Optional[pd.Series] = None):
        self.fitInputs = x.copy()


@pytest.fixture()
def ruleBasedFgen():
    return FeatureGeneratorTakeColumns()


@pytest.fixture()
def fittableFgen():
    return FittableFgen()


@pytest.fixture()
def ruleBasedDFT():
    return DFTDRowFilterOnIndex()


@pytest.fixture()
def fittableDFT():
    return FittableDFT()


@pytest.fixture()
def ruleBasedModel():
    return SampleRuleBasedVectorModel()


@pytest.fixture()
def vectorModel():
    return SampleVectorModel()


testX = pd.DataFrame({"bar": [1, 2, 3]})
testY = pd.DataFrame({"prediction": [0, 0, 0]})


def fittedVectorModel():
    model = SampleVectorModel()
    model.fit(testX, testY)
    return model


class TestIsFitted:
    @pytest.mark.parametrize("model", [SampleRuleBasedVectorModel(), fittedVectorModel()])
    def test_isFittedWhenPreprocessorsRuleBased(self, model, ruleBasedDFT, ruleBasedFgen):
        assert model.is_fitted()
        model.with_feature_generator(ruleBasedFgen)
        model.with_input_transformers(ruleBasedDFT)
        assert model.is_fitted()

    @pytest.mark.parametrize("modelConstructor", [SampleRuleBasedVectorModel, fittedVectorModel])
    def test_isFittedWithFittableProcessors(self, modelConstructor, fittableDFT, fittableFgen):
        # is fitted after fit with model
        model = modelConstructor().with_raw_input_transformers(fittableDFT)
        assert not model.is_fitted()
        model.fit(testX, testY)
        assert model.is_fitted()
        
        # is fitted if DFT is fitted
        fittedDFT = copy(fittableDFT)
        fittedDFT.fit(testX)
        model = modelConstructor().with_raw_input_transformers(fittedDFT)
        assert model.is_fitted()

        # same for fgen
        model = modelConstructor().with_feature_generator(fittableFgen)
        assert not model.is_fitted()
        model.fit(testX, testY)
        assert model.is_fitted()

        fittedFgen = copy(fittableFgen)
        fittedFgen.fit(testX)
        model = modelConstructor().with_feature_generator(fittedFgen)
        assert model.is_fitted()

    def test_isFittedWithTargetTransformer(self, vectorModel, fittableDFT):
        assert not vectorModel.is_fitted()
        vectorModel.fit(testX, testY)
        assert vectorModel.is_fitted()
        vectorModel.with_target_transformer(fittableDFT)

        # test fitting separately
        assert not vectorModel.is_fitted()
        vectorModel.get_target_transformer().fit(testX)
        assert vectorModel.is_fitted()

        # test fitting together
        vectorModel = SampleVectorModel().with_target_transformer(FittableDFT())
        assert not vectorModel.is_fitted()
        vectorModel.fit(testX, testY)
        assert vectorModel.is_fitted()
        assert vectorModel.get_target_transformer().is_fitted()


def test_InputRowsRemovedByTransformer(irisClassificationTestCase):
    """
    Tests handling of case where the input generation process removes rows from the raw data
    """
    iodata = irisClassificationTestCase.data

    # create some random N/A values in one of the columns
    numNAValues = 20
    inputs = iodata.inputs.copy()
    rand = random.Random(42)
    fullIndices = list(range(len(inputs)))
    indices = rand.sample(fullIndices, numNAValues)
    inputs.iloc[:, 0].iloc[indices] = np.nan
    iodata = InputOutputData(inputs, iodata.outputs)
    expectedLength = len(iodata) - numNAValues

    class MyModel(VectorClassificationModel):
        def _fit_classifier(self, x: pd.DataFrame, y: pd.DataFrame, weights: Optional[pd.Series] = None):
            assert len(x) == expectedLength
            assert len(y) == expectedLength
            assert all(x.index.values == y.index.values)

        def _predict_class_probabilities(self, x: pd.DataFrame) -> pd.DataFrame:
            pass

    model = MyModel().with_raw_input_transformers(DFTDropNA())
    model.fit(iodata.inputs, iodata.outputs)


def test_lastFeatureTransformerIsNotAppliedTwiceDuringModelFit():
    x = pd.DataFrame({"foo": [1.0, 2.0, 4.0]})
    y = pd.DataFrame({"prediction": [0.0, 0.0, 0.0]})
    model = FitRecordingSampleVectorModel().with_feature_transformers(
        DFTNormalisation(
            [DFTNormalisation.Rule(r"foo", transformer=sklearn.preprocessing.MaxAbsScaler())],
            inplace=True
        )
    )

    model.fit(x.copy(), y)

    transformed_x = model.compute_model_inputs(x.copy())
    assert np.allclose(transformed_x["foo"].values, np.array([0.25, 0.5, 1.0]))
    assert np.allclose(model.fitInputs["foo"].values, transformed_x["foo"].values)


@pytest.mark.parametrize("modelConstructor", [SampleVectorModel, SampleRuleBasedVectorModel])
@pytest.mark.parametrize("nested_feature_transformer_chain", [False, True])
def test_featureTransformersFitWithOriginalTrainingInputContext(modelConstructor, nested_feature_transformer_chain):
    x = pd.DataFrame({"foo": [1.0, 2.0, 4.0], "scale": [2.0, 4.0, 8.0]})
    y = pd.DataFrame({"prediction": [0.0, 0.0, 0.0]})
    ordinary_dft = RecordingDFT()
    context_aware_dft = OriginalInputFittingDFTNormalisation()
    feature_transformers = (
        (ordinary_dft.chain(context_aware_dft),)
        if nested_feature_transformer_chain
        else (ordinary_dft, context_aware_dft)
    )
    model = modelConstructor() \
        .with_feature_generator(ScaledRenameFeatureGenerator()) \
        .with_feature_transformers(*feature_transformers)

    model.fit(x, y)

    transformed_x = model.compute_model_inputs(x.copy())
    assert ordinary_dft.fitCalls == 1
    assert len(context_aware_dft.fitContexts) == 1
    assert np.allclose(transformed_x["foo2"].values, np.array([1.25, 2.5, 5.0]))
