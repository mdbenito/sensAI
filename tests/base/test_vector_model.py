import pickle
import random
from copy import copy
from typing import Optional, Any

import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing

from sensai import InputOutputData
from sensai.data_transformation import DFTDRowFilterOnIndex, \
    InvertibleDataFrameTransformer, DFTDropNA, DataFrameTransformer, DFTNormalisation, DFTContextAwareMixin
from sensai.featuregen import FeatureGeneratorTakeColumns, FeatureGenerator
from sensai.vector_model import RuleBasedVectorRegressionModel, VectorRegressionModel, VectorClassificationModel, \
    FeatureTransformerFitContextInput


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


class ContextScalingDFT(DFTContextAwareMixin, DataFrameTransformer):
    def __init__(self):
        super().__init__()
        self.fitContexts = []
        self.factor = None

    def _fit(self, df: pd.DataFrame):
        raise AssertionError("Context-aware fit path was not used")

    def fit_with_context(self, df: pd.DataFrame, ctx: Any):
        self.fitContexts.append(ctx)
        self.factor = ctx["factor"]
        self._isFitted = True

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["foo2"] = df["foo2"] * self.factor
        return df


class DualPathRecordingDFT(DFTContextAwareMixin, DataFrameTransformer):
    def __init__(self):
        super().__init__()
        self.ordinaryFitCalls = 0
        self.contextFitCalls = 0

    def _fit(self, df: pd.DataFrame):
        self.ordinaryFitCalls += 1

    def fit_with_context(self, df: pd.DataFrame, ctx: Any):
        self.contextFitCalls += 1
        self._isFitted = True

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


class StrictSampleRuleBasedVectorModel(SampleRuleBasedVectorModel):
    def __init__(self):
        super().__init__()
        self._checkInputColumns = True


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


def half_factor_fit_context_factory(fit_state: FeatureTransformerFitContextInput):
    return {"factor": 0.5}


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
    fit_context = context_aware_dft.fitContexts[0]
    assert isinstance(fit_context, FeatureTransformerFitContextInput)
    assert np.allclose(fit_context.original_input["scale"].values, np.array([2.0, 4.0, 8.0]))
    assert np.allclose(fit_context.raw_transformed_input["foo"].values, np.array([1.0, 2.0, 4.0]))
    assert np.allclose(fit_context.feature_transformer_input["foo2"].values, np.array([10.0, 20.0, 40.0]))
    assert np.allclose(transformed_x["foo2"].values, np.array([1.25, 2.5, 5.0]))


def test_featureTransformerFitContextFactoryProvidesCustomContext():
    x = pd.DataFrame({"foo": [1.0, 2.0, 4.0], "scale": [2.0, 4.0, 8.0]})
    y = pd.DataFrame({"prediction": [0.0, 0.0, 0.0]})
    context_aware_dft = ContextScalingDFT()
    recorded_fit_states = []

    def fit_context_factory(fit_state: FeatureTransformerFitContextInput):
        recorded_fit_states.append(fit_state)
        return {"factor": 0.5}

    model = SampleVectorModel() \
        .with_feature_generator(ScaledRenameFeatureGenerator()) \
        .with_feature_transformers(context_aware_dft) \
        .with_feature_transformer_fit_context_factory(fit_context_factory)

    model.fit(x.copy(), y)

    transformed_x = model.compute_model_inputs(x.copy())
    assert len(recorded_fit_states) == 1
    fit_state = recorded_fit_states[0]
    assert np.allclose(fit_state.original_input["foo"].values, np.array([1.0, 2.0, 4.0]))
    assert np.allclose(fit_state.raw_transformed_input["scale"].values, np.array([2.0, 4.0, 8.0]))
    assert np.allclose(fit_state.feature_transformer_input["foo2"].values, np.array([10.0, 20.0, 40.0]))
    assert context_aware_dft.fitContexts == [{"factor": 0.5}]
    assert np.allclose(transformed_x["foo2"].values, np.array([5.0, 10.0, 20.0]))


def test_featureTransformerFitContextFactoryNotCalledWithoutContextAwareFeatureTransformers():
    x = pd.DataFrame({"foo": [1.0, 2.0, 4.0]})
    y = pd.DataFrame({"prediction": [0.0, 0.0, 0.0]})
    ordinary_dft = RecordingDFT()
    factory_calls = []

    def fit_context_factory(fit_state: FeatureTransformerFitContextInput):
        factory_calls.append(fit_state)
        return {"unused": True}

    model = SampleVectorModel() \
        .with_feature_transformers(ordinary_dft) \
        .with_feature_transformer_fit_context_factory(fit_context_factory)

    model.fit(x.copy(), y)

    assert ordinary_dft.fitCalls == 1
    assert factory_calls == []


def test_featureTransformerFitContextCanBeDisabledWithNone():
    x = pd.DataFrame({"foo": [1.0, 2.0, 4.0], "scale": [2.0, 4.0, 8.0]})
    y = pd.DataFrame({"prediction": [0.0, 0.0, 0.0]})
    dft = DualPathRecordingDFT()
    model = SampleVectorModel() \
        .with_feature_generator(ScaledRenameFeatureGenerator()) \
        .with_feature_transformers(dft) \
        .with_feature_transformer_fit_context_factory(lambda _: None)

    model.fit(x.copy(), y)

    assert dft.ordinaryFitCalls == 1
    assert dft.contextFitCalls == 0


def test_featureTransformerFitContextFactoryConfigurationIsPickled():
    x = pd.DataFrame({"foo": [1.0, 2.0, 4.0], "scale": [2.0, 4.0, 8.0]})
    y = pd.DataFrame({"prediction": [0.0, 0.0, 0.0]})
    model = SampleVectorModel() \
        .with_feature_generator(ScaledRenameFeatureGenerator()) \
        .with_feature_transformers(ContextScalingDFT()) \
        .with_feature_transformer_fit_context_factory(half_factor_fit_context_factory)

    loaded_model = pickle.loads(pickle.dumps(model))
    loaded_model.fit(x.copy(), y)

    transformed_x = loaded_model.compute_model_inputs(x.copy())
    assert np.allclose(transformed_x["foo2"].values, np.array([5.0, 10.0, 20.0]))


def test_ruleBasedModelFitAcceptsNoneOutputs():
    x = pd.DataFrame({"foo": [1.0, 2.0], "scale": [2.0, 4.0]})
    model = SampleRuleBasedVectorModel()

    model.fit(x.copy(), None)

    assert model.get_predicted_variable_names() == ["prediction"]
    assert model.get_model_input_variable_names() == ["foo", "scale"]


def test_ruleBasedModelStoresModelInputVariableNames():
    x = pd.DataFrame({"foo": [1.0, 2.0], "scale": [2.0, 4.0]})
    y = pd.DataFrame({"prediction": [0.0, 0.0]})
    model = SampleRuleBasedVectorModel()

    model.fit(x.copy(), y)

    assert model.get_model_input_variable_names() == ["foo", "scale"]
    model.predict(pd.DataFrame({"foo": [1.0, 2.0]}))


def test_ruleBasedModelCanStillOptIntoInputColumnChecking():
    x = pd.DataFrame({"foo": [1.0, 2.0], "scale": [2.0, 4.0]})
    y = pd.DataFrame({"prediction": [0.0, 0.0]})
    model = StrictSampleRuleBasedVectorModel()

    model.fit(x.copy(), y)

    assert model.get_model_input_variable_names() == ["foo", "scale"]
    with pytest.raises(Exception, match="expected columns \\['foo', 'scale'\\]"):
        model.predict(pd.DataFrame({"foo": [1.0, 2.0]}))
