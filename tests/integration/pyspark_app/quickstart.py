from typing import List

import flytekit
import pandas as pd
from flytekitplugins.spark import Spark
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

from unionml import Dataset, Model

dataset = Dataset(name="digits_dataset", test_size=0.2, shuffle=True, targets=["target"])
model = Model(name="digits_classifier", init=LogisticRegression, dataset=dataset)

spark_task_config = Spark(
    # this configuration is applied to the spark cluster
    spark_conf={
        "spark.driver.memory": "1000M",
        "spark.executor.instances": "1",
        "spark.driver.cores": "4",
    }
)


@dataset.reader
def reader() -> pd.DataFrame:
    return load_digits(as_frame=True).frame


@model.trainer(task_config=spark_task_config)
def trainer(estimator: LogisticRegression, features: pd.DataFrame, target: pd.DataFrame) -> PipelineModel:
    sess = flytekit.current_context().spark_session

    df = pd.concat([features, target], axis=1)
    spark_df = sess.createDataFrame(df)

    # target column should match inputs, not what is set in model
    target_name = df.columns[-1]
    estimator.setLabelCol(target_name)

    # features column should match inputs, not what is set in model
    vec_assembler = VectorAssembler(
        inputCols=features.columns.tolist(),
        outputCol="features",
    )
    estimator.setFeaturesCol("features")

    return Pipeline(stages=[vec_assembler, estimator]).fit(spark_df)


@model.predictor(task_config=spark_task_config)
def predictor(estimator: PipelineModel, features: pd.DataFrame) -> List[float]:
    sess = flytekit.current_context().spark_session
    spark_features = sess.createDataFrame(features)

    predictions = estimator.transform(spark_features).toPandas()["prediction"].tolist()

    return [float(x) for x in predictions]


@model.evaluator
def evaluator(estimator: PipelineModel, features: pd.DataFrame, target: pd.DataFrame) -> float:
    return float(accuracy_score(target.squeeze(), predictor(estimator, features)))


@model.init
def init(hyperparameters: dict) -> LogisticRegression:
    return LogisticRegression()


if __name__ == "__main__":
    model_object, metrics = model.train(hyperparameters={"family": "multinomial", "maxIter": 10000})
    predictions = model.predict(features=load_digits(as_frame=True).frame.sample(5, random_state=42))
    print(model_object, metrics, predictions, sep="\n")

    # save model to a file, using joblib as the default serialization format
    model.save("/tmp/model_object/")
