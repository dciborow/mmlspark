// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.DataFrame

import scala.language.existentials

class TrainValidRecommendSplitRec
  extends TestBase {

  test("testALS") {
    val dfRaw2: DataFrame = session
      .createDataFrame(Seq(
        ("11", "Movie 01", 4),
        ("11", "Movie 03", 1),
        ("11", "Movie 04", 5),
        ("11", "Movie 05", 3),
        ("11", "Movie 06", 4),
        ("11", "Movie 07", 1),
        ("11", "Movie 08", 5),
        ("11", "Movie 09", 3),
        ("22", "Movie 01", 4),
        ("22", "Movie 02", 5),
        ("22", "Movie 03", 1),
        ("22", "Movie 05", 3),
        ("22", "Movie 06", 4),
        ("22", "Movie 07", 5),
        ("22", "Movie 08", 1),
        ("22", "Movie 10", 3),
        ("33", "Movie 01", 4),
        ("33", "Movie 03", 1),
        ("33", "Movie 04", 5),
        ("33", "Movie 05", 3),
        ("33", "Movie 06", 4),
        ("33", "Movie 08", 1),
        ("33", "Movie 09", 5),
        ("33", "Movie 10", 3),
        ("44", "Movie 01", 4),
        ("44", "Movie 02", 5),
        ("44", "Movie 03", 1),
        ("44", "Movie 05", 3),
        ("44", "Movie 06", 4),
        ("44", "Movie 07", 5),
        ("44", "Movie 08", 1),
        ("44", "Movie 10", 3)))
      .toDF("customerIDOrg", "itemIDOrg", "rating")

    val ratings = dfRaw2.dropDuplicates()

    val customerIndex = new StringIndexer()
      .setInputCol("customerIDOrg")
      .setOutputCol("customerID")

    val ratingsIndex = new StringIndexer()
      .setInputCol("itemIDOrg")
      .setOutputCol("itemID")

    val pipeline = new Pipeline()
      .setStages(Array(customerIndex, ratingsIndex))

    val transformedDf = pipeline.fit(ratings).transform(ratings)

    val als = new ALS()
      .setUserCol(customerIndex.getOutputCol)
      .setRatingCol("rating")
      .setItemCol(ratingsIndex.getOutputCol)

    val paramGrid = new ParamGridBuilder()
      .addGrid(als.regParam, Array(1.0))
      .build()

    val evaluator = new RecommendationEvaluator()
      .setK(3)

    val tvRecommendationSplit = new TrainValidRecommendSplit()
      .setEstimator(als)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
      .setUserCol(customerIndex.getOutputCol)
      .setRatingCol("rating")
      .setItemCol(ratingsIndex.getOutputCol)

    val tvModel = tvRecommendationSplit.fit(transformedDf)

    val model = tvModel.getBestModel.asInstanceOf[ALSModel]

    val items = model.recommendForAllUsers(3)
    val users = model.recommendForAllItems(3)

    evaluator.setSaveAll(true)
    tvRecommendationSplit.fit(transformedDf)
    evaluator.printMetrics()
  }

}
