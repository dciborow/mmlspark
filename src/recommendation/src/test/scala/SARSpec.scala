// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame

import scala.language.existentials

class SARSpec extends RankingTestBase with EstimatorFuzzing[SAR] {
  override def testObjects(): List[TestObject[SAR]] = {
    List(
      new TestObject(new SAR()
        .setUserCol(customerIndex.getOutputCol)
        .setItemCol(itemIndex.getOutputCol)
        .setRatingCol(ratingCol)
        .setAutoIndex(false), transformedDf)
    )
  }

  test("Movie Lens 1mil - SAR") {
    val ratings = movieLensSmall
    val customerId = "UserId"
    val itemId = "MovieId"

    val customerIndex = new StringIndexer()
      .setInputCol(customerId)
      .setOutputCol("customerID")

    val itemsIndex = new StringIndexer()
      .setInputCol(itemId)
      .setOutputCol("itemID")

    val pipeline = new Pipeline()
      .setStages(Array(customerIndex, itemsIndex))

    val pipelineModel: PipelineModel = pipeline.fit(ratings)
    val transformedDf = pipelineModel.transform(ratings)

    val threshold = 1
    val similarityFunction = "jacccard"

    val algo = new SAR()
      .setUserCol(customerIndex.getOutputCol)
      .setItemCol(itemsIndex.getOutputCol)
      .setRatingCol("Rating")
      .setSupportThreshold(threshold)
      .setSimilarityFunction(similarityFunction)
      .setActivityTimeFormat("EEE MMM dd HH:mm:ss Z yyyy")
      .setAutoIndex(false)

    val evaluator: RankingEvaluator = new RankingEvaluator()
      .setK(5)
      .setNItems(10)

    val adapter: RankingAdapter = new RankingAdapter()
      .setK(evaluator.getK)
      .setRecommender(algo)

    val output = adapter.fit(transformedDf).transform(transformedDf)

    val metrics = Array("ndcgAt", "fcp", "mrr")

    metrics.foreach(f => println(f + ": " + evaluator.setMetricName(f).evaluate(output)))
  }

  test("Movie Lens 1mil - ALS") {
    val ratings = movieLensSmall
    val customerId = "UserId"
    val itemId = "MovieId"

    val customerIndex = new StringIndexer()
      .setInputCol(customerId)
      .setOutputCol("customerID")

    val itemsIndex = new StringIndexer()
      .setInputCol(itemId)
      .setOutputCol("itemID")

    val pipeline = new Pipeline()
      .setStages(Array(customerIndex, itemsIndex))

    val pipelineModel: PipelineModel = pipeline.fit(ratings)
    val transformedDf = pipelineModel.transform(ratings)

    val algo = new ALS()
      .setUserCol(customerIndex.getOutputCol)
      .setItemCol(itemsIndex.getOutputCol)
      .setRatingCol("Rating")

    val evaluator: RankingEvaluator = new RankingEvaluator()
      .setK(5)
      .setNItems(10)

    val adapter: RankingAdapter = new RankingAdapter()
      .setK(evaluator.getK)
      .setRecommender(algo)

    val output = adapter.fit(transformedDf).transform(transformedDf)

    val metrics = Array("ndcgAt", "fcp", "mrr")

    metrics.foreach(f => println(f + ": " + evaluator.setMetricName(f).evaluate(output)))

  }

  private[spark] lazy val movieLensSmall: DataFrame = session.read
    .option("header", "true") //reading the headers
    .option("inferSchema", "true") //reading the headers
    .csv("./src/recommendation/src/test/resources/movielens1m.csv").na.drop

  override def reader: SAR.type = SAR

  override def modelReader: SARModel.type = SARModel
}
