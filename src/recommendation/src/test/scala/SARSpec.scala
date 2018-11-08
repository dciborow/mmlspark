// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Row}

import scala.language.existentials

class SARSpec extends RankingTestBase with EstimatorFuzzing[SAR] {
  override def testObjects(): List[TestObject[SAR]] = {
    List(
      new TestObject(new SAR()
        .setUserCol(customerIndex.getOutputCol)
        .setItemCol(itemIndex.getOutputCol)
        .setRatingCol(ratingCol), transformedDf)
    )
  }

  override def reader: SAR.type = SAR

  override def modelReader: SARModel.type = SARModel

  test("SAR") {
    val customerId = userCol
    val itemId = itemCol

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

    val evaluator: RankingEvaluator = new RankingEvaluator()
      .setK(5)
      .setNItems(10)

    val adapter: RankingAdapter = new RankingAdapter()
      .setK(evaluator.getK)
      .setRecommender(algo)

    val output = adapter.fit(transformedDf).transform(transformedDf)

    val metrics = Array("ndcgAt", "fcp", "mrr")

    assert(evaluator.setMetricName("ndcgAt").evaluate(output) == 0.7168486344464263)
    assert(evaluator.setMetricName("fcp").evaluate(output) == 0.05000000000000001)
    assert(evaluator.setMetricName("mrr").evaluate(output) == 1.0)
  }

  val testFile: String = getClass.getResource("/demoUsage.csv").getPath
  val sim_count1: String = getClass.getResource("/sim_count1.csv").getPath
  val sim_lift1: String = getClass.getResource("/sim_lift1.csv").getPath
  val sim_jac1: String = getClass.getResource("/sim_jac1.csv").getPath
  val sim_count3: String = getClass.getResource("/sim_count3.csv").getPath
  val sim_lift3: String = getClass.getResource("/sim_lift3.csv").getPath
  val sim_jac3: String = getClass.getResource("/sim_jac3.csv").getPath
  val user_aff: String = getClass.getResource("/user_aff.csv").getPath
  val userpred_count3_userid_only: String = getClass.getResource("/userpred_count3_userid_only.csv").getPath
  val userpred_lift3_userid_only: String = getClass.getResource("/userpred_lift3_userid_only.csv").getPath
  val userpred_jac3_userid_only: String = getClass.getResource("/userpred_jac3_userid_only.csv").getPath

  private lazy val tlcSampleData: DataFrame = session.read
    .option("header", "true") //reading the headers
    .option("inferSchema", "true")
    .csv(testFile).na.drop.cache

  test("tlc test sim count1")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 1, "cooc", sim_count1, user_aff))

  test("tlc test sim lift1")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 1, "lift", sim_lift1, user_aff))

  test("tlc test sim jac1")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 1, "jaccard", sim_jac1, user_aff))

  test("tlc test sim count3")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 3, "cooc", sim_count3, user_aff))

  test("tlc test sim lift3")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 3, "lift", sim_lift3, user_aff))

  test("tlc test sim jac3")(
    SarTLCSpec.test_affinity_matrices(tlcSampleData, 3, "jaccard", sim_jac3, user_aff))

  test("tlc test userpred count3 userid only")(
    SarTLCSpec.test_product_recommendations(tlcSampleData, 3, "cooc", sim_count3, user_aff,
      userpred_count3_userid_only))

  test("tlc test userpred lift3 userid only")(
    SarTLCSpec.test_product_recommendations(tlcSampleData, 3, "lift", sim_lift3, user_aff,
      userpred_lift3_userid_only))

  test("tlc test userpred jac3 userid only")(
    SarTLCSpec.test_product_recommendations(tlcSampleData, 3, "jaccard", sim_jac3, user_aff,
      userpred_jac3_userid_only))

}

object SarTLCSpec {

  def test_affinity_matrices(tlcSampleData: DataFrame, threshold: Int, similarityFunction: String, simFile: String,
    user_aff: String): (SARModel, Broadcast[Map[Int, String]], Broadcast[Map[Int, String]]) = {

    val session = tlcSampleData.sparkSession
    val sparkContext = session.sparkContext

    val ratings = tlcSampleData
    val customerId = "userId"
    val itemId = "productId"

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

    val sar = new SAR()
      .setUserCol(customerIndex.getOutputCol)
      .setItemCol(itemsIndex.getOutputCol)
      .setTimeCol("timestamp")
      .setSupportThreshold(threshold)
      .setSimilarityFunction(similarityFunction)
      .setStartTime("2015/06/09T19:39:37")
      .setStartTimeFormat("yyyy/MM/dd'T'h:mm:ss")

    val model = sar.fit(transformedDf)

    val userMap = pipelineModel
      .stages(0).asInstanceOf[StringIndexerModel]
      .labels
      .zipWithIndex
      .map(t => (t._2, t._1))
      .toMap

    val itemMap = pipelineModel
      .stages(1).asInstanceOf[StringIndexerModel]
      .labels
      .zipWithIndex
      .map(t => (t._2, t._1))
      .toMap

    val userMapBC = sparkContext.broadcast(userMap)
    val itemMapBC = sparkContext.broadcast(itemMap)

    val recoverUser = udf((userID: Integer) => userMapBC.value.getOrElse[String](userID, "-1"))
    val recoverItem = udf((itemID: Integer) => itemMapBC.value.getOrElse[String](itemID, "-1"))

    val row: Row = model
      .getUserDataFrame
      .select(recoverUser(col("customerID")) as "customerID", col("flatList"))
      .filter(col("customerID") === "0003000098E85347")
      .select("flatList")
      .collect()(0)

    val list = row.getList(0)

    val map = list.toArray.zipWithIndex.map(t => (itemMap.getOrElse(t._2, "-1"), t._1)).toMap

    val userAff = session.read.option("header", "true").csv(user_aff).drop("_c0")
    val affinityRow: Row = userAff.collect()(0)
    val columnNames = userAff.schema.fieldNames

    val itemDF = model.getItemDataFrame
    val simMap = itemDF.collect().map(row => {
      val itemI = itemMap.getOrElse(row.getInt(0), "-1")
      val similarityVectorMap = row.getList(1).toArray.zipWithIndex.map(t => (itemMap.getOrElse(t._2, "-1"), t._1))
        .toMap
      (itemI -> similarityVectorMap)
    }).toMap
    itemDF.count

    val itemAff = session.read.option("header", "true").csv(simFile)
    itemAff.count
    itemAff.collect().foreach(row => {
      val itemI = row.getString(0)
      itemAff.drop("_c0").schema.fieldNames.foreach(itemJ => {
        val groundTrueScore = row.getAs[String](itemJ).toFloat
        val sparkSarScore = simMap.getOrElse(itemI, Map()).getOrElse(itemJ, "-1")
        assert(groundTrueScore == sparkSarScore)
      })
    })
    (model, userMapBC, itemMapBC)
  }

  def test_product_recommendations(tlcSampleData: DataFrame, threshold: Int, similarityFunction: String,
    simFile: String, user_aff: String,
    userPredFile: String): Unit = {
    val session = tlcSampleData.sparkSession
    val sparkContext = session.sparkContext

    val (model, userMapBC, itemMapBC) = test_affinity_matrices(tlcSampleData, threshold, similarityFunction, simFile,
      user_aff)

    val recoverUser = udf((userID: Integer) => userMapBC.value.getOrElse[String](userID, "-1"))
    val recoverItem = udf((itemID: Integer) => itemMapBC.value.getOrElse[String](itemID, "-1"))

    val usersProducts = tlcSampleData
      .filter(col("userId") === "0003000098E85347")
      .select("productId")
      .distinct()
      .collect()
      .map(_.getString(0))

    val usersProductsBC = sparkContext.broadcast(usersProducts)

    val users = model.recommendForAllUsers(5 + usersProducts.length)

    val filterScore = udf((items: Seq[Int], ratings: Seq[Float]) => {
      items.zipWithIndex
        .filter(p => {
          val itemId = itemMapBC.value.getOrElse[String](p._1, "-1")
          val bol = usersProductsBC.value.contains(itemId)
          !bol
        }).map(p => (p._1, ratings.toList(p._2)))
    })

    val temp = users
      .select(col("customerID"), filterScore(col("recommendations.itemID"), col("recommendations.rating")) as
        "recommendations")
      .select(col("customerID"), col("recommendations._1") as "itemID", col("recommendations._2") as "rating")
      .select(
        recoverUser(col("customerID")) as "customerID",
        recoverItem(col("itemID")(0)) as "rec1",
        recoverItem(col("itemID")(1)) as "rec2",
        recoverItem(col("itemID")(2)) as "rec3",
        recoverItem(col("itemID")(3)) as "rec4",
        recoverItem(col("itemID")(4)) as "rec5",
        col("rating")(0) as "score1",
        col("rating")(1) as "score2",
        col("rating")(2) as "score3",
        col("rating")(3) as "score4",
        col("rating")(4) as "score5")

    val rows = temp
      .filter(col("customerID") === "0003000098E85347")
      .take(1)

    val answer = session.read.option("header", "true").csv(userPredFile).collect()

    assert(rows(0).getString(0) == "0003000098E85347", "Assert Customer ID's Match")

    assert(rows(0).getString(1) == answer(0).getString(1))
    assert(rows(0).getString(2) == answer(0).getString(2))
    assert(rows(0).getString(3) == answer(0).getString(3))
    assert(rows(0).getString(4) == answer(0).getString(4))
    assert(rows(0).getString(5) == answer(0).getString(5))
    assert("%.3f".format(rows(0).getFloat(6)) == "%.3f".format(answer(0).getString(11).toFloat))
    assert("%.3f".format(rows(0).getFloat(7)) == "%.3f".format(answer(0).getString(12).toFloat))
    assert("%.3f".format(rows(0).getFloat(8)) == "%.3f".format(answer(0).getString(13).toFloat))
    assert("%.3f".format(rows(0).getFloat(9)) == "%.3f".format(answer(0).getString(14).toFloat))
    assert("%.3f".format(rows(0).getFloat(10)) == "%.3f".format(answer(0).getString(15).toFloat))
    ()
  }
}
