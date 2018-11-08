// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.recommendation

import com.microsoft.ml.spark.Wrappable
import org.apache.spark.ml.param.shared.{HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.param.{IntParam, Param, ParamValidators, Params}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, _}

trait RecEvaluatorParams extends Wrappable
  with HasPredictionCol with HasLabelCol with hasK with ComplexParamsWritable

object SparkHelper {
  def flatten(ratings: Dataset[_], num: Int, dstOutputColumn: String, srcOutputColumn: String): DataFrame = {
    import ratings.sparkSession.implicits._

    val topKAggregator = new TopByKeyAggregator[Int, Int, Float](num, Ordering.by(_._2))
    val recs = ratings.as[(Int, Int, Float)].groupByKey(_._1).agg(topKAggregator.toColumn)
      .toDF("id", "recommendations")

    val arrayType = ArrayType(
      new StructType()
        .add(dstOutputColumn, IntegerType)
        .add("rating", FloatType)
    )
    recs.select(col("id").as(srcOutputColumn), col("recommendations").cast(arrayType))
  }
}

trait RecommendationParams extends ALSParams

trait BaseRecommendationModel extends Params with ALSModelParams with HasPredictionCol {

  private val id = Constants.idCol
  private val ratings = Constants.ratingCol + "s"
  private val recommendations = Constants.recommendations

  def getALSModel(uid: String,
    rank: Int,
    userFactors: DataFrame,
    itemFactors: DataFrame): ALSModel = {
    new ALSModel(uid, rank, userFactors, itemFactors)
  }

  def recommendForAllUsers(k: Int): DataFrame

  def recommendForAllUsers(userItemDataFrame: DataFrame, itemItemDataFrame: DataFrame, k: Int): DataFrame = {
    val userItemRDD = userItemDataFrame.rdd
      .flatMap(row =>
        row.getAs[Seq[Float]](1).zipWithIndex.map { case (list, index) => Row(row.getInt(0), index, list) })
      .map(item => MatrixEntry(item.getInt(0).toLong, item.getInt(1).toLong, item.getFloat(2).toDouble))

    val itemItemRDD = itemItemDataFrame.rdd
      .flatMap(row =>
        row.getAs[Seq[Float]](1).zipWithIndex.map { case (list, index) => Row(row.getInt(0), index, list) })
      .map(item => MatrixEntry(item.getInt(0).toLong, item.getInt(1).toLong, item.getFloat(2).toDouble))

    val userItemMatrix = new CoordinateMatrix(userItemRDD).toBlockMatrix().cache()
    val itemItemMatrix = new CoordinateMatrix(itemItemRDD).toBlockMatrix().cache()

    val userToItemMatrix = userItemMatrix
      .multiply(itemItemMatrix)
      .toIndexedRowMatrix()
      .rows.map(indexedRow => (indexedRow.index.toInt, indexedRow.vector))

    val orderAndTakeTopK = udf((vector: DenseVector) => {
      vector.toArray
        .zipWithIndex
        .map { case (list, index) => (index, list) }
        .sortWith(_._2 > _._2)
        .take(k)
    })

    val recommendationArrayType = ArrayType(
      new StructType()
        .add(getItemCol, IntegerType)
        .add(Constants.ratingCol, FloatType)
    )

    userItemDataFrame.sparkSession.createDataFrame(userToItemMatrix)
      .toDF(id, ratings).withColumn(recommendations, orderAndTakeTopK(col(ratings))).select(id, recommendations)
      .select(col(id).as(getUserCol), col(recommendations).cast(recommendationArrayType))
  }

  def transform(rank: Int, userDataFrame: DataFrame, itemDataFrame: DataFrame, dataset: Dataset[_]): DataFrame = {
    getALSModel(uid, rank,
      userDataFrame.withColumnRenamed(getUserCol, id).withColumnRenamed("flatList", "features"),
      itemDataFrame.withColumnRenamed(getItemCol, id).withColumnRenamed(Constants.itemAffinities, "features"))
      .setUserCol(getUserCol)
      .setItemCol(getItemCol)
      .setColdStartStrategy("drop")
      .transform(dataset)
  }
}

trait HasRecommenderCols extends Params {
  val userCol = new Param[String](this, "userCol", "Column of users")

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  def getUserCol: String = $(userCol)

  val itemCol = new Param[String](this, "itemCol", "Column of items")

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  def getItemCol: String = $(itemCol)

  val ratingCol = new Param[String](this, "ratingCol", "Column of ratings")

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  def getRatingCol: String = $(ratingCol)

}

trait hasK extends Params {
  val k: IntParam = new IntParam(this, "k", "number of items", ParamValidators.inRange(1, Integer.MAX_VALUE))

  /** @group getParam */
  def getK: Int = $(k)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  setDefault(k -> 10)
}

object Constants {
  val idCol = "id"
  val userCol = "user"
  val itemCol = "item"
  val ratingCol = "rating"
  val recommendations = "recommendations"
  val featuresCol = "featuresCol"
  val tagId = "tagId"
  val relevance = "relevance"
  val affinityCol = "affinity"
  val prediction = "prediction"
  val itemAffinities = "itemAffinities"
  val label = "label"
}
