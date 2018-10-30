// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.param._
import org.apache.spark.ml.recommendation._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType, StructType}

import scala.collection.mutable
import scala.util.Random

trait RankingParams extends Params {
  val minRatingsPerUser: IntParam = new IntParam(this, "minRatingsPerUser",
    "min ratings for users > 0", ParamValidators.inRange(0, Integer.MAX_VALUE))

  val minRatingsPerItem: IntParam = new IntParam(this, "minRatingsPerItem",
    "min ratings for items > 0", ParamValidators.inRange(0, Integer.MAX_VALUE))

  /** @group setParam */
  def setMinRatingsPerUser(value: Int): this.type = set(minRatingsPerUser, value)

  /** @group setParam */
  def setMinRatingsPerItem(value: Int): this.type = set(minRatingsPerItem, value)

  /** @group getParam */
  def getMinRatingsPerUser: Int = $(minRatingsPerUser)

  /** @group getParam */
  def getMinRatingsPerItem: Int = $(minRatingsPerItem)

  val recommender = new EstimatorParam(this, "recommender", "estimator for selection", { x: Estimator[_] => true })

  /** @group getParam */
  def getRecommender: Estimator[_ <: Model[_]] = $(recommender)

  /** @group setParam */
  def setRecommender(value: Estimator[_ <: Model[_]]): this.type = set(recommender, value)

  val k: IntParam = new IntParam(this, "k", "number of items to recommend")

  /** @group getParam */
  def getK: Int = $(k)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

}

trait RankingFunctions extends RankingParams with HasRecommenderCols {

  private def filterByItemCount(dataset: Dataset[_], itemCol: String, userCol: String): DataFrame =
    dataset
      .groupBy(userCol)
      .agg(col(userCol), count(col(itemCol)).alias("nitems"))
      .where(col("nitems") >= $(minRatingsPerUser))
      .drop("nitems")
      .cache()

  private def filterByUserRatingCount(dataset: Dataset[_], itemCol: String, userCol: String): DataFrame =
    dataset
      .groupBy(itemCol)
      .agg(col(itemCol), count(col(userCol)).alias("nusers"))
      .where(col("nusers") >= $(minRatingsPerItem))
      .join(dataset, itemCol)
      .drop("nusers")
      .cache()

  def filterRatings(dataset: Dataset[_], itemCol: String, userCol: String): DataFrame =
    filterByUserRatingCount(dataset, itemCol, userCol)
      .join(filterByItemCount(dataset, itemCol, userCol), userCol)

  def prepareTestData(userColumn: String, itemColumn: String,
    validationDataset: DataFrame, recs: DataFrame,
    k: Int): Dataset[_] = {
    import org.apache.spark.sql.functions.{collect_list, rank => r}

    val perUserRecommendedItemsDF: DataFrame = recs
      .select(userColumn, "recommendations." + itemColumn)
      .withColumnRenamed(itemColumn, "prediction")

    val perUserActualItemsDF = if (validationDataset.columns.contains($(ratingCol))) {
      val windowSpec = Window.partitionBy(userColumn).orderBy(col($(ratingCol)).desc)

      validationDataset
        .select(userColumn, itemColumn, $(ratingCol))
        .withColumn("rank", r().over(windowSpec).alias("rank"))
        .where(col("rank") <= k)
        .groupBy(userColumn)
        .agg(col(userColumn), collect_list(col(itemColumn)))
        .withColumnRenamed("collect_list(" + itemColumn + ")", "label")
        .select(userColumn, "label")
    } else {
      val windowSpec = Window.partitionBy(userColumn).orderBy(col($(itemCol)).desc)

      validationDataset
        .select(userColumn, itemColumn)
        .withColumn("rank", r().over(windowSpec).alias("rank"))
        .where(col("rank") <= k)
        .groupBy(userColumn)
        .agg(col(userColumn), collect_list(col(itemColumn)))
        .withColumnRenamed("collect_list(" + itemColumn + ")", "label")
        .select(userColumn, "label")
    }
    val joined_rec_actual = perUserRecommendedItemsDF
      .join(perUserActualItemsDF, userColumn)
      .drop(userColumn)

    joined_rec_actual
  }

  def split(dataframe: Dataset[_], trainRatio: Double, itemCol: String, userCol: String, ratingCol: String):
  (DataFrame, DataFrame) = {
    val dataset = filterRatings(dataframe.dropDuplicates(), itemCol, userCol)

    if (dataset.columns.contains(ratingCol)) {
      val wrapColumn = udf((itemId: Double, rating: Double) => Array(itemId, rating))

      val sliceudf = udf(
        (r: mutable.WrappedArray[Array[Double]]) => r.slice(0, math.round(r.length * trainRatio).toInt))
      val shuffle = udf((r: mutable.WrappedArray[Array[Double]]) => Random.shuffle(r.toSeq))
      val dropudf = udf((r: mutable.WrappedArray[Array[Double]]) => r.drop(math.round(r.length * trainRatio).toInt))

      val testds = dataset
        .withColumn("itemIDRating", wrapColumn(col(itemCol), col(ratingCol)))
        .groupBy(col(userCol))
        .agg(collect_list(col("itemIDRating")))
        .withColumn("shuffle", shuffle(col("collect_list(itemIDRating)")))
        .withColumn("train", sliceudf(col("shuffle")))
        .withColumn("test", dropudf(col("shuffle")))
        .drop(col("collect_list(itemIDRating)")).drop(col("shuffle"))
        .cache()

      val popLeft = udf((r: mutable.WrappedArray[Double]) => r(0))
      val popRight = udf((r: mutable.WrappedArray[Double]) => r(1))

      val train = testds
        .select(userCol, "train")
        .withColumn("itemIdRating", explode(col("train")))
        .drop("train")
        .withColumn(itemCol, popLeft(col("itemIdRating")))
        .withColumn(ratingCol, popRight(col("itemIdRating")))
        .drop("itemIdRating")

      val test = testds
        .select(userCol, "test")
        .withColumn("itemIdRating", explode(col("test")))
        .drop("test")
        .withColumn(itemCol, popLeft(col("itemIdRating")))
        .withColumn(ratingCol, popRight(col("itemIdRating")))
        .drop("itemIdRating")

      (train, test)
    } else {
      val shuffle = udf((r: mutable.WrappedArray[Double]) => Random.shuffle(r.toSeq))
      val sliceudf = udf((r: mutable.WrappedArray[Double]) => r.slice(0, math.round(r.length * trainRatio).toInt))
      val dropudf = udf((r: mutable.WrappedArray[Double]) => r.drop(math.round(r.length * trainRatio).toInt))

      val testds = dataset
        .groupBy(col(userCol))
        .agg(collect_list(col(itemCol)))
        .withColumn("shuffle", shuffle(col("collect_list(" + itemCol + ")")))
        .withColumn("train", sliceudf(col("shuffle")))
        .withColumn("test", dropudf(col("shuffle")))
        .drop(col("collect_list(" + itemCol + ")")).drop(col("shuffle"))
        .cache()

      val train = testds
        .select(userCol, "train")
        .withColumn(itemCol, explode(col("train")))
        .drop("train")

      val test = testds
        .select(userCol, "test")
        .withColumn(itemCol, explode(col("test")))
        .drop("test")

      (train, test)
    }
  }

  def prepareTestData(validationDataset: DataFrame, recs: DataFrame, k: Int): Dataset[_] = {
    import org.apache.spark.sql.functions.{collect_list, rank => r}

    val perUserRecommendedItemsDF: DataFrame = recs
      .select(getUserCol, "recommendations." + getItemCol)
      .withColumnRenamed(getItemCol, "prediction")

    val perUserActualItemsDF = if (validationDataset.columns.contains(getRatingCol)) {
      val windowSpec = Window.partitionBy(getUserCol).orderBy(col(getRatingCol).desc)

      validationDataset
        .select(getUserCol, getItemCol, getRatingCol)
        .withColumn("rank", r().over(windowSpec).alias("rank"))
        .where(col("rank") <= k)
        .groupBy(getUserCol)
        .agg(col(getUserCol), collect_list(col(getItemCol)))
        .withColumnRenamed("collect_list(" + getItemCol + ")", "label")
        .select(getUserCol, "label")
    } else {
      val windowSpec = Window.partitionBy(getUserCol).orderBy(col(getItemCol).desc)

      validationDataset
        .select(getUserCol, getItemCol)
        .withColumn("rank", r().over(windowSpec).alias("rank"))
        .where(col("rank") <= k)
        .groupBy(getUserCol)
        .agg(col(getUserCol), collect_list(col(getItemCol)))
        .withColumnRenamed("collect_list(" + getItemCol + ")", "label")
        .select(getUserCol, "label")
    }
    val joined_rec_actual = perUserRecommendedItemsDF
      .join(perUserActualItemsDF, getUserCol)
      .drop(getUserCol)

    joined_rec_actual
  }
}

class RankingAdapter(override val uid: String)
  extends Estimator[RankingAdapterModel] with ComplexParamsWritable with RankingFunctions {

  def this() = this(Identifiable.randomUID("RecommenderAdapter"))

  /** @group getParam */
  override def getUserCol: String = getRecommender.asInstanceOf[Estimator[_] with PublicALSParams].getUserCol

  /** @group getParam */
  override def getItemCol: String = getRecommender.asInstanceOf[Estimator[_] with PublicALSParams].getItemCol

  /** @group getParam */
  override def getRatingCol: String = getRecommender.asInstanceOf[Estimator[_] with PublicALSParams].getRatingCol

  val mode: Param[String] = new Param(this, "mode", "recommendation mode")

  /** @group getParam */
  def getMode: String = $(mode)

  /** @group setParam */
  def setMode(value: String): this.type = set(mode, value)

  setDefault(mode -> "allUsers", k -> 10)

  def transformSchema(schema: StructType): StructType = {
    val model = getRecommender.asInstanceOf[ALS]
    getMode match {
      case "allUsers" => new StructType()
        .add("userCol", IntegerType)
        .add("recommendations", ArrayType(
          new StructType().add("itemCol", IntegerType).add("rating", FloatType))
        )
      case "allItems" => new StructType()
        .add("itemCol", IntegerType)
        .add("recommendations", ArrayType(
          new StructType().add("userCol", IntegerType).add("rating", FloatType))
        )
      case "normal" => new StructType()
        .add("userCol", IntegerType)
        .add("recommendations", ArrayType(
          new StructType().add("itemCol", IntegerType).add("rating", FloatType))
        )
    }
  }

  def fit(dataset: Dataset[_]): RankingAdapterModel = {
    new RankingAdapterModel()
      .setRecommenderModel(getRecommender.fit(dataset))
      .setMode(getMode)
      .setNItems(getK)
      .setUserCol(getUserCol)
      .setItemCol(getItemCol)
      .setRatingCol(getRatingCol)
  }

  override def copy(extra: ParamMap): RankingAdapter = {
    defaultCopy(extra)
  }

}

object RankingAdapter extends ComplexParamsReadable[RankingAdapter]

/**
  * Model from train validation split.
  *
  * @param uid Id.
  */
class RankingAdapterModel private[ml](val uid: String)
  extends Model[RankingAdapterModel] with ComplexParamsWritable with Wrappable with RankingFunctions {

  def recommendForAllUsers(i: Int): DataFrame = getRecommenderModel.asInstanceOf[ALSModel].recommendForAllUsers(3)

  def recommendForAllItems(i: Int): DataFrame = getRecommenderModel.asInstanceOf[ALSModel].recommendForAllItems(3)

  def this() = this(Identifiable.randomUID("RankingAdapterModel"))

  val recommenderModel = new TransformerParam(this, "recommenderModel", "recommenderModel", { x: Transformer => true })

  def setRecommenderModel(m: Model[_]): this.type = set(recommenderModel, m)

  def getRecommenderModel: Model[_] = $(recommenderModel).asInstanceOf[Model[_]]

  val mode: Param[String] = new Param(this, "mode", "recommendation mode")

  /** @group getParam */
  def getMode: String = $(mode)

  /** @group setParam */
  def setMode(value: String): this.type = set(mode, value)

  val nItems: IntParam = new IntParam(this, "nItems", "recommendation mode")

  /** @group getParam */
  def getNItems: Int = $(nItems)

  /** @group setParam */
  def setNItems(value: Int): this.type = set(nItems, value)

  val nUsers: IntParam = new IntParam(this, "nUsers", "recommendation mode")

  /** @group getParam */
  def getNUsers: Int = $(nUsers)

  /** @group setParam */
  def setNUsers(value: Int): this.type = set(nUsers, value)

  def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    val model = getRecommenderModel
    getMode match {
      case "allUsers" => {
        val recs = this.recommendForAllUsers(getNItems)
        prepareTestData(getUserCol, getItemCol, dataset.toDF(), recs, 10).toDF()
      }
      case "allItems" => {
        val recs = model.asInstanceOf[ALSModel].recommendForAllItems(getNUsers)
        prepareTestData(getItemCol, getUserCol, dataset.toDF(), recs, 10).toDF()
      }
      case "normal" => {
        val recs = SparkHelper.flatten(model.transform(dataset), 10, getItemCol, getUserCol)
        prepareTestData(getItemCol, getUserCol, dataset.toDF(), recs, 10).toDF()
      }
    }
  }

  def transformSchema(schema: StructType): StructType = {
    val model = getRecommenderModel
    getMode match {
      case "allUsers" => new StructType()
        .add("userCol", IntegerType)
        .add("recommendations", ArrayType(
          new StructType().add("itemCol", IntegerType).add("rating", FloatType))
        )
      case "allItems" => new StructType()
        .add("itemCol", IntegerType)
        .add("recommendations", ArrayType(
          new StructType().add("userCol", IntegerType).add("rating", FloatType))
        )
      case "normal" => new StructType()
        .add("userCol", IntegerType)
        .add("recommendations", ArrayType(
          new StructType().add("itemCol", IntegerType).add("rating", FloatType))
        )
    }
  }

  override def copy(extra: ParamMap): RankingAdapterModel = {
    defaultCopy(extra)
  }

}

object RankingAdapterModel extends ComplexParamsReadable[RankingAdapterModel]
