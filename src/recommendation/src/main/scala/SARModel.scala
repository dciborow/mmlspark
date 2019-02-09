// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import org.apache.spark.ml.Model
import org.apache.spark.ml.param.{DataFrameParam, ParamMap}
import org.apache.spark.ml.recommendation.BaseRecommendationModel
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

/** SAR Model
  *
  * @param uid The id of the module
  */
@InternalWrapper
class SARModel(override val uid: String) extends Model[SARModel]
  with BaseRecommendationModel with Wrappable with SARParams with ComplexParamsWritable {

  /** @group setParam */
  def setUserDataFrame(value: DataFrame): this.type = set(userDataFrame, value)

  val userDataFrame = new DataFrameParam(this, "userDataFrame", "Time of activity")

  /** @group getParam */
  def getUserDataFrame: DataFrame = $(userDataFrame)

  /** @group setParam */
  def setItemDataFrame(value: DataFrame): this.type = set(itemDataFrame, value)

  val itemDataFrame = new DataFrameParam(this, "itemDataFrame", "Time of activity")

  /** @group getParam */
  def getItemDataFrame: DataFrame = $(itemDataFrame)

  def this() = this(Identifiable.randomUID("SARModel"))

  override def recommendForAllItems(k: Int): DataFrame = {
    recommendForAllItems($(rank), $(userDataFrame), $(itemDataFrame), k)
  }

  override def recommendForAllUsers(k: Int): DataFrame = {
    val recs =
      if ($(allowSeedItemsInRecommendations)) recommendForAllUsers(getUserDataFrame, getItemDataFrame, k)
      else recommendForAllUsersNoSeeds(k)

    if ($(autoIndex)) restoreIndex(recs) else recs
  }

  def restoreIndex(recs: DataFrame): DataFrame = {
    val itemMap = getItemDataFrame
      .select(getItemCol, getItemCol + "_org")
      .collect()
      .map(r => r.getInt(0) -> r.getString(1))
      .toMap

    val arrayTypeAfter = ArrayType(
      new StructType()
        .add(getItemCol, StringType)
        .add("rating", FloatType)
    )

    val restoreIndexUDF = udf((items: Seq[Int], ratings: Seq[Float]) => {
      items
        .zipWithIndex
        .map(p => (itemMap.getOrElse(p._1, "-1"), ratings.toList(p._2)))
    })

    recs
      .join(getUserDataFrame, getUserCol)
      .withColumn("recs",
        (restoreIndexUDF(col("recommendations." + getItemCol), col("recommendations.rating")) as "recs")
          .cast(arrayTypeAfter))
      .select(col(getUserCol + "_org") as getUserCol, col("recs") as "recommendations")
  }

  /**
    * Recommend Items for All users and filter seed items from recommendations
    *
    * Seed items are those interacted with by the user in the raw dataset
    *
    * @param k
    * @return
    */
  def recommendForAllUsersNoSeeds(k: Int): DataFrame = {

    val seenItems = udf((items: Seq[Float]) => {
      items.zipWithIndex
        .filter(_._1 > 0)
        .map(_._2)
    })

    val seenItemsCount = udf((items: Seq[Float]) => items.length)

    val items = getUserDataFrame
      .select(col(getUserCol), seenItems(col("flatList")) as "seenItems")
      .withColumn("seenItemsCount", seenItemsCount(col("seenItems")))
    items.cache.count

    val itemCountMax = items
      .select(col("seenItemsCount"))
      .groupBy()
      .max("seenItemsCount")
      .collect()(0).getInt(0)

    val filterScore = udf((items: Seq[Int], ratings: Seq[Float], seenItems: Seq[Int]) => {
      items
        .zipWithIndex
        .filter(p => !seenItems.contains(p._1))
        .map(p => (p._1, ratings.toList(p._2)))
        .take(k)
    })

    val arrayType = ArrayType(
      new StructType()
        .add(getItemCol, IntegerType)
        .add("rating", FloatType)
    )

    recommendForAllUsers(getUserDataFrame, getItemDataFrame, k + itemCountMax)
      .join(items, getUserCol)
      .select(
        col(getUserCol),
        (filterScore(
          col("recommendations." + getItemCol),
          col("recommendations.rating"),
          col("seenItems")) as "recommendations")
          .cast(arrayType))
  }

  override def copy(extra: ParamMap): SARModel = {
    val copied = new SARModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transform($(rank), $(userDataFrame), $(itemDataFrame), dataset)
  }

  override def transformSchema(schema: StructType): StructType = {
    checkNumericType(schema, $(userCol))
    checkNumericType(schema, $(itemCol))
    schema
  }

  /**
    * Check whether the given schema contains a column of the numeric data type.
    *
    * @param colName column name
    */
  private def checkNumericType(
    schema: StructType,
    colName: String,
    msg: String = ""): Unit = {
    val actualDataType = schema(colName).dataType
    val message = if (msg != null && msg.trim.length > 0) " " + msg else ""
    require(actualDataType.isInstanceOf[NumericType], s"Column $colName must be of type " +
      s"NumericType but was actually of type $actualDataType.$message")
  }
}

object SARModel extends ComplexParamsReadable[SARModel]
