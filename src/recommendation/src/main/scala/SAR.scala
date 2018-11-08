// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, Matrix => BM}
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.ml
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.param._
import org.apache.spark.ml.recommendation.{Constants, RecommendationParams}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Pipeline}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseVector, Matrices, SparseMatrix}
import org.apache.spark.sql.functions.{col, collect_list, sum, udf, _}
import org.apache.spark.sql.types.{IntegerType, StructType, _}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable
import scala.language.existentials

/** SAR
  *
  * https://aka.ms/reco-sar
  *
  * SAR is a fast scalable adaptive algorithm for personalized recommendations based on user transactions history and
  * items description. It produces easily explainable / interpretable recommendations and handles "cold item" and
  * "semi-cold user" scenarios.
  *
  * @param uid The id of the module
  */
@InternalWrapper
class SAR(override val uid: String) extends Estimator[SARModel] with SARParams with
  DefaultParamsWritable {

  /** @group getParam */
  def getSimilarityFunction: String = $(similarityFunction)

  /** @group getParam */
  def getTimeCol: String = $(timeCol)

  /** @group getParam */
  def getItemFeatures: DataFrame = $(itemFeatures)

  /** @group getParam */
  def getSupportThreshold: Int = $(supportThreshold)

  override def fit(dataset: Dataset[_]): SARModel = {

    //Create user and item affinity matricies
    val (userAffinityMatrix, itemAffinityMatrix) =
      if ($(autoIndex)) {

        val customerIndex = new StringIndexer()
          .setInputCol(getUserCol)
          .setOutputCol(getUserCol + "temp")

        val itemIndex = new StringIndexer()
          .setInputCol(getItemCol)
          .setOutputCol(getItemCol + "temp")

        val pipeline = new Pipeline()
          .setStages(Array(customerIndex, itemIndex))

        val pipelineModel = pipeline
          .fit(dataset)

        val indexedDataset = pipelineModel
          .transform(dataset)
          .withColumnRenamed(getItemCol, getItemCol + "_org")
          .withColumnRenamed(getUserCol, getUserCol + "_org")
          .withColumnRenamed(getItemCol + "temp", getItemCol)
          .withColumnRenamed(getUserCol + "temp", getUserCol)

        val userAffinityMatrix = calculateUserAffinities(indexedDataset)
        val itemAffinityMatrix = calculateItemAffinities(indexedDataset)

        val recoverUserId = {
          val userMap = pipelineModel
            .stages(0).asInstanceOf[StringIndexerModel]
            .labels
            .zipWithIndex
            .map(t => (t._2, t._1))
            .toMap

          val userMapBC = dataset.sparkSession.sparkContext.broadcast(userMap)
          udf((userID: Integer) => userMapBC.value.getOrElse[String](userID, "-1"))
        }
        val recoverItemId = {
          val itemMap = pipelineModel
            .stages(1).asInstanceOf[StringIndexerModel]
            .labels
            .zipWithIndex
            .map(t => (t._2, t._1))
            .toMap

          val itemMapBC = dataset.sparkSession.sparkContext.broadcast(itemMap)
          udf((itemID: Integer) => itemMapBC.value.getOrElse[String](itemID, "-1"))
        }

        (userAffinityMatrix.withColumn(getUserCol + "_org", recoverUserId(col(getUserCol))),
          itemAffinityMatrix.withColumn(getItemCol + "_org", recoverItemId(col(getItemCol))))
      }
      else (calculateUserAffinities(dataset), calculateItemAffinities(dataset))

    new SARModel(uid)
      .setUserDataFrame(userAffinityMatrix)
      .setItemDataFrame(itemAffinityMatrix)
      .setParent(this)
      .setSupportThreshold(getSupportThreshold)
      .setItemCol(getItemCol)
      .setUserCol(getUserCol)
      .setAllowSeedItemsInRecommendations($(allowSeedItemsInRecommendations))
      .setAutoIndex($(autoIndex))
  }

  override def copy(extra: ParamMap): SAR = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  def this() = this(Identifiable.randomUID("SAR"))

  def calculateItemAffinities(df: Dataset[_]): DataFrame = {
    SAR.calculateItemAffinities(getUserCol, getItemCol, getSupportThreshold, df, get(itemFeatures),
      getSimilarityFunction)
  }

  def calculateUserAffinities(dataset: Dataset[_]): DataFrame = {
    val referenceTime: Date = new SimpleDateFormat($(startTimeFormat))
      .parse(get(startTime).getOrElse(Calendar.getInstance().getTime.toString))

    val timeDecay = udf((time: String) => {
      val activityDate = new SimpleDateFormat($(activityTimeFormat)).parse(time)
      val timeDifference = (referenceTime.getTime - activityDate.getTime) / (1000 * 60)
      math.pow(2, -1.0 * timeDifference / ($(timeDecayCoeff) * 24 * 60))
    })
    val ratingWeight = udf((rating: Double) => rating)
    val blendWeights = udf((theta: Double, roe: Double) => theta * roe)

    val itemCount = dataset.select(col(getItemCol).cast(IntegerType)).groupBy().max(getItemCol).collect()(0).getInt(0)
    val numItems = dataset.sparkSession.sparkContext.broadcast(itemCount)

    val userAffinity = {
      if (dataset.columns.contains(getTimeCol) && dataset.columns.contains(getRatingCol)) {
        dataset.withColumn("affinity", blendWeights(timeDecay(col(getTimeCol)), ratingWeight(col(getRatingCol))))
          .select(getUserCol, getItemCol, "affinity")
      }
      else if (dataset.columns.contains(getTimeCol)) {
        dataset.withColumn("affinity", timeDecay(col(getTimeCol))).select(getUserCol, getItemCol, "affinity")
      }
      else if (dataset.columns.contains(getRatingCol)) {
        dataset.withColumn("affinity", ratingWeight(col(getRatingCol))).select(getUserCol, getItemCol, "affinity")
      } else {
        val fillOne = udf((_: String) => 1)
        dataset.withColumn("affinity", fillOne(col(getUserCol))).select(getUserCol, getItemCol, "affinity")
      }
    }

    val wrapColumn = udf((itemId: Double, rating: Double) => Array(itemId, rating))
    val flattenItems = udf((r: mutable.WrappedArray[mutable.WrappedArray[Double]]) => {
      val map = r.map(r => r(0).toInt -> r(1)).toMap
      val array = (0 to numItems.value).map(i => map.getOrElse(i, 0.0).toFloat).toArray
      array
    })

    userAffinity
      .groupBy(getUserCol, getItemCol).agg(sum(col("affinity")) as "affinity")
      .withColumn("itemUserAffinityPair", wrapColumn(col(getItemCol), col("affinity")))
      .groupBy(getUserCol)
      .agg(collect_list(col("itemUserAffinityPair")))
      .withColumn("flatList", flattenItems(col("collect_list(itemUserAffinityPair)")))
      .select(col(getUserCol).cast(IntegerType), col("flatList"))
  }
}

object SAR extends DefaultParamsReadable[SAR] {
  private val features = "features"

  def calculateItemAffinities(userColumn: String, itemColumn: String, supportThreshold: Int, transformedDf: Dataset[_],
    itemFeaturesDF: Option[DataFrame], similarityFunction: String): DataFrame = {

    def weightWarmItems(userColumn: String, itemColumn: String, supportThreshold: Int, transformedDf: Dataset[_],
      similarityFunction: String): DataFrame = {

      val spark = transformedDf.sparkSession

      transformedDf.cache
      val itemCounts = transformedDf
        .groupBy(col(itemColumn))
        .agg(countDistinct(col(userColumn)))
        .collect()
        .map(r => r.get(0) -> r.getLong(1)).toMap

      val broadcastItemCounts = spark.sparkContext.broadcast(itemCounts)

      val calculateFeature = udf((itemID: Int, features: linalg.Vector) => {
        val (jaccardFlag, liftFlag) =
          if (similarityFunction == "jaccard") (true, true)
          else if (similarityFunction == "lift") (false, true)
          else (false, false)

        def lift(countI: Double, countJ: Long, cooco: Double) = (cooco / (countI * countJ)).toFloat

        def jaccard(countI: Double, countJ: Long, cooco: Double) = (cooco / (countI + countJ - cooco)).toFloat

        val countI = features.apply(itemID)
        features.toArray.indices.map(i => {
          val countJ: Long = broadcastItemCounts.value.getOrElse(i, 0)
          val cooco = features.apply(i)
          if (!(cooco < supportThreshold)) {
            if (jaccardFlag)
              jaccard(countI, countJ, cooco)
            else if (liftFlag)
              lift(countI, countJ, cooco)
            else
              cooco.toFloat
          }
          else 0
        })
      })

      val itemCount = transformedDf
        .select(col(itemColumn).cast(IntegerType))
        .groupBy()
        .max(itemColumn)
        .collect()(0).getInt(0) + 1

      val userCount = transformedDf
        .select(col(userColumn).cast(IntegerType))
        .groupBy()
        .max(userColumn)
        .collect()(0).getInt(0) + 1

      val broadcastMatrix = {
        val userItemPairs: Array[(Int, Int, Double)] = transformedDf
          .groupBy(userColumn, itemColumn)
          .agg(count(itemColumn))
          .select(col(userColumn).cast(LongType), col(itemColumn).cast(LongType))
          .collect()
          .map(userItemPair => (userItemPair.getLong(0).toInt, userItemPair.getLong(1).toInt, 1.0))

        val sparse = SparseMatrix.fromCOO(userCount, itemCount, userItemPairs)
        val sparseBSM: BSM[Double] = new BSM[Double](sparse.values, sparse.numRows, sparse.numCols, sparse.colPtrs,
          sparse.rowIndices)
        spark.sparkContext.broadcast(sparseBSM)
      }

      val makeSparseMatrix = udf((wrappedList: mutable.WrappedArray[Int]) => {
        val vec = Array.fill[Double](userCount)(0.0)
        wrappedList.foreach(user => {
          vec(user) = 1.0
        })
        val sm = Matrices.dense(1, vec.length, vec).asML.toSparse
        val smBSM: BSM[Double] = new BSM[Double](sm.values, sm.numRows, sm.numCols, sm.colPtrs,
          sm.rowIndices)
        val value: BSM[Double] = smBSM * broadcastMatrix.value
        new DenseVector(value.toDense.toArray)
      })

      transformedDf
        .select(col(itemColumn).cast(IntegerType), col(userColumn).cast(IntegerType))
        .groupBy(itemColumn)
        .agg(collect_list(userColumn) as "collect_list")
        .withColumn(features, makeSparseMatrix(col("collect_list")))
        .select(col(itemColumn).cast(IntegerType), col(features))
        .select(col(itemColumn).cast(IntegerType), col(features))
        .withColumn("jaccardList", calculateFeature(col(itemColumn), col(features)))
        .select(
          col(itemColumn).cast(IntegerType),
          col("jaccardList")
        )

    }

    def weightColdItems(itemColumn: String, itemFeaturesDF: Dataset[_], jaccard: DataFrame): DataFrame = {
      val sc = itemFeaturesDF.sparkSession

      val itemFeatureVectors = itemFeaturesDF.select(
        col(itemColumn).cast(LongType),
        col("tagId").cast(LongType),
        col("relevance")
      )
        .rdd.map(r => MatrixEntry(r.getLong(0), r.getLong(1), 1.0))

      val itemFeatureMatrix = new CoordinateMatrix(itemFeatureVectors)
        .toBlockMatrix()
        .toIndexedRowMatrix()
        .rows.map(index => (index.index.toInt, index.vector))

      val pairs =
        itemFeatureMatrix.cartesian(itemFeatureMatrix)
          .map(row => {
            val vec1: linalg.Vector = row._1._2
            val vec2 = row._2._2
            //consider if not equal 0
            val productArray = (0 to vec1.argmax + 1).map(i => vec1.apply(i) * vec2.apply(i)).toArray
            (row._1._1, row._2._1, new ml.linalg.DenseVector(productArray))
          })

      val selectScore = udf((itemID: Integer, wrappedList: mutable.WrappedArray[Double]) => wrappedList(itemID))

      val itempairsDF = sc.createDataFrame(pairs)
        .toDF(itemColumn + "1", itemColumn + "2", features)
        .join(jaccard, col(itemColumn) === col(itemColumn + "1"))
        .withColumn("label", selectScore(col(itemColumn + "2"), col("jaccardList")))
        .select(itemColumn + "1", itemColumn + "2", features, "label")

      val cold = itempairsDF.where(col("label") < 0)

      val wrapColumn = udf((itemId: Double, rating: Double) => Array(itemId, rating))

      val coldData = new LinearRegression()
        .setMaxIter(10)
        .setRegParam(1.0)
        .setElasticNetParam(1.0)
        .fit(itempairsDF)
        .transform(cold)
        .withColumn("wrappedPrediction", wrapColumn(col(itemColumn + "2"), col("prediction")))
        .groupBy(itemColumn + "1")
        .agg(collect_list(col("wrappedPrediction")))
        .select(col(itemColumn + "1"), col("collect_list(wrappedPrediction)").as("wrappedPrediction"))

      val mergeScore = udf((jaccard: mutable.WrappedArray[Float], cold: mutable.WrappedArray[mutable
      .WrappedArray[Double]]) => {
        cold.foreach(coldItem => {
          jaccard.update(coldItem(0).toInt, coldItem(1).toFloat)
        })
        jaccard
      })

      val coldJaccard = jaccard
        .join(coldData, col(itemColumn) === col(itemColumn + "1"))
        .withColumn("output", mergeScore(col("jaccardList"), col("wrappedPrediction")))
        .select(col("itemID"), col("output").as("jaccardList"))
      coldJaccard
    }

    //Calculate Item Affinity Weights for all warm items
    val warmItemWeights =
      weightWarmItems(userColumn, itemColumn, supportThreshold, transformedDf, similarityFunction).cache

    //Use Optional Item Features to add Item Affinity Weights to cold items
    val itemWeights = itemFeaturesDF.map(weightColdItems(itemColumn, _, warmItemWeights))

    itemWeights.getOrElse(warmItemWeights)
  }

}

trait SARParams extends Wrappable with RecommendationParams {

  /** @group setParam */
  def setSimilarityFunction(value: String): this.type = set(similarityFunction, value)

  val similarityFunction = new Param[String](this, "similarityFunction",
    "Defines the similarity function to be used by " +
      "the model. Lift favors serendipity, Co-occurrence favors predictability, " +
      "and Jaccard is a nice compromise between the two.")

  /** @group setParam */
  def setTimeCol(value: String): this.type = set(timeCol, value)

  val timeCol = new Param[String](this, "timeCol", "Time of activity")

  /** @group setParam */
  def setItemFeatures(value: DataFrame): this.type = set(itemFeatures, value)

  val itemFeatures = new DataFrameParam(this, "itemFeatures", "Time of activity")

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  def setSupportThreshold(value: Int): this.type = set(supportThreshold, value)

  val supportThreshold = new IntParam(this, "supportThreshold", "Minimum number of ratings per item")

  def setStartTime(value: String): this.type = set(startTime, value)

  val startTime = new Param[String](this, "startTime", "Set time ")

  def setActivityTimeFormat(value: String): this.type = set(activityTimeFormat, value)

  val activityTimeFormat = new Param[String](this, "activityTimeFormat", "Time format for events, " +
    "default: yyyy/MM/dd'T'h:mm:ss")

  def setTimeDecayCoeff(value: Int): this.type = set(timeDecayCoeff, value)

  val timeDecayCoeff = new IntParam(this, "timeDecayCoeff", "Minimum number of ratings per item")

  def setStartTimeFormat(value: String): this.type = set(startTimeFormat, value)

  val startTimeFormat = new Param[String](this, "startTimeFormat", "Minimum number of ratings per item")

  def setAllowSeedItemsInRecommendations(value: Boolean): this.type = set(allowSeedItemsInRecommendations, value)

  val allowSeedItemsInRecommendations = new BooleanParam(this, "allowSeedItemsInRecommendations",
    "Allow seed items (items in the input or in the user history) to be returned as recommendation results. " +
      "True,False " +
      "Default: False")

  def setAutoIndex(value: Boolean): this.type = set(autoIndex, value)

  val autoIndex = new BooleanParam(this, "autoIndex",
    "Auto index customer and item ids if they are not ints" +
      "True,False " +
      "Default: False")

  setDefault(timeDecayCoeff -> 30, activityTimeFormat -> "yyyy/MM/dd'T'h:mm:ss", supportThreshold -> 4,
    ratingCol -> Constants.rating, userCol -> Constants.user, itemCol -> Constants.item, similarityFunction ->
      "jaccard", timeCol -> "time", startTimeFormat -> "EEE MMM dd HH:mm:ss Z yyyy", allowSeedItemsInRecommendations
      -> true, autoIndex -> false)
}
