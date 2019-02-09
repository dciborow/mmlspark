// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, Matrix => BM}
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.microsoft.ml.spark.schema.DatasetExtensions
import org.apache.spark.ml
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.ml.param._
import org.apache.spark.ml.recommendation.{RecommendationParams, Constants => C}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Pipeline}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.mllib.linalg.{DenseVector, Matrices, SparseMatrix}
import org.apache.spark.sql.functions.{col, collect_list, sum, udf, _}
import org.apache.spark.sql.types.{IntegerType, StructType, _}
import org.apache.spark.sql.{DataFrame, Dataset}

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

  /** @group getParam */
  def getStartTimeFormat: String = $(startTimeFormat)

  /** @group getParam */
  def getActivityTimeFormat: String = $(activityTimeFormat)

  /** @group getParam */
  def getTimeDecayCoeff: Int = $(timeDecayCoeff)

  /** @group getParam */
  def getAutoIndex: Boolean = $(autoIndex)

  /** @group getParam */
  def getAllowSeedItemsInRecommendations: Boolean = $(allowSeedItemsInRecommendations)

  override def fit(dataset: Dataset[_]): SARModel = {

    //Calculate user and item affinity matricies
    val (userItemAffinityMatrix, itemItemAffinityMatrix) =
      if (getAutoIndex) calculateIndexAndAffinities(dataset)
      else (calculateUserItemAffinities(dataset), calculateItemItemAffinities(dataset))

    new SARModel(uid)
      .setUserDataFrame(userItemAffinityMatrix)
      .setItemDataFrame(itemItemAffinityMatrix)
      .setParent(this)
      .setSupportThreshold(getSupportThreshold)
      .setItemCol(getItemCol)
      .setUserCol(getUserCol)
      .setAllowSeedItemsInRecommendations(getAllowSeedItemsInRecommendations)
      .setAutoIndex(getAutoIndex)
  }

  override def copy(extra: ParamMap): SAR = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  def this() = this(Identifiable.randomUID("SAR"))

  private[spark] def calculateIndexAndAffinities(dataset: Dataset[_]): (DataFrame, DataFrame) = {

    val getUserColTemp = DatasetExtensions.findUnusedColumnName(getUserCol)(dataset.columns.toSet)
    val getUserColOrg = DatasetExtensions.findUnusedColumnName(getUserCol)(dataset.columns.toSet)
    val getItemColTemp = DatasetExtensions.findUnusedColumnName(getItemCol)(dataset.columns.toSet)
    val getItemColOrg = DatasetExtensions.findUnusedColumnName(getItemCol)(dataset.columns.toSet)

    val customerIndex = new StringIndexer()
      .setInputCol(getUserCol)
      .setOutputCol(getUserColTemp)

    val itemIndex = new StringIndexer()
      .setInputCol(getItemCol)
      .setOutputCol(getItemColTemp)

    val pipelineModel = new Pipeline()
      .setStages(Array(customerIndex, itemIndex))
      .fit(dataset)

    //Index User and Item ID
    val indexedDataset = pipelineModel
      .transform(dataset)
      .withColumnRenamed(getUserCol, getUserColOrg)
      .withColumnRenamed(getItemCol, getItemColOrg)
      .withColumnRenamed(getUserColTemp, getUserCol)
      .withColumnRenamed(getItemColTemp, getItemCol)
      .cache

    def recoverId(stageIndex: Int) = {
      val map = pipelineModel
        .stages(stageIndex).asInstanceOf[StringIndexerModel]
        .labels
        .zipWithIndex
        .map(t => (t._2, t._1))
        .toMap

      val mapBC = dataset.sparkSession.sparkContext.broadcast(map)
      udf((iD: Integer) => mapBC.value.getOrElse[String](iD, "-1"))
    }

    //Calculate user item affinity matricies
    val userItemAffinityMatrix = calculateUserItemAffinities(indexedDataset)
      .withColumn(getUserColOrg, recoverId(0)(col(getUserCol)))

    //Calculate item item affinity matricies
    val itemItemAffinityMatrix = calculateItemItemAffinities(indexedDataset)
      .withColumn(getItemColOrg, recoverId(1)(col(getItemCol)))

    (userItemAffinityMatrix , itemItemAffinityMatrix )
  }

  private[spark] def calculateUserItemAffinities(dataset: Dataset[_]): DataFrame = {
    val referenceTime: Date = new SimpleDateFormat(getStartTimeFormat)
      .parse(get(startTime).getOrElse(Calendar.getInstance().getTime.toString))

    //Time Decay calculates the half life since the reference time
    val timeDecay = udf((time: String) => {
      val activityDate = new SimpleDateFormat(getActivityTimeFormat).parse(time)
      val timeDifference = (referenceTime.getTime - activityDate.getTime) / (1000 * 60)
      math.pow(2, -1.0 * timeDifference / (getTimeDecayCoeff * 24 * 60))
    })
    val blendWeights = udf((theta: Double, rho: Double) => theta * rho)

    val itemCount = dataset.select(col(getItemCol).cast(IntegerType)).groupBy().max(getItemCol).collect()(0).getInt(0)
    val numItems = dataset.sparkSession.sparkContext.broadcast(itemCount)

    val userItemAffinity = {
      if (dataset.columns.contains(getTimeCol) && dataset.columns.contains(getRatingCol)) {
        dataset.withColumn(C.affinityCol, blendWeights(timeDecay(col(getTimeCol)), col(getRatingCol)))
      }
      else if (dataset.columns.contains(getTimeCol)) {
        dataset.withColumn(C.affinityCol, timeDecay(col(getTimeCol)))
      }
      else if (dataset.columns.contains(getRatingCol)) {
        dataset.withColumn(C.affinityCol, col(getRatingCol))
      } else {
        val fillOne = udf((_: String) => 1)
        dataset.withColumn(C.affinityCol, fillOne(col(getUserCol)))
      }
    }.select(getUserCol, getItemCol, C.affinityCol)

    val columnsToArray = udf((itemId: Double, rating: Double) => Array(itemId, rating))

    val seqToArray = udf((r: Seq[Seq[Double]]) => {
      //Convert nested Set to Array
      val map = r.map(r => r.head.toInt -> r(1)).toMap
      val array = (0 to numItems.value).map(i => map.getOrElse(i, 0.0).toFloat).toArray
      array
    })

    userItemAffinity
      .groupBy(getUserCol, getItemCol).agg(sum(col(C.affinityCol)) as C.affinityCol)
      .withColumn("itemUserAffinityPair", columnsToArray(col(getItemCol), col(C.affinityCol)))
      .groupBy(getUserCol).agg(collect_list(col("itemUserAffinityPair")))
      .withColumn("flatList", seqToArray(col("collect_list(itemUserAffinityPair)")))
      .select(col(getUserCol).cast(IntegerType), col("flatList"))
  }

  private[spark] def calculateItemItemAffinities(dataset: Dataset[_]): DataFrame = {
    //Calculate Item Item Affinity Weights for all warm items
    val warmItemItemWeights = weightWarmItems(dataset).cache

    //Use Warm Item Item Weights to learn Cold Items if Item Features were provided
    optionalWeightColdItems(warmItemItemWeights)
  }

  private[spark] def weightWarmItems(dataset: Dataset[_]): DataFrame = {

    val spark = dataset.sparkSession

    dataset.cache
    val itemCounts = dataset
      .groupBy(col(getItemCol))
      .agg(countDistinct(col(getUserCol)))
      .collect()
      .map(r => r.get(0) -> r.getLong(1)).toMap

    val broadcastItemCounts = spark.sparkContext.broadcast(itemCounts)

    val calculateFeature = udf((itemID: Int, features: linalg.Vector) => {
      val (jaccardFlag, liftFlag) =
        if (getSimilarityFunction == "jaccard") (true, true)
        else if (getSimilarityFunction  == "lift") (false, true)
        else (false, false)

      def lift(countI: Double, countJ: Long, cooco: Double) = (cooco / (countI * countJ)).toFloat

      def jaccard(countI: Double, countJ: Long, cooco: Double) = (cooco / (countI + countJ - cooco)).toFloat

      val countI = features.apply(itemID)
      features.toArray.indices.map(i => {
        val countJ: Long = broadcastItemCounts.value.getOrElse(i, 0)
        val cooco = features.apply(i)
        if (!(cooco < getSupportThreshold)) {
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

    val itemCount = dataset
      .select(col(getItemCol).cast(IntegerType))
      .groupBy()
      .max(getItemCol)
      .collect()(0).getInt(0) + 1

    val userCount = dataset
      .select(col(getUserCol).cast(IntegerType))
      .groupBy()
      .max(getUserCol)
      .collect()(0).getInt(0) + 1

    val broadcastMatrix = {
      val userItemPairs: Array[(Int, Int, Double)] = dataset
        .groupBy(getUserCol, getItemCol)
        .agg(count(getItemCol))
        .select(col(getUserCol).cast(LongType), col(getItemCol).cast(LongType))
        .collect()
        .map(userItemPair => (userItemPair.getLong(0).toInt, userItemPair.getLong(1).toInt, 1.0))

      val sparse = SparseMatrix.fromCOO(userCount, itemCount, userItemPairs)
      val sparseBSM: BSM[Double] = new BSM[Double](sparse.values, sparse.numRows, sparse.numCols, sparse.colPtrs,
        sparse.rowIndices)
      spark.sparkContext.broadcast(sparseBSM)
    }

    val makeDenseFeatureVectors = udf((wrappedList: Seq[Int]) => {
      val vec = Array.fill[Double](userCount)(0.0)
      wrappedList.foreach(user => {
        vec(user) = 1.0
      })
      val sm = Matrices.dense(1, vec.length, vec).asML.toSparse
      val smBSM: BSM[Double] = new BSM[Double](sm.values, sm.numRows, sm.numCols, sm.colPtrs, sm.rowIndices)
      val value: BSM[Double] = smBSM * broadcastMatrix.value
      new DenseVector(value.toDense.toArray)
    })

    dataset
      .select(col(getItemCol).cast(IntegerType), col(getUserCol).cast(IntegerType))
      .groupBy(getItemCol)
      .agg(collect_list(getUserCol) as "collect_list")
      .withColumn(C.featuresCol, makeDenseFeatureVectors(col("collect_list")))
      .select(col(getItemCol).cast(IntegerType), col(C.featuresCol))
      .withColumn(C.itemAffinities, calculateFeature(col(getItemCol), col(C.featuresCol)))
      .select(col(getItemCol).cast(IntegerType), col(C.itemAffinities))
  }

  private[spark] def optionalWeightColdItems(warmItemItemWeights: DataFrame): DataFrame = {
    get(itemFeatures)
      .map(weightColdItems(_, warmItemItemWeights))
      .getOrElse(warmItemItemWeights)
  }

  private[spark] def weightColdItems(itemFeaturesDF: Dataset[_], warmItemItemWeights: DataFrame): DataFrame = {

    val sc = itemFeaturesDF.sparkSession

    val itemFeatureVectors = itemFeaturesDF.select(
      col(getItemCol).cast(LongType),
      col(C.tagId).cast(LongType),
      col(C.relevance)
    )
      .rdd.map(r => MatrixEntry(r.getLong(0), r.getLong(1), 1.0))

    val itemFeatureMatrix = new CoordinateMatrix(itemFeatureVectors)
      .toIndexedRowMatrix()
      .rows
      .map(index => (index.index.toInt, index.vector))

    val itemByItemFeatures =
      itemFeatureMatrix.cartesian(itemFeatureMatrix)
        .map{case ((item_id_i, item_i_vector), (item_id_j, item_j_vector)) =>
          //consider if not equal 0
          val productArray = (0 to item_i_vector.argmax + 1)
            .map(i => item_i_vector.apply(i) * item_j_vector.apply(i))
            .toArray

          (item_id_i, item_id_j, new ml.linalg.DenseVector(productArray))
        }

    val selectScore = udf((itemID: Integer, itemAffinities: Seq[Double]) => itemAffinities(itemID))

    val item_i = getItemCol + "_i"
    val item_j = getItemCol + "_j"

    val itemItemTrainingData = sc.createDataFrame(itemByItemFeatures)
      .toDF(item_i, item_j, C.featuresCol)
      .join(warmItemItemWeights, col(getItemCol) === col(item_i))
      .withColumn(C.label, selectScore(col(item_j), col(C.itemAffinities)))
      .select(item_i, item_j, C.featuresCol, C.label)

    val coldItems = itemItemTrainingData.where(col(C.label) < 0)

    val columnsToArray = udf((itemId: Double, rating: Double) => Array(itemId, rating))

    val coldWeightArray = DatasetExtensions.findUnusedColumnName("coldWeightArray")(itemItemTrainingData.columns.toSet)
    val coldItemItemWeights = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(1.0)
      .setElasticNetParam(1.0)
      .fit(itemItemTrainingData)
      .transform(coldItems)
      .withColumn(coldWeightArray, columnsToArray(col(item_j), col(C.prediction)))
      .groupBy(item_i).agg(collect_list(col(coldWeightArray)))
      .select(col(item_i), col("collect_list(" + coldWeightArray + ")").as(coldWeightArray))

    val mergeWarmAndColdItemAffinities = udf((itemAffinities: Seq[Float], coldItemAffinities: Seq[Seq[Double]]) => {
      coldItemAffinities.foreach(coldItem => {
        itemAffinities(coldItem.head.toInt) = coldItem(1).toFloat
      })
      itemAffinities
    })

    val outputTemp = DatasetExtensions.findUnusedColumnName("output")(itemItemTrainingData.columns.toSet)
    warmItemItemWeights
      .join(coldItemItemWeights, col(getItemCol) === col(item_i))
      .withColumn(outputTemp, mergeWarmAndColdItemAffinities(col(C.itemAffinities), col(coldWeightArray)))
      .select(col(getItemCol), col(outputTemp).as(C.itemAffinities))
  }


}

object SAR extends DefaultParamsReadable[SAR]

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
    ratingCol -> C.ratingCol, userCol -> C.userCol, itemCol -> C.itemCol, similarityFunction ->
      "jaccard", timeCol -> "time", startTimeFormat -> "EEE MMM dd HH:mm:ss Z yyyy", allowSeedItemsInRecommendations
      -> true, autoIndex -> false)
}
