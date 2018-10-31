// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package org.apache.spark.ml.tuning

import com.microsoft.ml.spark.RankingAdapterModel
import org.apache.spark.ml.Model
import org.apache.spark.ml.recommendation.HasRecommenderCols
import org.apache.spark.sql.{DataFrame, Dataset, RankingDataset}

class RankingTrainValidationSplit extends TrainValidationSplit with HasRecommenderCols {
  override def fit(dataset: Dataset[_]): RankingTrainValidationSplitModel = {
    val rankingDF = RankingDataset.toRankingDataSet[Any](dataset)
      .setUserCol(getUserCol)
      .setItemCol(getItemCol)
      .setRatingCol(getRatingCol)

    val model = super.fit(rankingDF)
    new RankingTrainValidationSplitModel("rtvs", model.bestModel, model.validationMetrics)
  }
}

class RankingTrainValidationSplitModel(
  override val uid: String,
  override val bestModel: Model[_],
  override val validationMetrics: Array[Double]) extends TrainValidationSplitModel(uid, bestModel, validationMetrics) {
  def recommendForAllUsers(k: Int): DataFrame =
    bestModel
      .asInstanceOf[RankingAdapterModel]
      .recommendForAllUsers(k)

  def recommendForAllItems(k: Int): DataFrame =
    bestModel
      .asInstanceOf[RankingAdapterModel]
      .recommendForAllItems(k)
}
