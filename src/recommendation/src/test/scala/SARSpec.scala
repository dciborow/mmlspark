// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark

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

  override def reader: SAR.type = SAR

  override def modelReader: SARModel.type = SARModel
}
